from __future__ import annotations

import os
from dataclasses import dataclass
from struct import unpack
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import replace_nan_with_zero


@dataclass
class GridSpec:
    height: int
    width: int


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[None, :, None, None]) / (self.std[None, :, None, None] + 1e-6)


def load_mask(mask_path: str, grid: GridSpec) -> np.ndarray:
    grid_size = grid.height * grid.width
    with open(mask_path, "rb") as f:
        binary_values = f.read(grid_size)
    mask = unpack("?" * grid_size, binary_values)
    return np.array(mask, dtype=np.float32).reshape(grid.height, grid.width)


def load_variable(path: str, grid: GridSpec) -> np.ndarray:
    arr = np.load(path)
    arr = arr.T
    arr = arr.reshape(arr.shape[0], grid.height, grid.width)
    return arr.astype(np.float32)


def load_all_data(data_dir: str, file_map: Dict[str, str], grid: GridSpec) -> Dict[str, np.ndarray]:
    data = {}
    for key, filename in file_map.items():
        path = os.path.join(data_dir, filename)
        data[key] = load_variable(path, grid)
        print(f"Loaded {key}: {data[key].shape}")
    return data


def make_lat_lon_channels(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    lat_vals = np.linspace(-90.0, 0.0, grid.height, dtype=np.float32)
    lon_vals = np.linspace(0.0, 80.0, grid.width, dtype=np.float32)

    lat_grid = np.repeat(lat_vals[:, None], grid.width, axis=1)
    lon_grid = np.repeat(lon_vals[None, :], grid.height, axis=0)

    lat_norm = lat_grid / 90.0
    lon_norm = (lon_grid - 40.0) / 40.0
    return lat_norm.astype(np.float32), lon_norm.astype(np.float32)


def compute_channel_stats_with_coords(
    data_dict: Dict[str, np.ndarray],
    channels: Sequence[str],
    mask_2d: np.ndarray,
    train_last_idx_exclusive: int,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ocean = mask_2d == 1
    means, stds = [], []

    for ch in channels:
        if ch == "LAT":
            vals = lat_grid[ocean]
        elif ch == "LON":
            vals = lon_grid[ocean]
        else:
            arr = data_dict[ch][:train_last_idx_exclusive]
            vals = arr[:, ocean]
            vals = vals[~np.isnan(vals)]

        mean = vals.mean()
        std = vals.std()
        means.append(mean)
        stds.append(std if std > 1e-6 else 1.0)

    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


class ERA5FluxDataset(Dataset):
    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        mask_2d: np.ndarray,
        channels: Sequence[str],
        dates: pd.DatetimeIndex,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        target_key: str,
        target_mode: str = "absolute",
        input_window: int = 30,
        horizons: Sequence[int] = (3, 7, 14),
        split_start: str | None = None,
        split_end: str | None = None,
        standardizer: Standardizer | None = None,
    ):
        self.data_dict = data_dict
        self.mask_2d = mask_2d.astype(np.float32)
        self.channels = list(channels)
        self.dates = dates
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.target_key = target_key
        self.target_mode = target_mode
        self.input_window = input_window
        self.horizons = list(horizons)
        self.max_h = max(horizons)
        self.standardizer = standardizer

        total_time = len(dates)
        all_candidate_indices = np.arange(input_window, total_time - self.max_h + 1)

        split_start_ts = pd.Timestamp(split_start) if split_start is not None else dates[0]
        split_end_ts = pd.Timestamp(split_end) if split_end is not None else dates[-1]

        self.indices: List[int] = []
        for t in all_candidate_indices:
            current_date = dates[t]
            if split_start_ts <= current_date <= split_end_ts:
                self.indices.append(t)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        t = self.indices[idx]

        x_list = []
        for ch in self.channels:
            if ch == "LAT":
                seq = np.repeat(self.lat_grid[None, :, :], self.input_window, axis=0)
            elif ch == "LON":
                seq = np.repeat(self.lon_grid[None, :, :], self.input_window, axis=0)
            else:
                seq = self.data_dict[ch][t - self.input_window : t]
            x_list.append(seq)
        x = np.stack(x_list, axis=1)

        current_target = self.data_dict[self.target_key][t - 1].copy()
        y_list = []
        for h in self.horizons:
            future_target = self.data_dict[self.target_key][t + h - 1]
            if self.target_mode == "delta":
                y_list.append(future_target - current_target)
            elif self.target_mode == "absolute":
                y_list.append(future_target)
            else:
                raise ValueError(f"Unsupported target_mode={self.target_mode}")
        y = np.stack(y_list, axis=0)

        if self.standardizer is not None:
            x = self.standardizer.transform(x)

        x = replace_nan_with_zero(x)
        y = replace_nan_with_zero(y)
        current_target = replace_nan_with_zero(current_target)

        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(y).float(),
            "current_target": torch.from_numpy(current_target).float(),
            "mask": torch.from_numpy(self.mask_2d).float(),
            "date": str(self.dates[t].date()),
        }


def build_dataloaders(config: dict):
    data_cfg = config["data"]
    train_cfg = config["train"]
    grid = GridSpec(height=data_cfg["grid"]["height"], width=data_cfg["grid"]["width"])

    data_dir = data_cfg["data_dir"]
    file_map = data_cfg["file_map"]
    mask = load_mask(os.path.join(data_dir, data_cfg["mask_filename"]), grid)
    data = load_all_data(data_dir, file_map, grid)
    dates = pd.date_range(start=data_cfg["date_range"]["start"], end=data_cfg["date_range"]["end"], freq="D")

    for key in data:
        assert data[key].shape[0] == len(dates), f"{key}: time mismatch"

    lat_grid, lon_grid = make_lat_lon_channels(grid)
    train_last_idx_exclusive = dates.get_loc(data_cfg["split"]["val_start"])

    x_mean, x_std = compute_channel_stats_with_coords(
        data_dict=data,
        channels=data_cfg["channels"],
        mask_2d=mask,
        train_last_idx_exclusive=train_last_idx_exclusive,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
    )
    standardizer = Standardizer(x_mean, x_std)

    ds_kwargs = dict(
        data_dict=data,
        mask_2d=mask,
        channels=data_cfg["channels"],
        dates=dates,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        target_key=data_cfg["target_key"],
        target_mode=data_cfg["target_mode"],
        input_window=data_cfg["input_window"],
        horizons=data_cfg["horizons"],
        standardizer=standardizer,
    )

    train_ds = ERA5FluxDataset(
        **ds_kwargs,
        split_start=str(pd.Timestamp(data_cfg["date_range"]["start"]) + pd.Timedelta(days=data_cfg["input_window"])),
        split_end=data_cfg["split"]["train_end"],
    )
    val_ds = ERA5FluxDataset(**ds_kwargs, split_start=data_cfg["split"]["val_start"], split_end=data_cfg["split"]["val_end"])
    test_ds = ERA5FluxDataset(**ds_kwargs, split_start=data_cfg["split"]["test_start"], split_end=data_cfg["split"]["test_end"])

    loader_kwargs = dict(batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    meta = {
        "mask": mask,
        "dates": dates,
        "standardizer": standardizer,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
    }
    return train_loader, val_loader, test_loader, meta
