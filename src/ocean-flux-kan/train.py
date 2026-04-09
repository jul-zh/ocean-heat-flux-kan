from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data import build_dataloaders
from .losses import get_loss, is_probabilistic_head
from .metrics import MetricTracker
from .models.model import HeatFluxForecastModel
from .utils import ensure_dir, resolve_device, set_seed


def build_model(config: dict, head_name: str) -> HeatFluxForecastModel:
    return HeatFluxForecastModel(
        in_channels=len(config["data"]["channels"]),
        hidden_dim=config["model"]["hidden_dim"],
        out_horizons=len(config["data"]["horizons"]),
        head_name=head_name,
        head_params=config["model"]["head_params"],
    )


def _make_scaler_tensors(target_mean: np.ndarray, target_std: np.ndarray, device: torch.device):
    mean = torch.tensor(target_mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor(target_std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return mean, std


def forward_and_loss(model, batch, criterion, device, probabilistic: bool, scaler_tensors):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    y_raw = batch["y_raw"].to(device)
    mask = batch["mask"].to(device)
    out = model(x)
    target_mean, target_std = scaler_tensors

    if probabilistic:
        mu = out["mu"]
        sigma = out["sigma"]
        loss = criterion(mu, sigma, y, mask)
        pred_denorm = mu * target_std + target_mean
        sigma_denorm = sigma * target_std
        return loss, pred_denorm, y_raw, sigma_denorm

    pred = out["pred"]
    loss = criterion(pred, y, mask)
    pred_denorm = pred * target_std + target_mean
    return loss, pred_denorm, y_raw, None


def train_one_epoch(model, loader, optimizer, criterion, device, horizons, probabilistic: bool, scaler_tensors):
    model.train()
    total_loss = 0.0
    tracker = MetricTracker(horizons=horizons, probabilistic=probabilistic)

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        loss, pred_denorm, y_raw, sigma_denorm = forward_and_loss(model, batch, criterion, device, probabilistic, scaler_tensors)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        tracker.update(pred_denorm.detach(), y_raw.detach(), batch["mask"].to(device), sigma=sigma_denorm.detach() if sigma_denorm is not None else None)

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, horizons, probabilistic: bool, scaler_tensors):
    model.eval()
    total_loss = 0.0
    tracker = MetricTracker(horizons=horizons, probabilistic=probabilistic)

    for batch in tqdm(loader, desc="eval", leave=False):
        loss, pred_denorm, y_raw, sigma_denorm = forward_and_loss(model, batch, criterion, device, probabilistic, scaler_tensors)
        total_loss += loss.item()
        tracker.update(pred_denorm, y_raw, batch["mask"].to(device), sigma=sigma_denorm)

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def _json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_experiment(config: dict, head_name: str) -> Tuple[Dict, Dict, Dict, str]:
    set_seed(config["seed"])
    train_loader, val_loader, test_loader, meta = build_dataloaders(config)
    device = resolve_device(config["train"]["device"])
    probabilistic = is_probabilistic_head(head_name)
    scaler_tensors = _make_scaler_tensors(meta["target_mean"], meta["target_std"], device)

    model = build_model(config, head_name).to(device)
    criterion = get_loss(head_name, config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    output_root = ensure_dir(config["data"]["output_dir"])
    experiment_dir = ensure_dir(output_root / f"{config['data']['target_key'].lower()}_{config['data']['target_mode']}_{head_name}")
    best_model_path = experiment_dir / "best_model.pt"
    history_path = experiment_dir / "history.json"
    config_path = experiment_dir / "resolved_config.json"

    best_val = float("inf")
    history = []

    for epoch in range(config["train"]["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, config["data"]["horizons"], probabilistic, scaler_tensors)
        val_metrics = evaluate(model, val_loader, criterion, device, config["data"]["horizons"], probabilistic, scaler_tensors)

        row = {
            "epoch": epoch + 1,
            **{f"train_{k}": float(v) for k, v in train_metrics.items()},
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        }
        history.append(row)

        print(f"\nEpoch {epoch + 1:03d}")
        print("Train:", train_metrics)
        print("Val:  ", val_metrics)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, config["data"]["horizons"], probabilistic, scaler_tensors)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=_json_safe)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    summary = {
        "head": head_name,
        "best_val_loss": float(best_val),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "best_model_path": str(best_model_path),
        "target_mean": [float(x) for x in meta["target_mean"]],
        "target_std": [float(x) for x in meta["target_std"]],
    }
    with open(experiment_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_safe)

    return history[-1] if history else {}, {"loss": float(best_val)}, test_metrics, str(experiment_dir)
