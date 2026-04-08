from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocean_flux_kan.config import load_yaml
from ocean_flux_kan.train import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--target-key", type=str, default=None)
    parser.add_argument("--target-mode", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.target_key is not None:
        config["data"]["target_key"] = args.target_key
    if args.target_mode is not None:
        config["data"]["target_mode"] = args.target_mode
    config["model"]["head"] = args.head

    _, best_val, test_metrics, experiment_dir = run_experiment(config, args.head)
    result = {
        "head": args.head,
        "best_val": best_val,
        "test_metrics": test_metrics,
        "experiment_dir": experiment_dir,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
