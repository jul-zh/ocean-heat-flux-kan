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
    parser.add_argument("--config", type=str, default="configs/head_sweep.yaml")
    args = parser.parse_args()

    sweep_cfg = load_yaml(args.config)
    base_cfg = load_yaml(sweep_cfg["base_config"])

    results = []
    for head in sweep_cfg["heads"]:
        print(f"\n===== Running head: {head} =====")
        base_cfg["model"]["head"] = head
        _, best_val, test_metrics, experiment_dir = run_experiment(base_cfg, head)
        results.append(
            {
                "head": head,
                "best_val": best_val,
                "test_metrics": test_metrics,
                "experiment_dir": experiment_dir,
            }
        )

    out_path = Path(base_cfg["data"]["output_dir"]) / f"{base_cfg['data']['target_key'].lower()}_head_sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sweep results to {out_path}")


if __name__ == "__main__":
    main()
