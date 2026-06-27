"""Run every sampling method enabled by a refactored run config."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse CLI arguments and run all enabled sampling methods."""

    root = Path(__file__).resolve().parent

    from src.config import load_run_config
    from src.runner import run_enabled_methods

    parser = argparse.ArgumentParser(description="Run all sampling methods enabled in a run config.")
    parser.add_argument("--config", type=str, default="configs/example_run.json", help="Path to the run config JSON.")
    parser.add_argument("--tag", type=str, required=True, help="Results tag used under the results folder.")
    args = parser.parse_args()

    cfg = load_run_config(root / args.config)
    results = run_enabled_methods(root, cfg, tag=args.tag)
    for result in results:
        print(f"{result['method_folder']} complete:", result["results_csv"])


if __name__ == "__main__":
    main()
