"""Build or validate a named dataset artifact from `datasets/config.json`."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse CLI arguments and build the requested dataset artifact."""

    root = Path(__file__).resolve().parent

    from src.config import RuntimeConfig, load_dataset_catalog
    from src.datasets import build_dataset

    parser = argparse.ArgumentParser(description="Build or validate a named dataset artifact.")
    parser.add_argument("--config", type=str, default="datasets/config.json", help="Path to the dataset catalog JSON.")
    parser.add_argument("--name", type=str, required=True, help="Exact dataset name to build.")
    parser.add_argument("--force", action="store_true", help="Rebuild the dataset even if it already exists.")
    args = parser.parse_args()

    catalog = load_dataset_catalog(root / args.config)
    if args.name not in catalog:
        raise KeyError(f"Dataset '{args.name}' is not defined in {args.config}.")

    dataset = build_dataset(root / "datasets", catalog[args.name], RuntimeConfig(), force=bool(args.force))
    print("Dataset ready:", dataset["dataset_path"])


if __name__ == "__main__":
    main()
