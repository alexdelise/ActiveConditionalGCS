"""Build or validate a named unweighted k-tilde artifact."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse CLI arguments and build the requested k-tilde artifact."""

    root = Path(__file__).resolve().parent

    from src.config import RuntimeConfig, load_ktilde_catalog
    from src.ktilde import build_ktilde

    parser = argparse.ArgumentParser(description="Build or validate a named k-tilde artifact.")
    parser.add_argument(
        "--config",
        type=str,
        default="ktilde/unweighted/config.json",
        help="Path to the k-tilde catalog JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ktilde/unweighted",
        help="Artifact directory.",
    )
    parser.add_argument("--name", type=str, required=True, help="Exact k-tilde name to build.")
    parser.add_argument("--force", action="store_true", help="Rebuild the k-tilde even if it already exists.")
    args = parser.parse_args()

    catalog = load_ktilde_catalog(root / args.config)
    if args.name not in catalog:
        raise KeyError(f"K-tilde '{args.name}' is not defined in {args.config}.")

    artifact_path, _ = build_ktilde(root / args.output_dir, catalog[args.name], RuntimeConfig(), force=bool(args.force))
    print("K-tilde ready:", artifact_path)


if __name__ == "__main__":
    main()
