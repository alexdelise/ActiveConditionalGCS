#!/usr/bin/env python3
"""Refresh the same LPIPS metric table used directly by weighted notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analyze_results"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import sd15_recovery_analysis as recovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = recovery.ensure_lpips_metrics(
        args.project_root,
        device=args.device,
        force=args.force,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
    )
    output_path = (
        args.project_root.resolve()
        / recovery.LPIPS_METRICS_RELATIVE_PATH
    )
    print(f"LPIPS table contains {len(table)} rows: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
