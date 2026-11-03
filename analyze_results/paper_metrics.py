"""Read the per-run metrics used by the manuscript tables and forest plots."""

from pathlib import Path
import json

import numpy as np


def saved_reconstruction_metrics(run: Path, *, lpips_fallback: float = float("nan")) -> dict[str, float]:
    """Use saved metrics and normalize legacy unweighted best losses by m/n.

    As in the aggregate loaders, a finite LPIPS value in run_data.npz takes
    precedence over the cached sidecar value. No metric is reevaluated on a PNG.
    """
    run = Path(run)
    config = json.loads((run / "run_config.json").read_text())
    with np.load(run / "run_data.npz", allow_pickle=False) as saved:
        metrics = {key: float(saved[key].item()) for key in
                   ("psnr_db", "ssim", "bp_best_loss")}
        lpips = float(saved["lpips"].item()) if "lpips" in saved else float("nan")
        metrics["lpips"] = lpips if np.isfinite(lpips) else float(lpips_fallback)
        if not config["sampling"]["weighted_ls"]:
            n = int(saved["H"].item()) * int(saved["W"].item())
            metrics["bp_best_loss"] *= int(saved["m_coeffs"].item()) / n
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError(f"Missing or nonfinite reconstruction metrics in {run}")
    return metrics


def lpips_lookup(root: Path, *, unweighted: bool = False) -> dict[str, float]:
    """Read saved sidecar values when needed; weighted NPZ metrics are sufficient."""
    import pandas as pd
    path = Path(root) / ("unweighted/results/metrics/lpips.csv" if unweighted else "results/weighted/metrics/lpips.csv")
    if not path.is_file():
        return {}
    rows = pd.read_csv(path)
    paths = rows.artifact_relpath.str.replace("results/unweighted/", "unweighted/results/", regex=False)
    if paths.duplicated().any():
        raise ValueError("Duplicate LPIPS paths")
    return dict(zip(paths, rows.lpips))
