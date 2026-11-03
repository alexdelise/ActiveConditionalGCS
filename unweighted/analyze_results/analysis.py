"""Analyze unweighted reconstructions and generate figures."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analyze_results"))
import sd15_recovery_analysis as recovery
import sd15_cfg_ablation_analysis as cfg
import sd15_conditioning_experiment as christoffel

RATIOS = (0.00015625, 0.0003125, 0.000625, 0.00125, 0.0025, 0.005, 0.01)
MAIN_RATIOS = RATIOS
CFG = ("cfg1", "cfg3", "cfg5", "cfg7p5")
EXPERIMENTS = ("prompt_matched_in_range", "prompt_mismatched_in_range", "out_of_range")
SCENARIOS = ("prompt_matched", "prompt_mismatched", "out_of_range")
SAMPLING_CONDITIONS = (
    "k0",
    "k1_daytime_beach",
    "k2_sunset_beach",
    "k4_cat",
    "mcs",
    "inverse_square",
)
SAMPLING_LABELS = {
    "k0": r"$\widetilde\mu_{c_{\mathrm{uc}}}$",
    "k1_daytime_beach": r"$\widetilde\mu_{c_{\mathrm{db}}}$",
    "k2_sunset_beach": r"$\widetilde\mu_{c_{\mathrm{sb}}}$",
    "k4_cat": r"$\widetilde\mu_{c_{\mathrm{ca}}}$",
    "mcs": r"$\mu_{\mathrm{MCS}}$",
    "inverse_square": r"$\mu_{\mathrm{IS}}$",
}
RECOVERY_CONDITIONS = ("unprompted", "daytime_beach", "sunset_beach", "cat")
RECOVERY_LABELS = {
    "unprompted": r"$c_r=c_{\mathrm{uc}}$",
    "daytime_beach": r"$c_r=c_{\mathrm{db}}$",
    "sunset_beach": r"$c_r=c_{\mathrm{sb}}$",
    "cat": r"$c_r=c_{\mathrm{ca}}$",
}
METRICS = ("lpips", "bp_best_loss", "psnr_db", "ssim")
METRIC_HEADERS = {
    "lpips": r"LPIPS $\downarrow$",
    "psnr_db": r"PSNR (dB) $\uparrow$",
    "ssim": r"SSIM $\uparrow$",
    "bp_best_loss": r"Best Unweighted Loss $\downarrow$",
}

LAWS = SAMPLING_CONDITIONS
LOSS_LABEL = r"Best Unweighted Loss $\downarrow$"
SPATIAL_DIMENSION = 512 * 512


def convert_loss_convention(frame):
    """Convert legacy measurement-mean/raw-FFT losses, before aggregation.

    Preserve the historical values in a separate column. Multiplication by m/n
    gives channel-mean/measurement-sum loss with a unitary Fourier transform,
    matching the weighted objective's convention with weights omitted.
    """
    frame = frame.copy()
    if "saved_bp_best_loss" in frame:
        raise ValueError("Loss convention has already been converted")
    counts = pd.to_numeric(frame.get("m_coeffs", pd.Series(np.nan, index=frame.index)), errors="coerce")
    for index in frame.index[counts.isna()]:
        path = ROOT / frame.loc[index, "artifact_relpath"] / "run_data.npz"
        with np.load(path, allow_pickle=False) as saved:
            counts.loc[index] = int(saved["m_coeffs"].item())
    if not ((counts > 0) & (counts <= SPATIAL_DIMENSION) & (counts == np.floor(counts))).all():
        raise ValueError("Invalid saved measurement count")
    frame["m_coeffs"] = counts.astype(int)
    frame["saved_bp_best_loss"] = frame["bp_best_loss"]
    frame["loss_conversion_factor"] = counts / SPATIAL_DIMENSION
    frame["bp_best_loss"] = frame.saved_bp_best_loss * frame.loss_conversion_factor
    np.testing.assert_allclose(frame.bp_best_loss / frame.loss_conversion_factor,
                               frame.saved_bp_best_loss, rtol=1e-12)
    return frame


def attach_archived_lpips(frame):
    """Resolve sidecar paths recorded before the unweighted archive was moved."""
    saved = pd.read_csv(ROOT / "unweighted/results/metrics/lpips.csv")
    saved["artifact_relpath"] = saved.artifact_relpath.str.replace(
        "results/unweighted/", "unweighted/results/", regex=False,
    )
    if saved.artifact_relpath.duplicated().any():
        raise ValueError("Duplicate archival LPIPS paths after relocation")
    saved = saved.set_index("artifact_relpath")
    frame = frame.copy()
    for metric in ("lpips", "zero_filled_lpips"):
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce").fillna(
            frame.artifact_relpath.map(saved[metric]),
        )
    return frame


def validate(frame, *, ablation, ratios=RATIOS):
    """Require a complete grid and five finite observations in every cell."""
    row, line = (("distribution_key", "line_condition") if ablation else
                 ("sampling_condition", "reconstruction_condition"))
    keys = [row, line, "samp_perc", "repeat_id"]
    expected = pd.MultiIndex.from_product(
        [LAWS[:4] if ablation else LAWS,
         CFG if ablation else RECOVERY_CONDITIONS, ratios, range(5)], names=keys,
    )
    actual = pd.MultiIndex.from_frame(frame[keys])
    if actual.has_duplicates or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError(
            f"Incomplete or duplicate unweighted grid: ablation={ablation}, rows={len(frame)}, "
            f"missing={list(expected.difference(actual))[:8]}, extra={list(actual.difference(expected))[:8]}"
        )
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Unweighted results require finite saved metrics")


def load_results():
    """Load archival main results and the four complete Christoffel CFG lines."""
    return ([load_scenario(scenario) for scenario in SCENARIOS],
            [load_scenario(scenario, ablation=True) for scenario in SCENARIOS])


def load_scenario(scenario, *, ablation=False):
    """Read and validate one reconstruction grid."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if not ablation:
        analysis, completion = recovery.load_unweighted_main_analysis(
            ROOT, base_tag=f"unweighted/{scenario}/sunset", include_partial=True,
        )
        frame = attach_archived_lpips(analysis.rows)
        validate(frame[frame.samp_perc.isin(MAIN_RATIOS)], ablation=False, ratios=MAIN_RATIOS)
        if completion["left"].sum() != 0:
            raise ValueError(f"Unexpected missing main observations in {scenario}")
    else:
        experiment = EXPERIMENTS[SCENARIOS.index(scenario)]
        frame = attach_archived_lpips(cfg.load_cfg_ablation_rows(ROOT, experiment=experiment))
        frame = frame[frame.distribution_key.isin(LAWS[:4]) & frame.line_condition.isin(CFG)].copy()
        validate(frame, ablation=True)
        for rank, law in enumerate(LAWS[:4]):
            frame.loc[frame.distribution_key.eq(law), "distribution_rank"] = rank
            frame.loc[frame.distribution_key.eq(law), "distribution_label"] = SAMPLING_LABELS[law]
    frame["scenario"] = scenario
    return convert_loss_convention(frame)


def plot_cfg_curves(frame, output_dir, *, show=False):
    previous = cfg.METRIC_SPECS
    try:
        cfg.METRIC_SPECS = [(key, "Best Unweighted Loss" if key == "bp_best_loss" else METRIC_HEADERS[key]) for key in METRICS]
        return cfg.plot_metric_curves(frame, metrics=METRICS, band="ci", xscale="log", output_dir=output_dir, show=show)
    finally:
        cfg.METRIC_SPECS = previous


def forest_arguments(*, ablation=False):
    return dict(row_column="distribution_key" if ablation else "sampling_condition",
                line_column="line_condition" if ablation else "reconstruction_condition",
                row_specs=[{"key": law, "label": SAMPLING_LABELS[law]} for law in (LAWS[:4] if ablation else LAWS)],
                line_specs=[{"key": key, "label": label, "color": color, "marker": "o"}
                            for key, label, color in (
                                zip(CFG, ("CFG 1", "CFG 3", "CFG 5", "CFG 7.5"), [cfg.LINE_COLORS[c] for c in CFG]) if ablation else
                                zip(RECOVERY_CONDITIONS, RECOVERY_LABELS.values(), recovery.reconstruction_colors()))],
                expected_sampling_ratios=RATIOS)


def plot_scenario_forest(frame, output_dir, *, ablation=False, show=False):
    metrics = tuple((key, LOSS_LABEL if key == "bp_best_loss" else label,
                     "linear" if key == "bp_best_loss" else scale)
                    for key, label, scale in recovery.AGGREGATE_FOREST_METRICS)
    return recovery.plot_aggregate_metric_forest(frame, **forest_arguments(ablation=ablation), metrics=metrics,
        output_path=Path(output_dir) / "aggregate_metric_forest.pdf", show=show)


def plot_combined_forest(frames, output_dir, *, ablation=False, show=False):
    previous = recovery.AGGREGATE_FOREST_METRICS
    try:
        recovery.AGGREGATE_FOREST_METRICS = tuple(
            (key, LOSS_LABEL if key == "bp_best_loss" else label, "linear" if key == "bp_best_loss" else scale)
            for key, label, scale in previous)
        if not ablation:
            return recovery.plot_combined_aggregate_recovery_forest(frames, expected_sampling_ratios=RATIOS,
                output_path=Path(output_dir) / "aggregate_metric_forest_combined.pdf", show=show)
        return recovery.plot_combined_aggregate_metric_forest(frames, **forest_arguments(ablation=ablation),
            output_path=Path(output_dir) / ("aggregate_cfg_forest_combined.pdf" if ablation else "aggregate_metric_forest_combined.pdf"), show=show)
    finally:
        recovery.AGGREGATE_FOREST_METRICS = previous


def export_traces(frame, output_dir, *, show=False):
    """Plot saved data-loss trajectories in the same units as reported losses."""
    records = []
    for row in frame.itertuples():
        with np.load(ROOT / row.artifact_relpath / "run_data.npz", allow_pickle=False) as saved:
            for iteration, value in zip(saved["bp_iter"], saved["bp_data_loss"]):
                records.append((row.sampling_condition if hasattr(row, "sampling_condition") else row.distribution_key,
                                row.reconstruction_condition if hasattr(row, "reconstruction_condition") else row.line_condition,
                                row.samp_perc, row.repeat_id, int(iteration), float(value) * row.loss_conversion_factor))
    traces = pd.DataFrame(records, columns=["sampling", "recovery", "ratio", "trial", "iteration", "loss"])
    within_trial = traces.groupby(["recovery", "trial", "iteration"]).loss.mean()
    summary = within_trial.groupby(level=[0, 2]).agg(["mean", "std", "count"]).reset_index()
    summary["ci"] = summary["std"] * t.ppf(.975, summary["count"] - 1) / np.sqrt(summary["count"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "optimization_traces.csv", index=False)
    with plt.rc_context(christoffel.SD15_PRESENTATION_RC):
        fig, ax = plt.subplots(figsize=(12, 7))
        for key, group in summary.groupby("recovery", sort=False):
            x, y, ci = (group[c].to_numpy(float) for c in ("iteration", "mean", "ci"))
            ax.plot(x, y, label=key)
            ax.fill_between(x, np.maximum(y-ci, np.finfo(float).tiny), y+ci, alpha=.15)
        ax.set(xlabel="Optimization Iteration", ylabel="Unweighted Loss", yscale="log")
        ax.legend()
        fig.savefig(output_dir / "optimization_traces.pdf", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
    return output_dir / "optimization_traces.pdf"
