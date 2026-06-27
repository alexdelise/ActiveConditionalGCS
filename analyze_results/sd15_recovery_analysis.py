"""Shared loading and plotting helpers for the SD1.5 recovery-result notebooks."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd

import sd15_conditioning_experiment as exp


DEFAULT_DISTRIBUTION_TAG_SUFFIXES: tuple[str, ...] = (
    "sample_k0_unconditioned",
    "sample_k1_daytime_beach",
    "sample_k2_sunset_beach",
    "sample_k4_cat",
)
DEFAULT_SAMPLING_METHODS: tuple[str, ...] = ("cs",)
DEFAULT_ALLOWED_SAMPLING_PERCENTAGES: tuple[float, ...] = (
    0.00015625,
    0.0003125,
    0.000625,
    0.00125,
    0.0025,
    0.005,
    0.01,
)

PROMPT_TEXT = {
    "unprompted": "",
    "daytime_beach": "daytime beach",
    "sunset_beach": "sunset beach",
    "cat": "cat",
}
METRIC_LABELS = {
    "psnr_db": "PSNR (dB)",
    "ssim": "SSIM",
    "pixel_mae": "Per-Pixel MAE",
}
SHOW_METRIC_CONFIDENCE_INTERVALS = True
METRIC_CONFIDENCE_LEVEL = 0.95
GLOBAL_SAMPLING_X_LABEL = r"Sampling Ratio $m/n$"
try:
    import sd15_cfg_ablation_analysis as cfg_ablation

    cfg_ablation = importlib.reload(cfg_ablation)
    ABLATION_RECON_COLORS = [
        cfg_ablation.LINE_COLORS[key]
        for key in ["unconditioned", "cfg1", "cfg7p5", "cfg5", "cfg3"]
    ]
except Exception:
    ABLATION_RECON_COLORS = ["#4C78A8", "#54A24B", "#E45756", "#6A3D9A", "#F58518"]
RECON_COLOR_BY_CONDITION = {
    "unprompted": ABLATION_RECON_COLORS[0],
    "daytime_beach": ABLATION_RECON_COLORS[1],
    "sunset_beach": ABLATION_RECON_COLORS[2],
    "cat": ABLATION_RECON_COLORS[3],
}
SWEEP_FIGSIZE_PER_COL = 4.9
SWEEP_FIGSIZE_PER_ROW = 3.7
SWEEP_SINGLE_FIGSIZE_HEIGHT = 5.25
SWEEP_SINGLE_LEGEND_Y = 0.985
SWEEP_SINGLE_TOP = 0.76
SWEEP_SINGLE_BOTTOM = 0.28
SWEEP_SINGLE_LEFT = 0.065
SWEEP_SINGLE_RIGHT = 0.995
SWEEP_SINGLE_WSPACE = 0.18
SWEEP_LEGEND_Y = 1.08
SWEEP_LINEWIDTH = 2.4
SWEEP_MARKERSIZE = 7.0
SWEEP_MARKER_EDGEWIDTH = 1.3

ZERO_FILLED_METRIC_COLUMNS = {
    "psnr_db": "zero_filled_psnr_db",
    "ssim": "zero_filled_ssim",
    "pixel_mae": "zero_filled_pixel_mae",
}


@dataclass(frozen=True)
class RecoveryAnalysis:
    """Loaded recovery rows and the output location selected from candidate tags."""

    sd15_root: Path
    active_tag: str
    loaded_tags: list[str]
    output_dir: Path
    rows: pd.DataFrame
    mean_table: pd.DataFrame


def find_sd15_root(start: str | Path | None = None) -> Path:
    """Resolve the SD1.5 project root."""

    return exp.find_sd15_root(start)


def split_tag_group_candidates(
    base_tag: str,
    *,
    distribution_suffixes: Sequence[str] = DEFAULT_DISTRIBUTION_TAG_SUFFIXES,
) -> list[tuple[str, list[str]]]:
    """Build split-suite, unsplit-suite, and aggregate fallback tag groups."""

    base = str(base_tag).strip("/")
    suffixes = [str(suffix) for suffix in distribution_suffixes]
    first_tags = [f"{base}/first4_{suffix}" for suffix in suffixes]
    last_tags = [f"{base}/last3_{suffix}" for suffix in suffixes]
    unsplit_tags = [f"{base}/{suffix}" for suffix in suffixes]
    return [
        (base, first_tags + last_tags),
        (base, unsplit_tags),
        (base, [base]),
    ]


def unsplit_tag_group_candidates(
    base_tag: str,
    *,
    distribution_suffixes: Sequence[str] = DEFAULT_DISTRIBUTION_TAG_SUFFIXES,
) -> list[tuple[str, list[str]]]:
    """Build per-distribution and aggregate fallback tag groups."""

    base = str(base_tag).strip("/")
    suffixes = [str(suffix) for suffix in distribution_suffixes]
    return [
        (base, [f"{base}/{suffix}" for suffix in suffixes]),
        (base, [base]),
    ]


def load_recovery_analysis(
    sd15_root: str | Path,
    *,
    tag_group_candidates: Sequence[tuple[str, Sequence[str]]],
    sampling_methods: Sequence[str] = DEFAULT_SAMPLING_METHODS,
    allowed_sampling_percentages: Optional[Iterable[float]] = DEFAULT_ALLOWED_SAMPLING_PERCENTAGES,
    excluded_sampling_conditions: Optional[Iterable[str]] = None,
    include_partial: bool = True,
    output_root: str | Path | None = None,
) -> RecoveryAnalysis:
    """Load the first available candidate tag group and prepare its analysis output folder."""

    root = find_sd15_root(sd15_root)
    candidates = [
        (str(output_tag), [str(tag) for tag in tags])
        for output_tag, tags in tag_group_candidates
    ]
    if not candidates:
        raise ValueError("tag_group_candidates must contain at least one candidate group.")

    rows = pd.DataFrame()
    active_tag = candidates[0][0]
    loaded_tags: list[str] = []
    for output_tag, tags in candidates:
        frames: list[pd.DataFrame] = []
        candidate_loaded_tags: list[str] = []
        for tag in tags:
            try:
                candidate_rows = exp.load_regression_rows(
                    root,
                    tag=tag,
                    sampling_methods=sampling_methods,
                    include_partial=include_partial,
                )
            except FileNotFoundError:
                candidate_rows = pd.DataFrame()
            if candidate_rows.empty:
                continue
            candidate_rows = candidate_rows.copy()
            candidate_rows["source_suite_tag"] = tag
            frames.append(candidate_rows)
            candidate_loaded_tags.append(tag)
        if frames:
            rows = pd.concat(frames, ignore_index=True)
            active_tag = output_tag
            loaded_tags = candidate_loaded_tags
            break

    resolved_output_root = Path(output_root) if output_root is not None else root / "results" / "figures"
    output_dir = resolved_output_root / active_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows.empty and allowed_sampling_percentages:
        allowed = np.array(sorted(float(value) for value in allowed_sampling_percentages), dtype=float)
        samp_values = rows["samp_perc"].astype(float).to_numpy()
        keep_allowed = np.isclose(
            samp_values[:, None],
            allowed[None, :],
            rtol=0.0,
            atol=5e-8,
        ).any(axis=1)
        rows = rows[keep_allowed].copy()

    excluded = {str(value) for value in (excluded_sampling_conditions or [])}
    if not rows.empty and excluded:
        rows = rows[~rows["sampling_condition"].isin(excluded)].copy()

    mean_table = exp.build_mean_metric_table(rows) if not rows.empty else pd.DataFrame()
    return RecoveryAnalysis(
        sd15_root=root,
        active_tag=active_tag,
        loaded_tags=loaded_tags,
        output_dir=output_dir,
        rows=rows,
        mean_table=mean_table,
    )


def title_case(text: Any) -> str:
    return " ".join(str(text).replace("_", " ").split()).title()


def prompt_text_for(condition: Any, default_label: Any = "") -> str:
    return PROMPT_TEXT.get(
        str(condition),
        str(default_label).replace(" Recovery", "").replace(" Reconstruction", "").lower(),
    )


def recovery_math_label(condition: Any, label: Any = None) -> str:
    prompt = prompt_text_for(condition, label or condition)
    return rf'$c_r = \texttt{{"{prompt}"}}$'


def sampling_mu_label(condition: Any) -> str:
    return rf"${exp._mu_symbol(str(condition), hat=True)}$"


def sampling_tick_labels(values: Sequence[float]) -> list[str]:
    labels: list[str] = []
    for value in values:
        value = float(value)
        if value < 0.01:
            labels.append(f"{value:.5f}")
        elif value < 0.1:
            labels.append(f"{value:.3f}")
        else:
            labels.append(f"{value:.2f}".rstrip("0").rstrip("."))
    return labels


def reconstruction_colors() -> list[str]:
    return ABLATION_RECON_COLORS


def reconstruction_color(condition_or_idx: str | int, idx: Optional[int] = None) -> str:
    colors = reconstruction_colors()
    if isinstance(condition_or_idx, str):
        if condition_or_idx in RECON_COLOR_BY_CONDITION:
            return RECON_COLOR_BY_CONDITION[condition_or_idx]
        if idx is not None:
            return colors[int(idx) % len(colors)]
    return colors[int(condition_or_idx) % len(colors)]


def legend_zero_filled_last(handles: Sequence[Any], labels: Sequence[str]) -> tuple[list[Any], list[str]]:
    paired = list(zip(handles, labels))
    ordered = [(handle, label) for handle, label in paired if label != "Zero-Filled"]
    ordered.extend((handle, label) for handle, label in paired if label == "Zero-Filled")
    if not ordered:
        return [], []
    return [handle for handle, _ in ordered], [label for _, label in ordered]


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return exp.build_metric_summary_table(frame, confidence_level=METRIC_CONFIDENCE_LEVEL)


def zero_filled_metric_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    ci_column = f"{metric}_ci_halfwidth"
    empty = pd.DataFrame(columns=["sampling_condition", "sampling_rank", "samp_perc", metric, ci_column])
    column = ZERO_FILLED_METRIC_COLUMNS.get(metric)
    if column is None or column not in frame.columns:
        return empty

    needed_columns = [
        "sampling_method",
        "sampling_condition",
        "sampling_rank",
        "samp_perc",
        "item_id",
        "repeat_id",
        column,
    ]
    available_columns = [col for col in needed_columns if col in frame.columns]
    baseline = frame[available_columns].copy()
    baseline[column] = pd.to_numeric(baseline[column], errors="coerce")
    baseline["samp_perc"] = pd.to_numeric(baseline["samp_perc"], errors="coerce")
    baseline = baseline.dropna(subset=["sampling_condition", "samp_perc", column])
    if baseline.empty:
        return empty
    dedupe_columns = [
        col
        for col in ["sampling_method", "sampling_condition", "samp_perc", "item_id", "repeat_id"]
        if col in baseline.columns
    ]
    if dedupe_columns:
        baseline = baseline.drop_duplicates(subset=dedupe_columns, keep="last")

    group_columns = [
        col for col in ["sampling_condition", "sampling_rank", "samp_perc"] if col in baseline.columns
    ]
    z_value = 0.0
    if METRIC_CONFIDENCE_LEVEL > 0.0:
        z_value = float(
            NormalDist().inv_cdf(0.5 + 0.5 * min(float(METRIC_CONFIDENCE_LEVEL), 0.999999))
        )
    summary = (
        baseline.groupby(group_columns, dropna=False, sort=False)[column]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["count"] = summary["count"].fillna(0).astype(int)
    summary["sem"] = 0.0
    valid = summary["count"] > 0
    summary.loc[valid, "sem"] = summary.loc[valid, "std"] / np.sqrt(
        summary.loc[valid, "count"].astype(float)
    )
    summary[ci_column] = summary["sem"] * z_value
    return summary.rename(columns={"mean": metric})[group_columns + [metric, ci_column]]


def plot_metric_curves(
    frame: pd.DataFrame,
    metric: str,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Optional[Path]:
    """Plot one metric across sampling ratios for each recovery prompt."""

    if frame.empty:
        print(f"No rows available for {metric}.")
        return None

    summary = metric_summary(frame)
    sampling_cases = (
        summary[["sampling_condition", "sampling_rank"]]
        .drop_duplicates()
        .sort_values("sampling_rank", kind="stable")
    )
    recon_cases = (
        summary[["reconstruction_condition", "reconstruction_label", "recon_rank"]]
        .drop_duplicates()
        .sort_values("recon_rank", kind="stable")
    )
    band_column = f"{metric}_ci_halfwidth"
    zero_summary = zero_filled_metric_summary(frame, metric)

    with plt.rc_context(exp.SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(
            1,
            len(sampling_cases),
            figsize=(SWEEP_FIGSIZE_PER_COL * len(sampling_cases), SWEEP_SINGLE_FIGSIZE_HEIGHT),
            sharey=True,
            constrained_layout=False,
        )
        axes = np.atleast_1d(axes)
        for ax, (_, sampling_case) in zip(axes, sampling_cases.iterrows()):
            subset = summary[summary["sampling_condition"] == sampling_case["sampling_condition"]]
            for idx, (_, recon_case) in enumerate(recon_cases.iterrows()):
                group = subset[
                    subset["reconstruction_condition"] == recon_case["reconstruction_condition"]
                ]
                if group.empty:
                    continue
                group = group.sort_values("samp_perc", kind="stable")
                x = group["samp_perc"].to_numpy(dtype=float)
                y = group[metric].to_numpy(dtype=float)
                ci_halfwidth = group[band_column].fillna(0.0).to_numpy(dtype=float)
                color = reconstruction_color(recon_case["reconstruction_condition"], idx)
                label = recovery_math_label(
                    recon_case["reconstruction_condition"],
                    recon_case["reconstruction_label"],
                )
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    marker=exp.SD15_RECON_MARKERS[idx % len(exp.SD15_RECON_MARKERS)],
                    markerfacecolor="white",
                    markeredgewidth=SWEEP_MARKER_EDGEWIDTH,
                    markersize=SWEEP_MARKERSIZE,
                    linewidth=SWEEP_LINEWIDTH,
                )
                if SHOW_METRIC_CONFIDENCE_INTERVALS and np.any(ci_halfwidth > 0.0):
                    ax.fill_between(
                        x,
                        y - ci_halfwidth,
                        y + ci_halfwidth,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                    )
            zero_group = zero_summary[
                zero_summary["sampling_condition"] == sampling_case["sampling_condition"]
            ]
            if not zero_group.empty:
                zero_group = zero_group.sort_values("samp_perc", kind="stable")
                x_zero = zero_group["samp_perc"].to_numpy(dtype=float)
                y_zero = zero_group[metric].to_numpy(dtype=float)
                zero_ci_halfwidth = zero_group[band_column].fillna(0.0).to_numpy(dtype=float)
                ax.plot(
                    x_zero,
                    y_zero,
                    label="Zero-Filled",
                    color="black",
                    linestyle="--",
                    marker="x",
                    markeredgewidth=SWEEP_MARKER_EDGEWIDTH,
                    markersize=SWEEP_MARKERSIZE,
                    linewidth=SWEEP_LINEWIDTH,
                )
                if SHOW_METRIC_CONFIDENCE_INTERVALS and np.any(zero_ci_halfwidth > 0.0):
                    ax.fill_between(
                        x_zero,
                        y_zero - zero_ci_halfwidth,
                        y_zero + zero_ci_halfwidth,
                        color="black",
                        alpha=0.10,
                        linewidth=0,
                    )
            ticks = sorted({float(value) for value in subset["samp_perc"].tolist()})
            ax.set_xscale("log")
            ax.set_xticks(ticks)
            ax.set_xticklabels(sampling_tick_labels(ticks), rotation=35, ha="right")
            ax.set_xlabel("")
            ax.set_title(sampling_mu_label(sampling_case["sampling_condition"]))
            ax.grid(True, which="major", axis="both", alpha=0.28, linestyle="--")
        axes[0].set_ylabel(METRIC_LABELS.get(metric, title_case(metric)))
        fig.subplots_adjust(
            left=SWEEP_SINGLE_LEFT,
            right=SWEEP_SINGLE_RIGHT,
            bottom=SWEEP_SINGLE_BOTTOM,
            top=SWEEP_SINGLE_TOP,
            wspace=SWEEP_SINGLE_WSPACE,
        )
        fig.supxlabel(
            GLOBAL_SAMPLING_X_LABEL,
            fontsize=exp.SD15_PRESENTATION_RC.get("axes.labelsize", 30),
            y=0.045,
        )
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for legend_ax in axes:
            handles, labels = legend_ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label and label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        if legend_handles:
            legend_handles, legend_labels = legend_zero_filled_last(legend_handles, legend_labels)
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, SWEEP_SINGLE_LEGEND_Y),
                ncol=min(len(legend_labels), 5),
                frameon=False,
                fontsize=exp.SD15_PRESENTATION_RC.get("legend.fontsize", 22),
            )
        resolved_output_path = Path(output_path) if output_path is not None else None
        if resolved_output_path is not None:
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(resolved_output_path, dpi=exp.SD15_EXPORT_DPI, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return resolved_output_path


def plot_combined_metric_curves(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    output_path: str | Path | None = None,
    show: bool = True,
) -> Optional[Path]:
    """Plot a multi-row metric sweep across sampling ratios and recovery prompts."""

    if frame.empty:
        print("No rows available for the combined metric sweep.")
        return None

    summary = metric_summary(frame)
    sampling_cases = (
        summary[["sampling_condition", "sampling_rank"]]
        .drop_duplicates()
        .sort_values("sampling_rank", kind="stable")
    )
    recon_cases = (
        summary[["reconstruction_condition", "reconstruction_label", "recon_rank"]]
        .drop_duplicates()
        .sort_values("recon_rank", kind="stable")
    )
    if sampling_cases.empty or recon_cases.empty:
        print("No cases available for the combined metric sweep.")
        return None

    n_rows = len(metrics)
    n_cols = len(sampling_cases)
    with plt.rc_context(exp.SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(SWEEP_FIGSIZE_PER_COL * n_cols, SWEEP_FIGSIZE_PER_ROW * n_rows),
            sharex="col",
            sharey="row",
            squeeze=False,
            constrained_layout=True,
        )
        for row_idx, metric in enumerate(metrics):
            band_column = f"{metric}_ci_halfwidth"
            zero_summary = zero_filled_metric_summary(frame, metric)
            for col_idx, (_, sampling_case) in enumerate(sampling_cases.iterrows()):
                ax = axes[row_idx, col_idx]
                subset = summary[summary["sampling_condition"] == sampling_case["sampling_condition"]]
                for idx, (_, recon_case) in enumerate(recon_cases.iterrows()):
                    group = subset[
                        subset["reconstruction_condition"] == recon_case["reconstruction_condition"]
                    ]
                    if group.empty:
                        continue
                    group = group.sort_values("samp_perc", kind="stable")
                    x = group["samp_perc"].to_numpy(dtype=float)
                    y = group[metric].to_numpy(dtype=float)
                    ci_halfwidth = group[band_column].fillna(0.0).to_numpy(dtype=float)
                    color = reconstruction_color(recon_case["reconstruction_condition"], idx)
                    label = recovery_math_label(
                        recon_case["reconstruction_condition"],
                        recon_case["reconstruction_label"],
                    )
                    ax.plot(
                        x,
                        y,
                        label=label,
                        color=color,
                        marker=exp.SD15_RECON_MARKERS[idx % len(exp.SD15_RECON_MARKERS)],
                        markerfacecolor="white",
                        markeredgewidth=SWEEP_MARKER_EDGEWIDTH,
                        markersize=SWEEP_MARKERSIZE,
                        linewidth=SWEEP_LINEWIDTH,
                    )
                    if SHOW_METRIC_CONFIDENCE_INTERVALS and np.any(ci_halfwidth > 0.0):
                        ax.fill_between(
                            x,
                            y - ci_halfwidth,
                            y + ci_halfwidth,
                            color=color,
                            alpha=0.16,
                            linewidth=0,
                        )
                zero_group = zero_summary[
                    zero_summary["sampling_condition"] == sampling_case["sampling_condition"]
                ]
                if not zero_group.empty:
                    zero_group = zero_group.sort_values("samp_perc", kind="stable")
                    x_zero = zero_group["samp_perc"].to_numpy(dtype=float)
                    y_zero = zero_group[metric].to_numpy(dtype=float)
                    zero_ci_halfwidth = zero_group[band_column].fillna(0.0).to_numpy(dtype=float)
                    ax.plot(
                        x_zero,
                        y_zero,
                        label="Zero-Filled",
                        color="black",
                        linestyle="--",
                        marker="x",
                        markeredgewidth=SWEEP_MARKER_EDGEWIDTH,
                        markersize=SWEEP_MARKERSIZE,
                        linewidth=SWEEP_LINEWIDTH,
                    )
                    if SHOW_METRIC_CONFIDENCE_INTERVALS and np.any(zero_ci_halfwidth > 0.0):
                        ax.fill_between(
                            x_zero,
                            y_zero - zero_ci_halfwidth,
                            y_zero + zero_ci_halfwidth,
                            color="black",
                            alpha=0.10,
                            linewidth=0,
                        )
                ticks = sorted({float(value) for value in subset["samp_perc"].tolist()})
                ax.set_xscale("log")
                ax.set_xticks(ticks)
                ax.set_xticklabels(sampling_tick_labels(ticks), rotation=35, ha="right")
                ax.set_xlabel("")
                if row_idx == 0:
                    ax.set_title(sampling_mu_label(sampling_case["sampling_condition"]))
                if col_idx == 0:
                    ax.set_ylabel(METRIC_LABELS.get(metric, title_case(metric)))
                ax.grid(True, which="major", axis="both", alpha=0.28, linestyle="--")

        fig.supxlabel(
            GLOBAL_SAMPLING_X_LABEL,
            fontsize=exp.SD15_PRESENTATION_RC.get("axes.labelsize", 30),
        )
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for legend_ax in axes.ravel():
            handles, labels = legend_ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label and label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        if legend_handles:
            legend_handles, legend_labels = legend_zero_filled_last(legend_handles, legend_labels)
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, SWEEP_LEGEND_Y),
                ncol=min(len(legend_labels), 5),
                frameon=False,
                fontsize=exp.SD15_PRESENTATION_RC.get("legend.fontsize", 22),
            )
        resolved_output_path = Path(output_path) if output_path is not None else None
        if resolved_output_path is not None:
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(resolved_output_path, dpi=exp.SD15_EXPORT_DPI, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return resolved_output_path


def export_metric_figures(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    sweep_metrics: Sequence[str] = ("psnr_db", "ssim", "pixel_mae"),
    combined_metrics: Sequence[str] = ("psnr_db", "ssim"),
    show: bool = True,
) -> list[Path]:
    """Export the metric PDFs produced by the recovery-result notebooks."""

    outputs: list[Path] = []
    output_root = Path(output_dir)
    if frame.empty:
        return outputs

    for sampling_method in frame["sampling_method"].drop_duplicates().tolist():
        subset = frame[frame["sampling_method"] == sampling_method].copy()
        if subset.empty:
            continue
        for metric in sweep_metrics:
            output = plot_metric_curves(
                subset,
                metric,
                output_path=output_root / f"{sampling_method}_{metric}_by_recovery_prompt.pdf",
                show=show,
            )
            if output is not None:
                outputs.append(output)
        output = plot_combined_metric_curves(
            subset,
            combined_metrics,
            output_path=output_root / f"{sampling_method}_combined_metrics_by_recovery_prompt.pdf",
            show=show,
        )
        if output is not None:
            outputs.append(output)
    return outputs


def sample_tag(value: float) -> str:
    return f"samp_{float(value):.5f}".replace(".", "p")


def run_artifact_dir(sd15_root: str | Path, row: pd.Series) -> Path:
    return (
        Path(sd15_root)
        / "results"
        / str(row["run_tag"])
        / str(row["sampling_method"])
        / f"item_{int(row['item_id']):03d}"
        / sample_tag(float(row["samp_perc"]))
        / f"rep_{int(row['repeat_id']):02d}"
    )


def recon_path_for(sd15_root: str | Path, row: pd.Series) -> Path:
    return run_artifact_dir(sd15_root, row) / f"recon_{row['sampling_method']}.png"


def zero_filled_path_for(sd15_root: str | Path, row: pd.Series) -> Path:
    return run_artifact_dir(sd15_root, row) / "zero_filled_ifft.png"


def load_target_path(sd15_root: str | Path, frame: pd.DataFrame, *, item_id: int = 0) -> Path:
    run_tag = str(frame["run_tag"].iloc[0])
    dataset_ref_path = Path(sd15_root) / "results" / run_tag / "dataset_ref.json"
    with dataset_ref_path.open("r", encoding="utf-8") as handle:
        dataset_ref = json.load(handle)
    item = next(item for item in dataset_ref["items"] if int(item["item_id"]) == int(item_id))
    return Path(item["gt_png_path"])


def add_zero_filled_metric_label(ax: Any, row: pd.Series) -> None:
    psnr = row.get("zero_filled_psnr_db", np.nan)
    ssim = row.get("zero_filled_ssim", np.nan)
    ppmae = row.get("zero_filled_pixel_mae", np.nan)
    if pd.isna(psnr) or pd.isna(ssim) or pd.isna(ppmae):
        return
    ax.text(
        0.02,
        0.96,
        f"PSNR {float(psnr):.2f} dB\nSSIM {float(ssim):.3f}\nPPMAE {float(ppmae):.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="white",
        bbox={
            "boxstyle": "square,pad=0.22",
            "facecolor": (0, 0, 0, 0.46),
            "edgecolor": "none",
        },
    )


def show_image_or_placeholder(ax: Any, path: str | Path, *, cmap: Any = None) -> None:
    image_path = Path(path)
    if image_path.is_file():
        ax.imshow(mpimg.imread(image_path), cmap=cmap)
    else:
        ax.text(
            0.5,
            0.5,
            "Missing",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#6B7280",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def best_reconstruction_record(
    rows: pd.DataFrame,
    *,
    selection_metrics: Sequence[str] = ("psnr_db", "ssim"),
) -> pd.Series:
    sort_columns = [column for column in selection_metrics if column in rows.columns]
    tie_breakers = [column for column in ["item_id", "repeat_id", "run_tag"] if column in rows.columns]
    if sort_columns or tie_breakers:
        rows = rows.sort_values(
            sort_columns + tie_breakers,
            ascending=([False] * len(sort_columns)) + ([True] * len(tie_breakers)),
            na_position="last",
            kind="stable",
        )
    return rows.iloc[0]


def plot_recovery_grid(
    frame: pd.DataFrame,
    *,
    sd15_root: str | Path,
    sampling_condition: str,
    item_id: Optional[int] = 0,
    repeat_id: Optional[int] = None,
    sampling_method: str = DEFAULT_SAMPLING_METHODS[0],
    sampling_percentage: float = 0.00125,
    panel_width_in: float = 3.0,
    panel_height_in: float = 4.2,
    selection_metrics: Sequence[str] = ("psnr_db", "ssim"),
    output_path: str | Path | None = None,
    show: bool = True,
) -> Optional[Path]:
    """Plot the ground truth, zero-filled image, and best recovery for each prompt."""

    if frame.empty:
        print("No rows available for the recovery grid.")
        return None

    panel = frame[
        (frame["sampling_method"] == sampling_method)
        & (frame["sampling_condition"] == sampling_condition)
    ].copy()
    if item_id is not None:
        panel = panel[panel["item_id"] == item_id]
    if repeat_id is not None:
        panel = panel[panel["repeat_id"] == repeat_id]
    if panel.empty:
        print(f"No matching rows for sampling distribution {sampling_condition}.")
        return None

    panel = panel.sort_values(["samp_perc", "recon_rank"], kind="stable")
    target_samp_perc = float(sampling_percentage)
    keep_rate = np.isclose(
        panel["samp_perc"].astype(float).to_numpy(),
        target_samp_perc,
        rtol=0.0,
        atol=5e-8,
    )
    panel = panel[keep_rate].copy()
    if panel.empty:
        print(
            f"No matching rows for sampling distribution {sampling_condition} "
            f"at samp_perc={target_samp_perc:.5f}."
        )
        return None
    sampling_rates = [target_samp_perc]
    recon_cases = (
        panel[["reconstruction_condition", "reconstruction_label", "recon_rank"]]
        .drop_duplicates()
        .sort_values("recon_rank", kind="stable")
    )
    target_path = load_target_path(sd15_root, panel, item_id=int(item_id or 0))
    n_cols = 2 + len(recon_cases)

    with plt.rc_context(exp.SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(
            len(sampling_rates),
            n_cols,
            figsize=(panel_width_in * n_cols, panel_height_in * len(sampling_rates)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
        fig.suptitle(
            rf"Sampling Distribution: {sampling_mu_label(sampling_condition)}",
            fontsize=34,
        )
        for row_idx, samp_perc in enumerate(sampling_rates):
            rate_rows = panel[np.isclose(panel["samp_perc"].astype(float), float(samp_perc))]
            reference_row = best_reconstruction_record(
                rate_rows,
                selection_metrics=selection_metrics,
            )
            show_image_or_placeholder(axes[row_idx, 0], target_path)
            show_image_or_placeholder(
                axes[row_idx, 1],
                zero_filled_path_for(sd15_root, reference_row),
            )
            add_zero_filled_metric_label(axes[row_idx, 1], reference_row)
            axes[row_idx, 0].set_ylabel(f"{float(samp_perc):.5f}", fontsize=22, labelpad=4)
            if row_idx == 0:
                axes[row_idx, 0].set_title("Ground Truth", fontsize=22, pad=6)
                axes[row_idx, 1].set_title("Zero-Filled", fontsize=22, pad=6)
            for col_offset, (_, recon_case) in enumerate(recon_cases.iterrows(), start=2):
                match = rate_rows[
                    rate_rows["reconstruction_condition"] == recon_case["reconstruction_condition"]
                ]
                ax = axes[row_idx, col_offset]
                if match.empty:
                    show_image_or_placeholder(ax, Path("__missing__"))
                    continue
                record = best_reconstruction_record(match, selection_metrics=selection_metrics)
                show_image_or_placeholder(ax, recon_path_for(sd15_root, record))
                if row_idx == 0:
                    ax.set_title(
                        recovery_math_label(
                            recon_case["reconstruction_condition"],
                            recon_case["reconstruction_label"],
                        ),
                        fontsize=22,
                        pad=6,
                    )
                ax.text(
                    0.02,
                    0.96,
                    f"PSNR {float(record['psnr_db']):.2f} dB\n"
                    f"SSIM {float(record['ssim']):.3f}\n"
                    f"PPMAE {float(record['pixel_mae']):.4f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=12,
                    color="white",
                    bbox={
                        "boxstyle": "square,pad=0.22",
                        "facecolor": (0, 0, 0, 0.46),
                        "edgecolor": "none",
                    },
                )
        resolved_output_path = Path(output_path) if output_path is not None else None
        if resolved_output_path is not None:
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(resolved_output_path, dpi=exp.SD15_EXPORT_DPI, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return resolved_output_path


def export_recovery_grids(
    frame: pd.DataFrame,
    sd15_root: str | Path,
    output_dir: str | Path,
    *,
    sampling_conditions: Optional[Iterable[str]] = None,
    item_id: Optional[int] = 0,
    repeat_id: Optional[int] = None,
    sampling_method: str = DEFAULT_SAMPLING_METHODS[0],
    sampling_percentage: float = 0.00125,
    panel_width_in: float = 3.0,
    panel_height_in: float = 4.2,
    selection_metrics: Sequence[str] = ("psnr_db", "ssim"),
    show: bool = True,
) -> list[Path]:
    """Export one recovery-image grid per sampling distribution."""

    outputs: list[Path] = []
    if frame.empty:
        return outputs

    image_sampling_cases = (
        frame[["sampling_condition", "sampling_rank"]]
        .drop_duplicates()
        .sort_values("sampling_rank", kind="stable")
    )
    if sampling_conditions is not None:
        keep = {str(value) for value in sampling_conditions}
        image_sampling_cases = image_sampling_cases[
            image_sampling_cases["sampling_condition"].isin(keep)
        ]
    for _, sampling_case in image_sampling_cases.iterrows():
        sampling_condition = str(sampling_case["sampling_condition"])
        output = plot_recovery_grid(
            frame,
            sd15_root=sd15_root,
            sampling_condition=sampling_condition,
            item_id=item_id,
            repeat_id=repeat_id,
            sampling_method=sampling_method,
            sampling_percentage=sampling_percentage,
            panel_width_in=panel_width_in,
            panel_height_in=panel_height_in,
            selection_metrics=selection_metrics,
            output_path=Path(output_dir) / f"recovery_image_grid_{sampling_condition}.pdf",
            show=show,
        )
        if output is not None:
            outputs.append(output)
    return outputs
