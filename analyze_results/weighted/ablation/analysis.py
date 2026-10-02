"""Analysis helpers for the weighted recovery-CFG ablation."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = PROJECT_ROOT / "results" / "weighted" / "ablation"
RESULT_ROOT = RUN_ROOT
FIGURE_ROOT = RUN_ROOT
ANALYZE_ROOT = PROJECT_ROOT / "analyze_results"
if str(ANALYZE_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYZE_ROOT))

import sd15_cfg_ablation_analysis as cfgviz
import sd15_conditioning_experiment as experiment
import sd15_recovery_analysis as recovery

# Jupyter may retain the shared plotting module while reloading this study
# helper, so register the study-specific scalar explicitly on every reload
if "bp_best_loss" not in dict(cfgviz.METRIC_SPECS):
    cfgviz.METRIC_SPECS.append(("bp_best_loss", "Best Weighted Loss"))


SCENARIOS = {
    "prompt_matched": {
        "title": "Prompt-Matched In-Range",
        "dataset_name": "sunset_beach_signal_sd15_512x512",
    },
    "prompt_mismatched": {
        "title": "Prompt-Mismatched In-Range",
        "dataset_name": "sunset_sandy_coast_signal_sd15_512x512",
    },
    "out_of_range": {
        "title": "Out-of-Range",
        "dataset_name": "out_of_range_512x512",
    },
}

LAW_INFO = {
    "k0": {
        "condition": "k0",
        "label": r"$\widetilde{\mu}_{c_{\mathrm{uc}}}$",
        "name": "Unconditioned sampling",
        "prefix": "sample_k0_unconditioned",
    },
    "k1": {
        "condition": "k1_daytime_beach",
        "label": r"$\widetilde{\mu}_{c_{\mathrm{db}}}$",
        "name": "Daytime-beach sampling",
        "prefix": "sample_k1_daytime_beach",
    },
    "k2": {
        "condition": "k2_sunset_beach",
        "label": r"$\widetilde{\mu}_{c_{\mathrm{sb}}}$",
        "name": "Sunset-beach sampling",
        "prefix": "sample_k2_sunset_beach",
    },
    "k4": {
        "condition": "k4_cat",
        "label": r"$\widetilde{\mu}_{c_{\mathrm{ca}}}$",
        "name": "Cat sampling",
        "prefix": "sample_k4_cat",
    },
}

LINE_INFO = {
    "cfg1": {"label": "CFG 1", "rank": 0, "cfg_scale": 1.0},
    "cfg3": {"label": "CFG 3", "rank": 1, "cfg_scale": 3.0},
    "cfg5": {"label": "CFG 5", "rank": 2, "cfg_scale": 5.0},
    "cfg7p5": {"label": "CFG 7.5", "rank": 3, "cfg_scale": 7.5},
}

SAMPLING_RATIOS = (0.01, 0.02, 0.03, 0.04, 0.05)
NEW_LINES = ("cfg3", "cfg5", "cfg7p5")
REFERENCE_LINES = ("cfg1",)
NEW_REPEATS = (0, 1, 2, 3, 4)
REFERENCE_REPEATS = (0, 1, 2, 3, 4)
REPEATS_BY_LINE = {
    line: (REFERENCE_REPEATS if line in REFERENCE_LINES else NEW_REPEATS)
    for line in LINE_INFO
}
EXPECTED_NEW_ROWS_PER_SCENARIO = (
    len(LAW_INFO) * len(NEW_LINES) * len(SAMPLING_RATIOS) * len(NEW_REPEATS)
)
EXPECTED_ROWS_PER_SCENARIO = len(LAW_INFO) * len(SAMPLING_RATIOS) * sum(
    len(repeats) for repeats in REPEATS_BY_LINE.values()
)
EXPECTED_ROWS = EXPECTED_ROWS_PER_SCENARIO * len(SCENARIOS)


def case_name(law: str, line: str) -> str:
    """Return the stable case directory for one law and recovery-CFG line."""

    prefix = str(LAW_INFO[law]["prefix"])
    if line == "unconditioned":
        return f"{prefix}__recover_unprompted"
    return f"{prefix}__recover_prompt_sunset_beach_{line}"


def main_case_name(law: str, line: str) -> str:
    """Return the matching five-trial main-study case for a reused control."""

    prefix = str(LAW_INFO[law]["prefix"])
    if line == "unconditioned":
        return f"{prefix}__recover_unprompted"
    if line == "cfg1":
        return f"{prefix}__recover_prompt_sunset_beach"
    raise KeyError(f"Line {line!r} is not a reusable main-study control")


def _case_root(scenario: str, law: str, line: str) -> tuple[Path, str]:
    """Resolve a result directory and provenance for one recovery line."""

    if line in REFERENCE_LINES:
        return PROJECT_ROOT / "results" / "weighted" / scenario / law / main_case_name(law, line), "main_reference"
    return RESULT_ROOT / scenario / law / case_name(law, line), "ablation"


def ensure_lpips(scenario: str, *, device: str = "cpu") -> pd.DataFrame:
    """Populate the study-local LPIPS table from saved or legacy artifacts."""

    return recovery.ensure_lpips_metrics(
        PROJECT_ROOT,
        result_namespace="weighted",
        artifact_roots=[RESULT_ROOT / scenario, PROJECT_ROOT / "results" / "weighted" / scenario],
        metrics_path=RESULT_ROOT / "lpips_metrics.csv",
        device=device,
    )


def load_rows(scenario: str) -> pd.DataFrame:
    """Load completed rows directly from stable case directories."""

    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario!r}; choose one of {tuple(SCENARIOS)}")
    frames: list[pd.DataFrame] = []
    for distribution_rank, (law, law_info) in enumerate(LAW_INFO.items()):
        for line, line_info in LINE_INFO.items():
            name = case_name(law, line)
            case_root, source_kind = _case_root(scenario, law, line)
            if not case_root.is_dir():
                continue
            frame = experiment._load_partial_run_frame(case_root, "cs")
            if frame.empty:
                continue
            frame = experiment._attach_regression_metadata(
                frame,
                sampling_method="cs",
                case={
                    "name": name,
                    "sampling_condition": law_info["condition"],
                    "sampling_label": law_info["label"],
                    "sampling_rank": distribution_rank,
                    "reconstruction_condition": line,
                    "reconstruction_label": line_info["label"],
                    "recon_rank": line_info["rank"],
                },
                case_tag=str(case_root.relative_to(PROJECT_ROOT)),
            )
            # Match the established CFG-ablation plotting schema
            frame["distribution_key"] = law_info["condition"]
            frame["distribution_label"] = law_info["label"]
            frame["distribution_name"] = law_info["name"]
            frame["distribution_rank"] = distribution_rank
            frame["line_condition"] = line
            frame["line_label"] = line_info["label"]
            frame["line_rank"] = line_info["rank"]
            frame["cfg_scale"] = line_info["cfg_scale"]
            frame["dataset_name"] = SCENARIOS[scenario]["dataset_name"]
            frame["case_root"] = str(case_root)
            frame["source_kind"] = source_kind
            frame["reference_reused"] = source_kind == "main_reference"
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    rows = experiment._drop_duplicate_run_rows(pd.concat(frames, ignore_index=True, sort=False))
    rows = recovery.attach_lpips_metrics(
        rows,
        PROJECT_ROOT,
        result_namespace="weighted",
        metrics_path=RESULT_ROOT / "lpips_metrics.csv",
    )
    valid_rates = np.isclose(
        rows["samp_perc"].astype(float).to_numpy()[:, None],
        np.asarray(SAMPLING_RATIOS)[None, :],
        rtol=0.0,
        atol=5e-10,
    ).any(axis=1)
    if not valid_rates.all():
        raise ValueError("A result row has a sampling ratio outside the ablation grid")
    return rows.sort_values(
        ["distribution_rank", "line_rank", "samp_perc", "repeat_id"],
        kind="stable",
    ).reset_index(drop=True)


def completion_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one record for each expected law, CFG, ratio, and trial."""

    records: list[dict[str, object]] = []
    expected_cells = (
        (law, line, ratio, repeat)
        for law in LAW_INFO
        for line in LINE_INFO
        for ratio in SAMPLING_RATIOS
        for repeat in REPEATS_BY_LINE[line]
    )
    for law, line, ratio, repeat in expected_cells:
        if rows.empty:
            observed = 0
        else:
            observed = int(
                (
                    rows["distribution_key"].astype(str).eq(str(LAW_INFO[law]["condition"]))
                    & rows["line_condition"].astype(str).eq(line)
                    & np.isclose(rows["samp_perc"].astype(float), ratio, rtol=0.0, atol=5e-10)
                    & pd.to_numeric(rows["repeat_id"], errors="coerce").eq(repeat)
                ).sum()
            )
        records.append(
            {
                "sampling_law": law,
                "recovery_line": line,
                "samp_perc": ratio,
                "repeat_id": repeat,
                "observed": observed,
                "expected": 1,
                "left": max(0, 1 - observed),
                "complete": observed == 1,
            }
        )
    return pd.DataFrame.from_records(records)


def count_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Show observed trial counts in the established ablation layout."""

    return cfgviz.count_table(rows)


def plot_metric_curves(rows: pd.DataFrame, *, output_dir: Path, show: bool = True):
    """Render one main-study-style figure for each reconstruction metric."""

    # Repeat the registration at call time for long-lived notebook kernels
    if "bp_best_loss" not in dict(cfgviz.METRIC_SPECS):
        cfgviz.METRIC_SPECS.append(("bp_best_loss", "Best Weighted Loss"))
    outputs: dict[str, Path] = {}
    for metric in ("psnr_db", "ssim", "lpips", "bp_best_loss"):
        generated = cfgviz.plot_metric_curves(
            rows,
            output_dir=output_dir,
            show=show,
            band="ci",
            xscale="linear",
            metrics=(metric,),
        )
        outputs.update({f"{metric}_{kind}": path for kind, path in generated.items()})
    return outputs


def plot_aggregate_forest(
    rows: pd.DataFrame,
    *,
    output_dir: str | Path,
    show: bool = True,
) -> Path | None:
    """Render the ratio-aggregated CFG comparison in the shared forest style."""

    row_specs = [
        {
            "key": str(info["condition"]),
            "label": str(info["label"]),
        }
        for info in LAW_INFO.values()
    ]
    line_specs = [
        {
            "key": line,
            "label": str(info["label"]),
            "color": cfgviz.LINE_COLORS[line],
            "marker": "o",
        }
        for line, info in LINE_INFO.items()
    ]
    return recovery.plot_aggregate_metric_forest(
        rows,
        row_column="distribution_key",
        line_column="line_condition",
        row_specs=row_specs,
        line_specs=line_specs,
        expected_sampling_ratios=SAMPLING_RATIOS,
        output_path=Path(output_dir) / "aggregate_cfg_forest.pdf",
        show=show,
    )


def plot_combined_aggregate_forest(
    frames: list[pd.DataFrame],
    *,
    output_path: str | Path,
    show: bool = True,
) -> Path | None:
    """Render all three CFG-ablation scenarios in one shared-legend forest grid."""

    row_specs = [
        {
            "key": str(info["condition"]),
            "label": str(info["label"]),
        }
        for info in LAW_INFO.values()
    ]
    line_specs = [
        {
            "key": line,
            "label": str(info["label"]),
            "color": cfgviz.LINE_COLORS[line],
            "marker": "o",
        }
        for line, info in LINE_INFO.items()
    ]
    return recovery.plot_combined_aggregate_metric_forest(
        frames,
        row_column="distribution_key",
        line_column="line_condition",
        row_specs=row_specs,
        line_specs=line_specs,
        expected_sampling_ratios=SAMPLING_RATIOS,
        output_path=output_path,
        show=show,
    )


def load_optimization_traces(scenario: str) -> pd.DataFrame:
    """Load complete and in-progress objective traces for every available line."""

    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario!r}; choose one of {tuple(SCENARIOS)}")
    records: list[dict[str, object]] = []
    for law, line, ratio in product(LAW_INFO, LINE_INFO, SAMPLING_RATIOS):
        root, source_kind = _case_root(scenario, law, line)
        for repeat in REPEATS_BY_LINE[line]:
            leaf = (
                root
                / "cs"
                / "item_000"
                / f"samp_{ratio:.5f}".replace(".", "p")
                / f"rep_{repeat:02d}"
            )
            source = leaf / "run_data.npz"
            status = "complete"
            if not source.is_file():
                source = leaf / "optimization_trace.npz"
                status = "running"
            if not source.is_file():
                continue
            with np.load(source, allow_pickle=False) as payload:
                iterations = np.asarray(payload["bp_iter"], dtype=np.int64)
                values = np.asarray(payload["bp_loss"], dtype=np.float64)
            if iterations.shape != values.shape:
                raise ValueError(f"Inconsistent trace arrays in {source}")
            for iteration, value in zip(iterations, values):
                records.append(
                    {
                        "distribution_key": LAW_INFO[law]["condition"],
                        "line_condition": line,
                        "line_label": LINE_INFO[line]["label"],
                        "line_rank": LINE_INFO[line]["rank"],
                        "samp_perc": float(ratio),
                        "repeat_id": int(repeat),
                        "iteration": int(iteration),
                        "bp_loss": float(value),
                        "status": status,
                        "source_kind": source_kind,
                        "source": str(source),
                    }
                )
    return pd.DataFrame.from_records(records)


def optimization_trace_summary(traces: pd.DataFrame) -> pd.DataFrame:
    """Summarize objective trajectories on their original scale."""

    if traces.empty:
        return traces.copy()
    columns = ["distribution_key", "line_condition", "samp_perc", "iteration"]
    summary = traces.groupby(columns, sort=True)["bp_loss"].agg(
        mean="mean", std="std", count="count", minimum="min", maximum="max"
    ).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def export_optimization_trace_figures(
    traces: pd.DataFrame,
    output_dir: str | Path,
    *,
    show_uncertainty: bool = True,
    show: bool = True,
) -> list[Path]:
    """Plot one five-ratio objective grid for each fixed sampling law."""

    if traces.empty:
        print("No optimization traces are available yet.")
        return []
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = optimization_trace_summary(traces)
    summary.to_csv(output_root / "optimization_trace_summary.csv", index=False)
    colors = cfgviz.LINE_COLORS
    outputs: list[Path] = []
    for law, info in LAW_INFO.items():
        condition = str(info["condition"])
        selected = traces[traces["distribution_key"].eq(condition)]
        if selected.empty:
            continue
        with plt.rc_context(experiment.SD15_PRESENTATION_RC):
            fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.5), constrained_layout=False)
            plot_axes = list(axes.flat[:5])
            legend_axis = axes.flat[5]
            for axis, ratio in zip(plot_axes, SAMPLING_RATIOS):
                for line, line_info in LINE_INFO.items():
                    raw = selected[
                        selected["line_condition"].eq(line)
                        & np.isclose(selected["samp_perc"], ratio, rtol=0.0, atol=5e-10)
                    ]
                    if raw.empty:
                        continue
                    for _, trial in raw.groupby("repeat_id", sort=True):
                        trial = trial.sort_values("iteration")
                        axis.plot(
                            trial["iteration"], trial["bp_loss"],
                            color=colors[line], linewidth=0.85, alpha=0.20,
                        )
                    aggregate = summary[
                        summary["distribution_key"].eq(condition)
                        & summary["line_condition"].eq(line)
                        & np.isclose(summary["samp_perc"], ratio, rtol=0.0, atol=5e-10)
                    ].sort_values("iteration")
                    if aggregate.empty:
                        continue
                    x = aggregate["iteration"].to_numpy(dtype=float)
                    mean = aggregate["mean"].to_numpy(dtype=float)
                    if show_uncertainty:
                        std = aggregate["std"].to_numpy(dtype=float)
                        low = np.maximum(mean - std, np.finfo(float).tiny)
                        axis.fill_between(x, low, mean + std, color=colors[line], alpha=0.12, linewidth=0)
                    axis.plot(x, mean, color=colors[line], linewidth=2.5)
                axis.set_yscale("log")
                axis.grid(alpha=0.25, which="both")
                axis.tick_params(direction="out")
                axis.set_title(rf"{info['label']}, $m/n={ratio:.2f}$", fontweight="bold")
                axis.set_xlabel("Optimization Iteration")
            legend_axis.axis("off")
            handles = [
                Line2D([0], [0], color=colors[line], linewidth=2.7, label=str(info["label"]))
                for line, info in LINE_INFO.items()
            ]
            handles.extend(
                [
                    Line2D([0], [0], color="0.35", linewidth=0.9, alpha=0.35, label="Individual Trial"),
                    Line2D([0], [0], color="0.35", linewidth=2.7, label="Trial Mean"),
                ]
            )
            legend_axis.legend(handles=handles, loc="center", frameon=False, handlelength=3.2)
            fig.supylabel(
                r"Weighted Objective $\frac{1}{2C}\|\mathbf{A}_{\Omega}G(\mathbf{z},c)-\mathbf{y}\|_2^2$",
                x=0.005,
            )
            fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.95, wspace=0.24, hspace=0.32)
            output = output_root / f"bp_loss_traces_{law}.pdf"
            fig.savefig(output, dpi=experiment.SD15_EXPORT_DPI, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            outputs.append(output)
    return outputs


def plot_reconstruction_panel(
    rows: pd.DataFrame,
    *,
    sampling_ratio: float = 0.01,
    output_dir: Path,
    show: bool = True,
):
    """Render the established best-LPIPS reconstruction panel."""

    return cfgviz.plot_reconstruction_panel(
        rows,
        sd15_root=PROJECT_ROOT,
        samp_perc=sampling_ratio,
        output_dir=output_dir,
        show=show,
    )
