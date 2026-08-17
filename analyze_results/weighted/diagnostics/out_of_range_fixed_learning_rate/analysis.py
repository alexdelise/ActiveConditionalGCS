"""Loading and plotting for the fixed-learning-rate out-of-range diagnostic."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT = PROJECT_ROOT / "results" / "weighted" / "diagnostics" / "out_of_range_fixed_learning_rate"
RESULT_ROOT = RUN_ROOT
ANALYZE_ROOT = PROJECT_ROOT / "analyze_results"
if str(ANALYZE_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYZE_ROOT))

import sd15_recovery_analysis as recovery
import sd15_conditioning_experiment as experiment


SAMPLING_CONDITIONS = (
    "k0",
    "k1_daytime_beach",
    "k2_sunset_beach",
    "k4_cat",
    "mcs",
    "inverse_square",
)
RECOVERY_CONDITIONS = ("unprompted", "daytime_beach", "sunset_beach", "cat")
SAMPLING_RATIOS = (0.01, 0.02, 0.03, 0.04, 0.05)
REPEATS = (0, 1, 2, 3, 4)
EXPECTED_ROWS = 600
DIAGNOSTIC_CMAP = "magma"
LAW_INFO = {
    "k0": {"condition": "k0", "method": "cs", "case_prefix": "sample_k0_unconditioned__recover_"},
    "k1": {"condition": "k1_daytime_beach", "method": "cs", "case_prefix": "sample_k1_daytime_beach__recover_"},
    "k2": {"condition": "k2_sunset_beach", "method": "cs", "case_prefix": "sample_k2_sunset_beach__recover_"},
    "k4": {"condition": "k4_cat", "method": "cs", "case_prefix": "sample_k4_cat__recover_"},
    "mcs": {"condition": "mcs", "method": "mcs", "case_prefix": "baseline_mcs__recover_"},
    "inverse_square": {"condition": "inverse_square", "method": "inverse_square", "case_prefix": "baseline_inverse_square__recover_"},
}
SAMPLING_LABELS = {
    "k0": r"$\widetilde{\mu}_{c_{\mathrm{uc}}}$",
    "k1_daytime_beach": r"$\widetilde{\mu}_{c_{\mathrm{db}}}$",
    "k2_sunset_beach": r"$\widetilde{\mu}_{c_{\mathrm{sb}}}$",
    "k4_cat": r"$\widetilde{\mu}_{c_{\mathrm{ca}}}$",
    "mcs": r"$\mu_{\mathrm{MCS}}$",
    "inverse_square": r"$\mu_{\mathrm{IS}}$",
}
RECOVERY_LABELS = {
    "unprompted": r"$c_r=c_{\mathrm{uc}}$",
    "daytime_beach": r"$c_r=c_{\mathrm{db}}$",
    "sunset_beach": r"$c_r=c_{\mathrm{sb}}$",
    "cat": r"$c_r=c_{\mathrm{ca}}$",
}


def _case_name(law: str, recovery_condition: str) -> str:
    """Return the stable suite-case directory for one law and recovery prompt."""

    info = LAW_INFO[law]
    suffix = recovery_condition
    if info["method"] == "cs" and recovery_condition != "unprompted":
        suffix = f"prompt_{recovery_condition}"
    return f"{info['case_prefix']}{suffix}"


def _palette(count: int, *, low: float = 0.18, high: float = 0.78) -> np.ndarray:
    """Return evenly spaced colors from the shared diagnostic colormap."""

    if count <= 0:
        return np.empty((0, 4), dtype=float)
    return plt.colormaps[DIAGNOSTIC_CMAP](np.linspace(low, high, count))


RECOVERY_COLORS = dict(zip(RECOVERY_CONDITIONS, _palette(len(RECOVERY_CONDITIONS))))


def configure_plots() -> None:
    """Use the same presentation typography as the reconstruction figures."""

    plt.rcParams.update(experiment.SD15_PRESENTATION_RC)


def _trajectory_style(recovery_condition: str) -> dict[str, object]:
    """Return the common trajectory style for one recovery condition."""

    return {
        "color": RECOVERY_COLORS[recovery_condition],
        "linewidth": 1.8,
        "alpha": 0.95,
    }


def _style_trajectory_axis(axis: plt.Axes) -> None:
    """Apply the common formatting used by optimization-trace axes."""

    axis.set_yscale("log")
    axis.grid(alpha=0.25, which="both")
    axis.tick_params(direction="out")


def load_rows(*, include_partial: bool = True) -> pd.DataFrame:
    """Load current rows and reject incompatible diagnostic artifacts."""

    # Case-filtered concurrent launches rewrite suite-level manifests. Discover
    # stable case directories directly so no active shard can hide other rows.
    frames: list[pd.DataFrame] = []
    result_root = RESULT_ROOT
    for sampling_rank, (law, info) in enumerate(LAW_INFO.items()):
        sampling_condition = str(info["condition"])
        sampling_method = str(info["method"])
        for recon_rank, recovery_condition in enumerate(RECOVERY_CONDITIONS):
            case_name = _case_name(law, recovery_condition)
            case_root = result_root / law / case_name
            if not case_root.is_dir():
                continue
            frame = experiment._load_partial_run_frame(case_root, sampling_method)
            if frame.empty:
                continue
            case_tag = (
                "results/weighted/diagnostics/out_of_range_fixed_learning_rate/"
                f"{law}/{case_name}"
            )
            frame = experiment._attach_regression_metadata(
                frame,
                sampling_method=sampling_method,
                case={
                    "name": case_name,
                    "sampling_condition": sampling_condition,
                    "sampling_label": SAMPLING_LABELS[sampling_condition],
                    "sampling_rank": sampling_rank,
                    "reconstruction_condition": recovery_condition,
                    "reconstruction_label": RECOVERY_LABELS[recovery_condition],
                    "recon_rank": recon_rank,
                },
                case_tag=case_tag,
            )
            frame["source_suite_tag"] = law
            frames.append(frame)
    rows = (
        experiment._drop_duplicate_run_rows(pd.concat(frames, ignore_index=True, sort=False))
        if frames
        else pd.DataFrame()
    )
    if rows.empty:
        return rows
    rows = recovery.attach_lpips_metrics(
        rows,
        PROJECT_ROOT,
        result_namespace="weighted",
        metrics_path=RESULT_ROOT / "lpips_metrics.csv",
    )
    if set(rows["sampling_condition"].astype(str)) - set(SAMPLING_CONDITIONS):
        raise ValueError("Unexpected sampling condition in diagnostic results.")
    if set(rows["reconstruction_condition"].astype(str)) - set(RECOVERY_CONDITIONS):
        raise ValueError("Unexpected recovery condition in diagnostic results.")
    expected_methods = {str(info["method"]) for info in LAW_INFO.values()}
    if set(rows["sampling_method"].astype(str)) - expected_methods:
        raise ValueError("Study contains an unexpected sampling method.")
    rates = rows["samp_perc"].astype(float).to_numpy()
    if not np.isclose(
        rates[:, None], np.asarray(SAMPLING_RATIOS)[None, :], rtol=0.0, atol=5e-8
    ).any(axis=1).all():
        raise ValueError("Diagnostic contains an unexpected sampling ratio.")
    return rows.reset_index(drop=True)


def completion_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one row for every expected distribution/prompt/rate/trial cell."""

    records = []
    for sampling, recovery_condition, ratio, repeat in product(
        SAMPLING_CONDITIONS,
        RECOVERY_CONDITIONS,
        SAMPLING_RATIOS,
        REPEATS,
    ):
        if rows.empty:
            observed = 0
        else:
            observed = int(
                (
                    rows["sampling_condition"].astype(str).eq(sampling)
                    & rows["reconstruction_condition"].astype(str).eq(recovery_condition)
                    & np.isclose(rows["samp_perc"].astype(float), ratio, rtol=0.0, atol=5e-8)
                    & pd.to_numeric(rows["repeat_id"], errors="coerce").eq(repeat)
                ).sum()
            )
        records.append(
            {
                "sampling_condition": sampling,
                "reconstruction_condition": recovery_condition,
                "samp_perc": ratio,
                "repeat_id": repeat,
                "observed": observed,
                "expected": 1,
                "left": max(0, 1 - observed),
                "complete": observed == 1,
            }
        )
    return pd.DataFrame.from_records(records)


def load_optimization_traces() -> pd.DataFrame:
    """Load completed and live scalar traces from every expected run leaf."""

    records: list[dict[str, object]] = []
    result_root = RESULT_ROOT
    fields = (
        "bp_loss",
        "bp_raw_resid_l2",
        "bp_weighted_resid_l2",
        "bp_grad_norm",
    )
    for law, info in LAW_INFO.items():
        condition = str(info["condition"])
        sampling_method = str(info["method"])
        for recovery_condition, ratio, repeat in product(
            RECOVERY_CONDITIONS, SAMPLING_RATIOS, REPEATS
        ):
            recovery_tag = _case_name(law, recovery_condition)
            sample_tag = f"samp_{ratio:.5f}".replace(".", "p")
            leaf = (
                result_root
                / law
                / recovery_tag
                / sampling_method
                / "item_000"
                / sample_tag
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
                arrays = {
                    field: np.asarray(payload[field], dtype=np.float64)
                    for field in fields
                }
            if any(values.shape != iterations.shape for values in arrays.values()):
                raise ValueError(f"Inconsistent trace arrays in {source}.")
            for index, iteration in enumerate(iterations):
                record: dict[str, object] = {
                    "sampling_condition": condition,
                    "reconstruction_condition": recovery_condition,
                    "samp_perc": float(ratio),
                    "repeat_id": int(repeat),
                    "iteration": int(iteration),
                    "status": status,
                    "source": str(source),
                }
                record.update({field: float(values[index]) for field, values in arrays.items()})
                records.append(record)
    return pd.DataFrame.from_records(records)


def optimization_trace_summary(traces: pd.DataFrame) -> pd.DataFrame:
    """Summarize trace means and 95% Student-t intervals at each iteration."""

    if traces.empty:
        return traces.copy()
    group_columns = [
        "sampling_condition",
        "reconstruction_condition",
        "samp_perc",
        "iteration",
    ]
    metric_columns = (
        "bp_loss",
        "bp_raw_resid_l2",
        "bp_weighted_resid_l2",
        "bp_grad_norm",
    )
    pieces = []
    grouped = traces.groupby(group_columns, sort=True, dropna=False)
    for metric in metric_columns:
        summary = grouped[metric].agg(["mean", "std", "min", "max", "count"]).reset_index()
        summary["sem"] = summary["std"] / np.sqrt(summary["count"])
        # The complete design has five trials. During partial execution, use
        # the appropriate finite-sample critical value whenever n >= 2.
        from scipy.stats import t as student_t

        critical = student_t.ppf(0.975, summary["count"] - 1)
        half_width = critical * summary["sem"]
        summary["ci_low"] = summary["mean"] - half_width
        summary["ci_high"] = summary["mean"] + half_width
        summary["metric"] = metric
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True)


def export_optimization_trace_figures(
    traces: pd.DataFrame,
    output_dir: str | Path,
    *,
    metrics: tuple[str, ...] = ("bp_loss",),
    show_confidence_interval: bool = False,
    show: bool = True,
) -> list[Path]:
    """Plot trial trajectories, arithmetic means, and optional confidence intervals."""

    if traces.empty:
        print("No optimization traces are available yet.")
        return []
    if isinstance(metrics, str):
        metrics = (metrics,)

    configure_plots()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = optimization_trace_summary(traces)
    summary.to_csv(output_root / "optimization_trace_summary.csv", index=False)

    metric_labels = {
        "bp_loss": r"Weighted Objective $\frac{1}{2C}\left\|\mathbf{A}_{\Omega}G(\mathbf{z},c)-\mathbf{y}\right\|_2^2$",
    }
    unknown_metrics = sorted(set(metrics).difference(metric_labels))
    if unknown_metrics:
        raise ValueError(f"Unknown optimization-trace metrics: {unknown_metrics}")

    outputs: list[Path] = []
    metric = "bp_loss"
    for condition in SAMPLING_CONDITIONS:
        condition_traces = traces[traces["sampling_condition"].eq(condition)]
        if condition_traces.empty:
            continue
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18.5, 11.5),
            sharex=True,
            constrained_layout=False,
        )
        plot_axes = list(axes.flat[:5])
        legend_axis = axes.flat[5]
        for axis, ratio in zip(plot_axes, SAMPLING_RATIOS):
            for recovery_condition in RECOVERY_CONDITIONS:
                raw = condition_traces[
                    condition_traces["reconstruction_condition"].eq(recovery_condition)
                    & np.isclose(condition_traces["samp_perc"], ratio, rtol=0.0, atol=5e-8)
                ]
                if raw.empty:
                    continue
                color = RECOVERY_COLORS[recovery_condition]
                for _, trial in raw.groupby("repeat_id", sort=True):
                    trial = trial.sort_values("iteration")
                    axis.plot(
                        trial["iteration"], trial[metric], color=color,
                        linestyle="-", alpha=0.20, linewidth=0.85,
                    )
                aggregate = summary[
                    summary["metric"].eq(metric)
                    & summary["sampling_condition"].eq(condition)
                    & summary["reconstruction_condition"].eq(recovery_condition)
                    & np.isclose(summary["samp_perc"], ratio, rtol=0.0, atol=5e-8)
                ].sort_values("iteration")
                x = aggregate["iteration"].to_numpy(dtype=float)
                if show_confidence_interval:
                    ci_low = aggregate["ci_low"].to_numpy(dtype=float)
                    ci_high = aggregate["ci_high"].to_numpy(dtype=float)
                    valid_ci = np.isfinite(ci_low) & np.isfinite(ci_high) & (ci_low > 0.0)
                    if np.any(valid_ci):
                        axis.fill_between(
                            x[valid_ci], ci_low[valid_ci], ci_high[valid_ci],
                            color=color, alpha=0.14, linewidth=0,
                        )
                axis.plot(
                    x, aggregate["mean"].to_numpy(dtype=float), color=color,
                    linestyle="-", linewidth=2.5,
                )
            _style_trajectory_axis(axis)
            # Shared x-axes hide upper-row tick labels by default
            axis.tick_params(axis="x", labelbottom=True)
            axis.set_title(
                rf"{SAMPLING_LABELS[condition]}, $m/N={ratio:.2f}$",
                fontweight="bold",
            )
        prompt_handles = [
            Line2D([0], [0], color=RECOVERY_COLORS[value], linewidth=2.7,
                   label=RECOVERY_LABELS[value])
            for value in RECOVERY_CONDITIONS
        ]
        trajectory_handles = [
            Line2D([0], [0], color="0.35", linewidth=0.9, alpha=0.35,
                   label="Individual Trial"),
            Line2D([0], [0], color="0.35", linewidth=2.7,
                   label="Trial Mean"),
        ]
        legend_axis.axis("off")
        legend_axis.legend(
            handles=[*prompt_handles, *trajectory_handles],
            loc="center",
            frameon=False,
            fontsize=experiment.SD15_PRESENTATION_RC["legend.fontsize"],
            handlelength=3.2,
            labelspacing=1.0,
        )
        fig.supxlabel(
            "Optimization Iteration",
            fontsize=experiment.SD15_PRESENTATION_RC["axes.labelsize"],
            y=0.025,
        )
        fig.supylabel(
            metric_labels[metric],
            fontsize=experiment.SD15_PRESENTATION_RC["axes.labelsize"],
            x=0.005,
        )
        fig.subplots_adjust(
            left=0.09,
            right=0.985,
            bottom=0.10,
            top=0.95,
            wspace=0.24,
            hspace=0.30,
        )
        output = output_root / f"bp_loss_traces_{condition}.pdf"
        fig.savefig(
            output,
            dpi=experiment.SD15_EXPORT_DPI,
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close(fig)
        outputs.append(output)

    return outputs
