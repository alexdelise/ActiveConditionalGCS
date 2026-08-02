#!/usr/bin/env python
"""Build reviewer-facing K-tilde convergence and stability tables."""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT / "analyze_results"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import sd15_conditioning_experiment as experiment  # noqa: E402


OUTPUT_ROOT = ROOT / "notestoself" / "response"
MARKDOWN_PATH = OUTPUT_ROOT / "ktilde_convergence_stability_numeric.md"
LATEX_PATH = OUTPUT_ROOT / "ktilde_convergence_stability_numeric.tex"
CHECKPOINT_CSV_PATH = OUTPUT_ROOT / "ktilde_convergence_checkpoint_summary.csv"
ENDPOINT_CSV_PATH = OUTPUT_ROOT / "ktilde_convergence_endpoint_trials.csv"
STABILITY_CSV_PATH = OUTPUT_ROOT / "ktilde_convergence_stability_summary.csv"

METRICS = ("relative_l2_error", "lambda_ref_over_mu_m")
CHECKPOINTS = (10, 100, 500, 1000, 2500, 5000, 7500, 10000)
LATE_WINDOW_START = 9000
COMPARISON_START = 1000
AGGREGATE_ROLE = "all_prompts"
ROLE_ORDER = (
    "k0",
    "k1_daytime_beach",
    "k2_sunset_beach",
    "k4_cat",
    AGGREGATE_ROLE,
)
ROLE_LABEL = {
    "k0": "Unconditioned",
    "k1_daytime_beach": "Daytime beach",
    "k2_sunset_beach": "Sunset beach",
    "k4_cat": "Cat",
    AGGREGATE_ROLE: "All prompts",
}
ROLE_MATH_MD = {
    "k0": r"$c=\mathrm{uc}$",
    "k1_daytime_beach": r"$c=\mathrm{db}$",
    "k2_sunset_beach": r"$c=\mathrm{sb}$",
    "k4_cat": r"$c=\mathrm{cat}$",
    AGGREGATE_ROLE: "All prompts",
}
ROLE_MATH_TEX = {
    "k0": r"$c=\mathrm{uc}$",
    "k1_daytime_beach": r"$c=\mathrm{db}$",
    "k2_sunset_beach": r"$c=\mathrm{sb}$",
    "k4_cat": r"$c=\mathrm{cat}$",
    AGGREGATE_ROLE: r"All prompts",
}


def mean_ci(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "ci_halfwidth": math.nan,
        }
    mean = float(array.mean())
    if count < 2:
        return {
            "n": count,
            "mean": mean,
            "std": math.nan,
            "sem": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "ci_halfwidth": math.nan,
        }
    std = float(array.std(ddof=1))
    sem = std / math.sqrt(count)
    halfwidth = float(t.ppf(0.975, count - 1) * sem)
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lower": mean - halfwidth,
        "ci_upper": mean + halfwidth,
        "ci_halfwidth": halfwidth,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int = 20_000,
) -> dict[str, float | int]:
    """Return a percentile-bootstrap CI for a nonnegative derived metric."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return mean_ci(array)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else math.nan
    sem = std / math.sqrt(count) if count > 1 else math.nan
    if count < 2:
        lower = upper = mean
    else:
        rng = np.random.default_rng(seed)
        samples = array[
            rng.integers(0, count, size=(resamples, count), endpoint=False)
        ].mean(axis=1)
        lower, upper = (
            float(value) for value in np.quantile(samples, [0.025, 0.975])
        )
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_halfwidth": (upper - lower) / 2.0,
    }


def load_trial_frame() -> pd.DataFrame:
    traces = experiment.load_ktilde_convergence_trial_traces(ROOT)
    records: list[dict[str, float | int | str]] = []
    for role, info in traces.items():
        iterations = np.asarray(info["iteration"], dtype=np.int64)
        for trial in info["trials"]:
            for index, iteration in enumerate(iterations):
                records.append(
                    {
                        "role": str(role),
                        "prompt": ROLE_LABEL[str(role)],
                        "trial": int(trial["trial"]),
                        "seed": int(trial["seed"]),
                        "iteration": int(iteration),
                        **{
                            metric: float(
                                np.asarray(trial[metric], dtype=float)[index]
                            )
                            for metric in METRICS
                        },
                    }
                )
    frame = pd.DataFrame.from_records(records)
    expected_rows = 4 * 5 * 1000
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} trace rows, found {len(frame)}.")
    counts = frame.groupby(["role", "trial"]).size()
    if not bool(counts.eq(1000).all()):
        raise ValueError("Every role/trial trace must contain 1,000 checkpoints.")
    expected_iterations = np.arange(10, 10001, 10, dtype=np.int64)
    for _, group in frame.groupby(["role", "trial"]):
        if not np.array_equal(
            group.sort_values("iteration")["iteration"].to_numpy(dtype=np.int64),
            expected_iterations,
        ):
            raise ValueError("Unexpected trace iteration grid.")
    return frame


def prompt_balanced_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        frame.groupby(["trial", "seed", "iteration"], as_index=False)[list(METRICS)]
        .mean()
    )
    aggregate["role"] = AGGREGATE_ROLE
    aggregate["prompt"] = ROLE_LABEL[AGGREGATE_ROLE]
    return aggregate[
        ["role", "prompt", "trial", "seed", "iteration", *METRICS]
    ]


def checkpoint_summary(full_frame: pd.DataFrame) -> pd.DataFrame:
    subset = full_frame[full_frame["iteration"].isin(CHECKPOINTS)]
    records: list[dict[str, float | int | str]] = []
    for (role, prompt, iteration), group in subset.groupby(
        ["role", "prompt", "iteration"],
        sort=False,
    ):
        for metric in METRICS:
            records.append(
                {
                    "role": role,
                    "prompt": prompt,
                    "iteration": int(iteration),
                    "metric": metric,
                    **mean_ci(group[metric].to_numpy(dtype=float)),
                }
            )
    summary = pd.DataFrame.from_records(records)
    summary["role_rank"] = summary["role"].map(
        {role: rank for rank, role in enumerate(ROLE_ORDER)}
    )
    summary["metric_rank"] = summary["metric"].map(
        {metric: rank for rank, metric in enumerate(METRICS)}
    )
    return summary.sort_values(
        ["metric_rank", "role_rank", "iteration"],
        kind="stable",
    ).reset_index(drop=True)


def endpoint_trials(frame: pd.DataFrame) -> pd.DataFrame:
    endpoint = frame[frame["iteration"].eq(10000)].copy()
    endpoint["role_rank"] = endpoint["role"].map(
        {role: rank for rank, role in enumerate(ROLE_ORDER)}
    )
    return endpoint.sort_values(["role_rank", "trial"], kind="stable").reset_index(
        drop=True
    )


def stability_summary(full_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for (role, prompt), group in full_frame.groupby(["role", "prompt"], sort=False):
        for metric in METRICS:
            endpoints = group[group["iteration"].eq(10000)].sort_values("trial")
            endpoint_stats = mean_ci(endpoints[metric].to_numpy(dtype=float))
            endpoint_mean = float(endpoint_stats["mean"])
            endpoint_cv = (
                100.0 * float(endpoint_stats["std"]) / endpoint_mean
                if endpoint_mean > 0.0
                else math.nan
            )
            relative_ci_halfwidth = (
                100.0 * float(endpoint_stats["ci_halfwidth"]) / endpoint_mean
                if endpoint_mean > 0.0
                else math.nan
            )

            changes: list[float] = []
            relative_changes: list[float] = []
            late_ranges: list[float] = []
            late_endpoint_changes: list[float] = []
            for _, trial_group in group.groupby("trial", sort=True):
                trial_group = trial_group.sort_values("iteration")
                start = float(
                    trial_group.loc[
                        trial_group["iteration"].eq(COMPARISON_START),
                        metric,
                    ].iloc[0]
                )
                final = float(
                    trial_group.loc[
                        trial_group["iteration"].eq(10000),
                        metric,
                    ].iloc[0]
                )
                late_start = float(
                    trial_group.loc[
                        trial_group["iteration"].eq(LATE_WINDOW_START),
                        metric,
                    ].iloc[0]
                )
                late = trial_group[
                    trial_group["iteration"].between(
                        LATE_WINDOW_START,
                        10000,
                    )
                ][metric].to_numpy(dtype=float)
                changes.append(final - start)
                relative_changes.append(100.0 * (final - start) / start)
                late_ranges.append(100.0 * (float(late.max()) - float(late.min())) / final)
                late_endpoint_changes.append(
                    100.0 * (final - late_start) / late_start
                )

            change_stats = mean_ci(np.asarray(changes))
            relative_change_stats = mean_ci(np.asarray(relative_changes))
            role_seed = ROLE_ORDER.index(role) + 1
            metric_seed = METRICS.index(metric) + 1
            late_range_stats = bootstrap_mean_ci(
                np.asarray(late_ranges),
                seed=10_000 * role_seed + metric_seed,
            )
            late_change_stats = mean_ci(np.asarray(late_endpoint_changes))
            records.append(
                {
                    "role": role,
                    "prompt": prompt,
                    "metric": metric,
                    **{
                        f"endpoint_{key}": value
                        for key, value in endpoint_stats.items()
                    },
                    "endpoint_cv_percent": endpoint_cv,
                    "endpoint_ci_halfwidth_percent": relative_ci_halfwidth,
                    **{
                        f"change_1000_to_10000_{key}": value
                        for key, value in change_stats.items()
                    },
                    **{
                        f"relative_change_1000_to_10000_percent_{key}": value
                        for key, value in relative_change_stats.items()
                    },
                    **{
                        f"late_range_percent_{key}": value
                        for key, value in late_range_stats.items()
                    },
                    **{
                        f"late_change_9000_to_10000_percent_{key}": value
                        for key, value in late_change_stats.items()
                    },
                }
            )
    summary = pd.DataFrame.from_records(records)
    summary["role_rank"] = summary["role"].map(
        {role: rank for rank, role in enumerate(ROLE_ORDER)}
    )
    summary["metric_rank"] = summary["metric"].map(
        {metric: rank for rank, metric in enumerate(METRICS)}
    )
    return summary.sort_values(
        ["metric_rank", "role_rank"],
        kind="stable",
    ).reset_index(drop=True)


def fmt_ci(
    mean: float,
    lower: float,
    upper: float,
    digits: int,
    *,
    signed: bool = False,
) -> str:
    if not all(np.isfinite([mean, lower, upper])):
        return "NA"
    sign = "+" if signed else ""
    return (
        f"{mean:{sign}.{digits}f} "
        f"[{lower:{sign}.{digits}f}, {upper:{sign}.{digits}f}]"
    )


def row_ci(row: pd.Series, prefix: str, digits: int, *, signed: bool = False) -> str:
    return fmt_ci(
        float(row[f"{prefix}_mean"]),
        float(row[f"{prefix}_ci_lower"]),
        float(row[f"{prefix}_ci_upper"]),
        digits,
        signed=signed,
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def checkpoint_matrix(
    checkpoints: pd.DataFrame,
    metric: str,
    *,
    latex: bool,
) -> list[list[str]]:
    digits = 3 if metric == "relative_l2_error" else 1
    rows: list[list[str]] = []
    for role in ROLE_ORDER:
        role_rows = checkpoints[
            checkpoints["role"].eq(role) & checkpoints["metric"].eq(metric)
        ]
        values = []
        for iteration in CHECKPOINTS:
            match = role_rows[role_rows["iteration"].eq(iteration)]
            if len(match) != 1:
                raise ValueError("Missing checkpoint-summary row.")
            row = match.iloc[0]
            cell = fmt_ci(
                float(row["mean"]),
                float(row["ci_lower"]),
                float(row["ci_upper"]),
                digits,
            )
            values.append(tex_math_ci(cell) if latex else cell)
        rows.append(
            [
                ROLE_MATH_TEX[role] if latex else ROLE_MATH_MD[role],
                *values,
            ]
        )
    return rows


def tex_math_ci(value: str) -> str:
    if value == "NA":
        return r"$\mathrm{NA}$"
    mean, interval = value.split(" ", 1)
    return f"${mean}\\;{interval.replace(' ', '')}$"


def endpoint_trial_matrix(
    endpoint: pd.DataFrame,
    metric: str,
    *,
    latex: bool,
) -> list[list[str]]:
    digits = 3 if metric == "relative_l2_error" else 1
    rows: list[list[str]] = []
    for role in ROLE_ORDER[:-1]:
        group = endpoint[
            endpoint["role"].eq(role)
        ].sort_values("trial")
        if len(group) != 5:
            raise ValueError(f"Expected five endpoint trials for {role}.")
        values = [f"{float(value):.{digits}f}" for value in group[metric]]
        stats = mean_ci(group[metric].to_numpy(dtype=float))
        summary = fmt_ci(
            float(stats["mean"]),
            float(stats["ci_lower"]),
            float(stats["ci_upper"]),
            digits,
        )
        rows.append(
            [
                ROLE_MATH_TEX[role] if latex else ROLE_MATH_MD[role],
                *values,
                tex_math_ci(summary) if latex else summary,
            ]
        )
    return rows


def stability_rows(
    stability: pd.DataFrame,
    metric: str,
    *,
    latex: bool,
) -> list[list[str]]:
    digits = 3 if metric == "relative_l2_error" else 1
    rows: list[list[str]] = []
    for role in ROLE_ORDER:
        match = stability[
            stability["role"].eq(role) & stability["metric"].eq(metric)
        ]
        if len(match) != 1:
            raise ValueError("Missing stability-summary row.")
        row = match.iloc[0]
        endpoint = row_ci(row, "endpoint", digits)
        relative_change = row_ci(
            row,
            "relative_change_1000_to_10000_percent",
            1,
            signed=True,
        )
        late_range = row_ci(row, "late_range_percent", 2)
        late_change = row_ci(
            row,
            "late_change_9000_to_10000_percent",
            2,
            signed=True,
        )
        percent = r"\%" if latex else "%"
        cells = [
            ROLE_MATH_TEX[role] if latex else ROLE_MATH_MD[role],
            tex_math_ci(endpoint) if latex else endpoint,
            f"{float(row['endpoint_cv_percent']):.1f}{percent}",
            f"{float(row['endpoint_ci_halfwidth_percent']):.1f}{percent}",
            tex_math_ci(relative_change) if latex else relative_change,
            tex_math_ci(late_change) if latex else late_change,
            tex_math_ci(late_range) if latex else late_range,
        ]
        rows.append(cells)
    return rows


def narrative(
    stability: pd.DataFrame,
    checkpoints: pd.DataFrame,
    *,
    latex: bool,
) -> str:
    l2 = stability[
        stability["metric"].eq("relative_l2_error")
        & ~stability["role"].eq(AGGREGATE_ROLE)
    ]
    lam = stability[
        stability["metric"].eq("lambda_ref_over_mu_m")
        & ~stability["role"].eq(AGGREGATE_ROLE)
    ]
    aggregate_l2 = stability[
        stability["metric"].eq("relative_l2_error")
        & stability["role"].eq(AGGREGATE_ROLE)
    ].iloc[0]
    aggregate_lam = stability[
        stability["metric"].eq("lambda_ref_over_mu_m")
        & stability["role"].eq(AGGREGATE_ROLE)
    ].iloc[0]
    best_l2 = l2.loc[l2["endpoint_mean"].idxmin()]
    best_lam = lam.loc[lam["endpoint_mean"].idxmin()]
    least_stable_l2 = l2.loc[l2["late_range_percent_mean"].idxmax()]
    remaining_l2 = l2.drop(index=least_stable_l2.name)
    max_late_l2_remaining = float(remaining_l2["late_range_percent_mean"].max())
    max_late_l2 = float(least_stable_l2["late_range_percent_mean"])
    max_late_lam = float(lam["late_range_percent_mean"].max())
    aggregate_l2_endpoint = row_ci(aggregate_l2, "endpoint", 3)
    aggregate_lam_endpoint = row_ci(aggregate_lam, "endpoint", 1)
    aggregate_l2_change = row_ci(
        aggregate_l2,
        "relative_change_1000_to_10000_percent",
        1,
        signed=True,
    )
    aggregate_lam_change = row_ci(
        aggregate_lam,
        "relative_change_1000_to_10000_percent",
        1,
        signed=True,
    )
    text = (
        f"Across the four prompts, the prompt-balanced endpoint is "
        f"{aggregate_l2_endpoint} for relative L2 error and "
        f"{aggregate_lam_endpoint} for the reference compatibility statistic. "
        f"From $M=1,000$ to $M=10,000$, the corresponding paired changes are "
        f"{aggregate_l2_change}% and {aggregate_lam_change}%, respectively. "
        f"The smallest prompt-specific endpoint means occur for "
        f"{ROLE_LABEL[str(best_l2['role'])].lower()} in relative L2 "
        f"({float(best_l2['endpoint_mean']):.3f}) and "
        f"{ROLE_LABEL[str(best_lam['role'])].lower()} in the compatibility "
        f"statistic ({float(best_lam['endpoint_mean']):.1f}). Over the final "
        f"1,000 iterations, the mean within-trial compatibility-statistic "
        f"range is at most {max_late_lam:.2f}% for every prompt. The relative "
        f"L2 range is at most {max_late_l2_remaining:.2f}% for three prompts, "
        f"while {ROLE_LABEL[str(least_stable_l2['role'])].lower()} retains a "
        f"larger {max_late_l2:.2f}% range. Thus the tail-sensitive statistic "
        f"is locally stable for all four prompts, whereas the bulk relative L2 "
        f"diagnostic retains noticeable late-stage variability for one prompt. "
        f"The nonzero endpoint confidence intervals also quantify variation "
        f"among independent S10000 seed blocks."
    )
    if latex:
        text = text.replace("relative L2", r"relative $\ell^2$")
        text = text.replace("%", r"\%")
    else:
        text = text.replace("relative L2", r"relative $\ell^2$")
    return text


def latex_table(headers: list[str], rows: list[list[str]], spec: str) -> str:
    lines = [
        r"\begin{center}",
        r"\fontsize{7}{7.7}\selectfont",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [r"\hline", r"\end{tabular}%", r"}", r"\end{center}"]
    )
    return "\n".join(lines)


def build_markdown(
    checkpoints: pd.DataFrame,
    endpoint: pd.DataFrame,
    stability: pd.DataFrame,
    timestamp: str,
) -> str:
    summary = narrative(stability, checkpoints, latex=False)
    checkpoint_headers = ["Prompt", *[f"$M={value:,}$" for value in CHECKPOINTS]]
    endpoint_headers = [
        "Prompt",
        "Trial 1",
        "Trial 2",
        "Trial 3",
        "Trial 4",
        "Trial 5",
        "Mean [95% CI]",
    ]
    stability_headers = [
        "Prompt",
        "Endpoint [95% CI]",
        "Endpoint CV",
        "CI half-width / mean",
        "$\\Delta_{1000\\to10000}$ (%)",
        "$\\Delta_{9000\\to10000}$ (%)",
        "Final-window range (%)",
    ]
    return "\n".join(
        [
            "# S10000 K-tilde convergence and stability digest",
            "",
            f"Last generated: **{timestamp}**.",
            "",
            "## Reviewer-facing takeaway",
            "",
            summary,
            "",
            "The study uses the four saved S10000 estimates as fixed references "
            "and five independent S10000 trials per prompt. Scalar metrics are "
            "saved every 10 iterations. All reported means are ordinary "
            "arithmetic means. Endpoint, checkpoint, and signed-change 95% "
            "intervals are Student-$t$ intervals computed on the original "
            "metric scale across the five trials. The nonnegative final-window "
            "range uses a percentile-bootstrap interval across trials, avoiding "
            "an impossible negative lower bound without clipping. "
            "Logarithmic axes used in the figures do not change the averaging "
            "or uncertainty calculation. The `All prompts` row first averages "
            "the four prompts within each shared trial seed, then computes its "
            "interval across the five paired seed blocks.",
            "",
            "The relative $\\ell^2$ error measures bulk disagreement between "
            "an independent estimate and the fixed S10000 reference. It need "
            "not converge to zero because both are finite-sample estimates. "
            "For prompt $c$ and trial $r$, it is",
            "",
            "$$",
            "E^{(r)}_{2,c}(M)="
            "\\frac{\\left\\|\\widetilde K^{(r)}_{c,M}-"
            "\\widetilde K^\\star_c\\right\\|_2}"
            "{\\left\\|\\widetilde K^\\star_c\\right\\|_2}.",
            "$$",
            "",
            "The statistic $\\widetilde\\Lambda(M)$ fixes the reference "
            "Christoffel estimate in the numerator and places the iteration-$M$, "
            "$\\zeta=1/2$ regularized sampling law in the denominator. Because "
            "it is a maximum ratio, it directly probes the small-probability "
            "tail emphasized by the reviewers. Its relevant convergence "
            "criterion is stabilization with $M$, not convergence to one.",
            "",
            "$$",
            "\\widetilde\\mu^{(r)}_{c,M}(i)="
            "\\frac{1}{2}\\frac{\\widetilde K^{(r)}_{c,M}(i)}"
            "{\\sum_j\\widetilde K^{(r)}_{c,M}(j)}+\\frac{1}{2n},"
            "\\qquad"
            "\\widetilde\\Lambda^{(r)}_c(M)="
            "\\max_i\\frac{\\widetilde K^\\star_{c,\\mathrm{u}}(i)}"
            "{\\widetilde\\mu^{(r)}_{c,M}(i)}.",
            "$$",
            "",
            "Here the subscript $\\mathrm{u}$ means that the fixed reference "
            "Fourier-energy estimate is expressed under the unitary FFT "
            "convention used by the weighted operator.",
            "",
            "## Relative $\\ell^2$ convergence checkpoints",
            "",
            markdown_table(
                checkpoint_headers,
                checkpoint_matrix(
                    checkpoints,
                    "relative_l2_error",
                    latex=False,
                ),
            ),
            "",
            "## $\\widetilde\\Lambda(M)$ convergence checkpoints",
            "",
            markdown_table(
                checkpoint_headers,
                checkpoint_matrix(
                    checkpoints,
                    "lambda_ref_over_mu_m",
                    latex=False,
                ),
            ),
            "",
            "## Five independent S10000 endpoints",
            "",
            "### Relative $\\ell^2$ error",
            "",
            markdown_table(
                endpoint_headers,
                endpoint_trial_matrix(
                    endpoint,
                    "relative_l2_error",
                    latex=False,
                ),
            ),
            "",
            "### $\\widetilde\\Lambda(10{,}000)$",
            "",
            markdown_table(
                endpoint_headers,
                endpoint_trial_matrix(
                    endpoint,
                    "lambda_ref_over_mu_m",
                    latex=False,
                ),
            ),
            "",
            "## Aggregated convergence and late-stage stability",
            "",
            "The endpoint CV measures between-seed dispersion at $M=10{,}000$. "
            "The paired $M=1{,}000\\to10{,}000$ change measures longer-horizon "
            "convergence. The $M=9{,}000\\to10{,}000$ change and final-window "
            "range measure local drift within each trial over its last 1,000 "
            "iterations. Percentages are computed within trial before averaging.",
            "",
            "### Relative $\\ell^2$ diagnostics",
            "",
            markdown_table(
                stability_headers,
                stability_rows(
                    stability,
                    "relative_l2_error",
                    latex=False,
                ),
            ),
            "",
            "### $\\widetilde\\Lambda(M)$ diagnostics",
            "",
            markdown_table(
                stability_headers,
                stability_rows(
                    stability,
                    "lambda_ref_over_mu_m",
                    latex=False,
                ),
            ),
            "",
            "## Interpretation",
            "",
            "- Checkpoint means decrease substantially overall for both the "
            "bulk $\\ell^2$ diagnostic and the tail-sensitive maximum ratio. "
            "The maximum ratio is not monotone at every checkpoint, so the "
            "evidence is the overall downward trend together with its "
            "late-window stability.",
            "- The final-window drift diagnostics distinguish a curve that has "
            "merely decreased from one that has actually stabilized near "
            "$M=10{,}000$.",
            "- Nonzero endpoint CVs and confidence intervals quantify the "
            "remaining seed sensitivity. They should not be described as exact "
            "agreement with the fixed reference.",
            "- The $\\zeta=1/2$ mixture prevents tiny empirical probabilities "
            "from making the max ratio numerically undefined, while preserving "
            "its sensitivity to tail mismatch.",
            "- These trials vary the latent/secant seed blocks. They do not "
            "constitute a separate sweep over optimizer initialization or the "
            "number of optimization restarts, which should remain explicitly "
            "identified as outside the scope of this diagnostic.",
            "",
            "## Response-ready paragraph",
            "",
            "> " + summary,
            "",
            "Source data are saved in "
            "[checkpoint-summary CSV](ktilde_convergence_checkpoint_summary.csv), "
            "[endpoint-trial CSV](ktilde_convergence_endpoint_trials.csv), and "
            "[stability-summary CSV](ktilde_convergence_stability_summary.csv). "
            "The matching LaTeX fragment is "
            "[ktilde_convergence_stability_numeric.tex]"
            "(ktilde_convergence_stability_numeric.tex).",
            "",
        ]
    )


def build_latex(
    checkpoints: pd.DataFrame,
    endpoint: pd.DataFrame,
    stability: pd.DataFrame,
    timestamp: str,
) -> str:
    summary = narrative(stability, checkpoints, latex=True)
    checkpoint_headers = ["Prompt", *[f"$M={value:,}$" for value in CHECKPOINTS]]
    endpoint_headers = [
        "Prompt",
        "Trial 1",
        "Trial 2",
        "Trial 3",
        "Trial 4",
        "Trial 5",
        "Mean [95\\% CI]",
    ]
    stability_headers = [
        "Prompt",
        "Endpoint [95\\% CI]",
        "CV",
        "CI/mean",
        "$\\Delta_{1k\\to10k}$ (\\%)",
        "$\\Delta_{9k\\to10k}$ (\\%)",
        "Range (\\%)",
    ]
    sections = [
        "% Generated by scripts/response/build_ktilde_convergence_stability_tables.py",
        "% Input-ready fragment; requires graphicx for table resizing.",
        f"% Last generated: {timestamp}",
        r"\subsection*{S10000 K-tilde convergence and stability}",
        "",
        summary,
        "",
        r"The four saved S10000 estimates are fixed references, and five "
        r"independent S10000 trials are evaluated per prompt. Metrics are saved "
        r"every 10 iterations. Means are arithmetic means. Endpoint, checkpoint, "
        r"and signed-change 95\% intervals are Student-$t$ intervals on the "
        r"original metric scale. The nonnegative final-window range uses a "
        r"percentile-bootstrap interval across trials, avoiding an impossible "
        r"negative lower bound without clipping. Logarithmic plotting does not "
        r"change the calculation. The ``All prompts'' row first averages the "
        r"four prompts within each shared trial seed, then computes its interval "
        r"across the five paired seed blocks.",
        "",
        r"The relative $\ell^2$ error measures bulk disagreement with the "
        r"fixed finite-sample reference and need not approach zero:",
        r"\begin{equation*}",
        r"E^{(r)}_{2,c}(M)="
        r"\frac{\left\|\widetilde K^{(r)}_{c,M}-\widetilde K^\star_c"
        r"\right\|_2}{\left\|\widetilde K^\star_c\right\|_2}.",
        r"\end{equation*}",
        r"The "
        r"statistic $\widetilde\Lambda(M)$ fixes the reference Christoffel "
        r"estimate in the numerator and uses the iteration-$M$, $\zeta=1/2$ "
        r"regularized law in the denominator. Since it is a maximum ratio, it "
        r"probes the low-probability tail; stabilization, rather than "
        r"convergence to one, is the relevant diagnostic.",
        r"\begin{equation*}",
        r"\widetilde\mu^{(r)}_{c,M}(i)="
        r"\frac{1}{2}\frac{\widetilde K^{(r)}_{c,M}(i)}"
        r"{\sum_j\widetilde K^{(r)}_{c,M}(j)}+\frac{1}{2n},"
        r"\qquad"
        r"\widetilde\Lambda^{(r)}_c(M)="
        r"\max_i\frac{\widetilde K^\star_{c,\mathrm{u}}(i)}"
        r"{\widetilde\mu^{(r)}_{c,M}(i)}.",
        r"\end{equation*}",
        r"Here the subscript $\mathrm{u}$ means that the fixed reference "
        r"Fourier-energy estimate is expressed under the unitary FFT convention "
        r"used by the weighted operator.",
        "",
        r"\paragraph{Relative $\ell^2$ checkpoints.}",
        latex_table(
            checkpoint_headers,
            checkpoint_matrix(
                checkpoints,
                "relative_l2_error",
                latex=True,
            ),
            "lrrrrrrrr",
        ),
        r"\paragraph{$\widetilde\Lambda(M)$ checkpoints.}",
        latex_table(
            checkpoint_headers,
            checkpoint_matrix(
                checkpoints,
                "lambda_ref_over_mu_m",
                latex=True,
            ),
            "lrrrrrrrr",
        ),
        r"\paragraph{Five independent S10000 endpoints: relative $\ell^2$.}",
        latex_table(
            endpoint_headers,
            endpoint_trial_matrix(
                endpoint,
                "relative_l2_error",
                latex=True,
            ),
            "lrrrrrr",
        ),
        r"\paragraph{Five independent S10000 endpoints: "
        r"$\widetilde\Lambda(10{,}000)$.}",
        latex_table(
            endpoint_headers,
            endpoint_trial_matrix(
                endpoint,
                "lambda_ref_over_mu_m",
                latex=True,
            ),
            "lrrrrrr",
        ),
        r"\paragraph{Aggregated relative $\ell^2$ stability diagnostics.}",
        latex_table(
            stability_headers,
            stability_rows(
                stability,
                "relative_l2_error",
                latex=True,
            ),
            "lrrrrrr",
        ),
        r"\paragraph{Aggregated $\widetilde\Lambda(M)$ stability diagnostics.}",
        latex_table(
            stability_headers,
            stability_rows(
                stability,
                "lambda_ref_over_mu_m",
                latex=True,
            ),
            "lrrrrrr",
        ),
        r"The endpoint CV measures between-seed dispersion. The "
        r"$M=1{,}000\to10{,}000$ change measures longer-horizon convergence, "
        r"while the $M=9{,}000\to10{,}000$ change and final-window range "
        r"measure local late-stage drift. Percentages are calculated within "
        r"trial before averaging.",
        "",
        r"The five trials vary the latent/secant seed blocks. They do not "
        r"constitute a separate sweep over optimizer initialization or the "
        r"number of optimization restarts, which remains outside the scope of "
        r"this diagnostic.",
        "",
        r"\paragraph{Response-ready summary.}",
        summary,
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    trials = load_trial_frame()
    aggregate = prompt_balanced_aggregate(trials)
    full = pd.concat([trials, aggregate], ignore_index=True)
    checkpoints = checkpoint_summary(full)
    endpoint = endpoint_trials(full)
    stability = stability_summary(full)
    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime(
        "%Y-%m-%d %H:%M %Z"
    )

    checkpoints.to_csv(CHECKPOINT_CSV_PATH, index=False)
    endpoint.to_csv(ENDPOINT_CSV_PATH, index=False)
    stability.to_csv(STABILITY_CSV_PATH, index=False)
    MARKDOWN_PATH.write_text(
        build_markdown(checkpoints, endpoint, stability, timestamp),
        encoding="utf-8",
    )
    LATEX_PATH.write_text(
        build_latex(checkpoints, endpoint, stability, timestamp),
        encoding="utf-8",
    )
    print(f"trace rows: {len(trials)}")
    print(f"checkpoint rows: {len(checkpoints)}")
    print(f"endpoint rows: {len(endpoint)}")
    print(f"stability rows: {len(stability)}")
    for path in (
        MARKDOWN_PATH,
        LATEX_PATH,
        CHECKPOINT_CSV_PATH,
        ENDPOINT_CSV_PATH,
        STABILITY_CSV_PATH,
    ):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
