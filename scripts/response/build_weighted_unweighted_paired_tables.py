#!/usr/bin/env python
"""Build paired weighted-versus-unweighted numerical response tables."""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT / "analyze_results"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import sd15_recovery_analysis as recovery  # noqa: E402


OUTPUT_ROOT = ROOT / "notestoself" / "response"
MARKDOWN_PATH = OUTPUT_ROOT / "weighted_unweighted_paired_numeric_comparison.md"
LATEX_PATH = OUTPUT_ROOT / "weighted_unweighted_paired_numeric_comparison.tex"
PAIR_CSV_PATH = OUTPUT_ROOT / "weighted_unweighted_paired_pair_summary.csv"
RATE_CSV_PATH = OUTPUT_ROOT / "weighted_unweighted_paired_rate_summary.csv"
RATE_PAIR_CSV_PATH = (
    OUTPUT_ROOT / "weighted_unweighted_paired_rate_pair_summary.csv"
)

SHARED_RATES = (0.00125, 0.0025, 0.005, 0.01)
METRICS = ("psnr_db", "ssim", "pixel_mae")
PAIR_KEYS = (
    "sampling_condition",
    "reconstruction_condition",
    "samp_perc",
    "repeat_id",
    "item_id",
)
SCENARIOS = {
    "prompt_matched": "Prompt-matched in-range",
    "prompt_mismatched": "Prompt-mismatched in-range",
    "out_of_range": "Out-of-range",
}
SAMPLING_LABEL_MD = {
    "k0": r"$\widetilde\mu_{\mathrm{uc}}$",
    "k1_daytime_beach": r"$\widetilde\mu_{\mathrm{db}}$",
    "k2_sunset_beach": r"$\widetilde\mu_{\mathrm{sb}}$",
    "k4_cat": r"$\widetilde\mu_{\mathrm{cat}}$",
}
SAMPLING_LABEL_TEX = {
    "k0": r"$\widetilde\mu_{\mathrm{uc}}$",
    "k1_daytime_beach": r"$\widetilde\mu_{\mathrm{db}}$",
    "k2_sunset_beach": r"$\widetilde\mu_{\mathrm{sb}}$",
    "k4_cat": r"$\widetilde\mu_{\mathrm{cat}}$",
}
SAMPLING_ORDER = {name: rank for rank, name in enumerate(SAMPLING_LABEL_MD)}
RECOVERY_LABEL = {
    "unprompted": "Unprompted",
    "daytime_beach": "Daytime beach",
    "sunset_beach": "Sunset beach",
    "cat": "Cat",
}
RECOVERY_ORDER = {name: rank for rank, name in enumerate(RECOVERY_LABEL)}


def paired_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario, scenario_label in SCENARIOS.items():
        unweighted, _ = recovery.load_unweighted_main_analysis(
            ROOT,
            base_tag=f"unweighted/{scenario}/sunset",
            output_root=Path("/tmp/weighted-unweighted-paired-table"),
            include_partial=True,
        )
        weighted, _ = recovery.load_weighted_main_analysis(
            ROOT,
            base_tag=f"weighted/{scenario}/sunset",
            output_root=Path("/tmp/weighted-unweighted-paired-table"),
            include_partial=True,
        )
        left = unweighted.rows[
            unweighted.rows["sampling_method"].astype(str).eq("cs")
            & unweighted.rows["samp_perc"].astype(float).isin(SHARED_RATES)
        ].copy()
        right = weighted.rows[
            weighted.rows["sampling_method"].astype(str).eq("cs")
            & weighted.rows["samp_perc"].astype(float).isin(SHARED_RATES)
        ].copy()
        columns = [*PAIR_KEYS, *METRICS]
        if left.duplicated(list(PAIR_KEYS)).any():
            raise ValueError(f"Duplicate unweighted pairing keys in {scenario}.")
        if right.duplicated(list(PAIR_KEYS)).any():
            raise ValueError(f"Duplicate weighted pairing keys in {scenario}.")
        merged = left[columns].merge(
            right[columns],
            on=list(PAIR_KEYS),
            how="inner",
            suffixes=("_unweighted", "_weighted"),
            validate="one_to_one",
        )
        merged["scenario"] = scenario
        merged["scenario_label"] = scenario_label
        for metric in METRICS:
            merged[f"{metric}_difference"] = (
                pd.to_numeric(merged[f"{metric}_weighted"], errors="coerce")
                - pd.to_numeric(merged[f"{metric}_unweighted"], errors="coerce")
            )
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def mean_ci(values: Iterable[float]) -> tuple[int, float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return 0, math.nan, math.nan, math.nan
    mean = float(array.mean())
    if count < 2:
        return count, mean, math.nan, math.nan
    sem = float(array.std(ddof=1) / math.sqrt(count))
    halfwidth = float(t.ppf(0.975, count - 1) * sem)
    return count, mean, mean - halfwidth, mean + halfwidth


def clustered_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    cluster_columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = frame.groupby(group_columns, sort=False, dropna=False)
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        record = dict(zip(group_columns, keys))
        cluster = (
            group.groupby(cluster_columns, sort=False, dropna=False)
            [
                [
                    *[f"{metric}_unweighted" for metric in METRICS],
                    *[f"{metric}_weighted" for metric in METRICS],
                    *[f"{metric}_difference" for metric in METRICS],
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        record["paired_leaves"] = int(len(group))
        record["trial_clusters"] = int(len(cluster))
        for metric in METRICS:
            record[f"{metric}_unweighted_mean"] = float(
                cluster[f"{metric}_unweighted"].mean()
            )
            record[f"{metric}_weighted_mean"] = float(
                cluster[f"{metric}_weighted"].mean()
            )
            count, mean, low, high = mean_ci(cluster[f"{metric}_difference"])
            record[f"{metric}_difference_n"] = count
            record[f"{metric}_difference_mean"] = mean
            record[f"{metric}_difference_ci_low"] = low
            record[f"{metric}_difference_ci_high"] = high
        records.append(record)
    return pd.DataFrame.from_records(records)


def rate_counts(group: pd.DataFrame) -> str:
    counts = []
    for rate in SHARED_RATES:
        count = int(
            np.isclose(
                group["samp_perc"].astype(float),
                rate,
                rtol=0.0,
                atol=5e-8,
            ).sum()
        )
        counts.append(str(count))
    return "/".join(counts)


def pair_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = clustered_summary(
        frame,
        ["scenario", "scenario_label", "sampling_condition", "reconstruction_condition"],
        ["repeat_id"],
    )
    coverage = (
        frame.groupby(
            ["scenario", "sampling_condition", "reconstruction_condition"],
            sort=False,
        )
        .apply(rate_counts, include_groups=False)
        .rename("rate_counts")
        .reset_index()
    )
    scenario_grid = pd.DataFrame.from_records(
        [
            {
                "scenario": scenario,
                "scenario_label": scenario_label,
                "sampling_condition": sampling_condition,
                "reconstruction_condition": reconstruction_condition,
            }
            for scenario, scenario_label in SCENARIOS.items()
            for sampling_condition in SAMPLING_LABEL_MD
            for reconstruction_condition in RECOVERY_LABEL
        ]
    )
    summary = scenario_grid.merge(
        summary,
        on=[
            "scenario",
            "scenario_label",
            "sampling_condition",
            "reconstruction_condition",
        ],
        how="left",
        validate="one_to_one",
    ).merge(
        coverage,
        on=["scenario", "sampling_condition", "reconstruction_condition"],
        how="left",
        validate="one_to_one",
    )
    summary["rate_counts"] = summary["rate_counts"].fillna("0/0/0/0")
    for column in (
        "paired_leaves",
        "trial_clusters",
        *[f"{metric}_difference_n" for metric in METRICS],
    ):
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        ).fillna(0).astype(int)
    summary["sampling_rank"] = summary["sampling_condition"].map(SAMPLING_ORDER)
    summary["recovery_rank"] = summary["reconstruction_condition"].map(RECOVERY_ORDER)
    return summary.sort_values(
        ["scenario", "sampling_rank", "recovery_rank"],
        kind="stable",
    ).reset_index(drop=True)


def rate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    per_scenario = clustered_summary(
        frame,
        ["scenario", "scenario_label", "samp_perc"],
        ["repeat_id"],
    )
    overall_frame = frame.copy()
    overall_frame["scenario_cluster"] = overall_frame["scenario"]
    overall_frame["scenario"] = "all"
    overall_frame["scenario_label"] = "All scenarios"
    overall = clustered_summary(
        overall_frame,
        ["scenario", "scenario_label", "samp_perc"],
        ["scenario_cluster", "item_id", "repeat_id"],
    )
    summary = pd.concat([per_scenario, overall], ignore_index=True)
    scenario_order = {name: rank for rank, name in enumerate([*SCENARIOS, "all"])}
    summary["scenario_rank"] = summary["scenario"].map(scenario_order)
    return summary.sort_values(
        ["scenario_rank", "samp_perc"],
        kind="stable",
    ).reset_index(drop=True)


def rate_pair_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = clustered_summary(
        frame,
        [
            "scenario",
            "scenario_label",
            "sampling_condition",
            "reconstruction_condition",
            "samp_perc",
        ],
        ["repeat_id"],
    )
    grid = pd.DataFrame.from_records(
        [
            {
                "scenario": scenario,
                "scenario_label": scenario_label,
                "sampling_condition": sampling_condition,
                "reconstruction_condition": reconstruction_condition,
                "samp_perc": rate,
            }
            for scenario, scenario_label in SCENARIOS.items()
            for sampling_condition in SAMPLING_LABEL_MD
            for reconstruction_condition in RECOVERY_LABEL
            for rate in SHARED_RATES
        ]
    )
    summary = grid.merge(
        summary,
        on=[
            "scenario",
            "scenario_label",
            "sampling_condition",
            "reconstruction_condition",
            "samp_perc",
        ],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "paired_leaves",
        "trial_clusters",
        *[f"{metric}_difference_n" for metric in METRICS],
    ):
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        ).fillna(0).astype(int)
    summary["sampling_rank"] = summary["sampling_condition"].map(SAMPLING_ORDER)
    summary["recovery_rank"] = summary["reconstruction_condition"].map(RECOVERY_ORDER)
    return summary.sort_values(
        ["scenario", "sampling_rank", "recovery_rank", "samp_perc"],
        kind="stable",
    ).reset_index(drop=True)


def overall_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["scope"] = "All scenarios and shared rates"
    return clustered_summary(
        work,
        ["scope"],
        ["scenario", "repeat_id"],
    )


def fmt_number(value: float, digits: int, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    if signed:
        return f"{float(value):+.{digits}f}"
    return f"{float(value):.{digits}f}"


def fmt_ci(row: pd.Series, metric: str, digits: int) -> str:
    mean = float(row[f"{metric}_difference_mean"])
    low = float(row[f"{metric}_difference_ci_low"])
    high = float(row[f"{metric}_difference_ci_high"])
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(low) or not np.isfinite(high):
        return f"{mean:+.{digits}f} [NA]"
    return f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def numerical_summary_paragraph(
    paired: pd.DataFrame,
    pairs: pd.DataFrame,
    overall: pd.DataFrame,
) -> str:
    row = overall.iloc[0]
    complete_rate_cells = 0
    total_rate_cells = len(SCENARIOS) * 4 * 4 * len(SHARED_RATES)
    for _, group in paired.groupby(
        ["scenario", "sampling_condition", "reconstruction_condition", "samp_perc"]
    ):
        complete_rate_cells += int(group["repeat_id"].nunique() == 5)
    estimable = pairs[
        pd.to_numeric(
            pairs["psnr_db_difference_mean"],
            errors="coerce",
        ).notna()
    ].copy()
    psnr_below = int(
        (
            pd.to_numeric(estimable["psnr_db_difference_ci_high"], errors="coerce")
            < 0.0
        ).sum()
    )
    psnr_above = int(
        (
            pd.to_numeric(estimable["psnr_db_difference_ci_low"], errors="coerce")
            > 0.0
        ).sum()
    )
    psnr_uncertain = int(len(estimable) - psnr_below - psnr_above)
    unavailable = int(len(pairs) - len(estimable))
    best = estimable.loc[estimable["psnr_db_difference_mean"].idxmax()]
    worst = estimable.loc[estimable["psnr_db_difference_mean"].idxmin()]
    return (
        f"Across {int(row['paired_leaves'])} currently paired repeat-level "
        f"reconstructions at the four shared sampling ratios, the weighted "
        f"pipeline changes PSNR by {fmt_ci(row, 'psnr_db', 3)} dB, SSIM by "
        f"{fmt_ci(row, 'ssim', 4)}, and per-pixel MAE by "
        f"{fmt_ci(row, 'pixel_mae', 4)} (weighted minus unweighted; 95% "
        f"Student-$t$ intervals over scenario-by-trial clusters). "
        f"{complete_rate_cells}/{total_rate_cells} distribution/recovery/rate "
        f"cells currently contain all five paired trials. Across the "
        f"{len(estimable)}/48 currently estimable "
        f"scenario/distribution/recovery summaries, {psnr_below} PSNR "
        f"intervals lie entirely below zero, {psnr_above} lie entirely above "
        f"zero, and {psnr_uncertain} cross zero; {unavailable} pair has no "
        f"paired observation yet. The least negative current "
        f"pair is {best['scenario_label']}, "
        f"{SAMPLING_LABEL_MD[str(best['sampling_condition'])]} sampling with "
        f"{RECOVERY_LABEL[str(best['reconstruction_condition'])].lower()} "
        f"recovery ({fmt_ci(best, 'psnr_db', 3)} dB); the largest deficit is "
        f"{worst['scenario_label']}, "
        f"{SAMPLING_LABEL_MD[str(worst['sampling_condition'])]} sampling with "
        f"{RECOVERY_LABEL[str(worst['reconstruction_condition'])].lower()} "
        f"recovery ({fmt_ci(worst, 'psnr_db', 3)} dB). These remain "
        f"complete-pipeline comparisons rather than an isolated estimate of "
        f"the effect of weighting."
    )


def build_markdown(
    paired: pd.DataFrame,
    pairs: pd.DataFrame,
    rates: pd.DataFrame,
    rate_pairs: pd.DataFrame,
    overall: pd.DataFrame,
    timestamp: str,
) -> str:
    summary = numerical_summary_paragraph(paired, pairs, overall)
    rate_rows: list[list[str]] = []
    for _, row in rates.iterrows():
        rate_rows.append(
            [
                str(row["scenario_label"]),
                f"{float(row['samp_perc']):.5f}".rstrip("0"),
                f"{int(row['paired_leaves'])}/{int(row['trial_clusters'])}",
                fmt_number(row["psnr_db_unweighted_mean"], 2),
                fmt_number(row["psnr_db_weighted_mean"], 2),
                fmt_ci(row, "psnr_db", 3),
                fmt_ci(row, "ssim", 4),
                fmt_ci(row, "pixel_mae", 4),
            ]
        )
    sections = [
        "# Paired numerical weighted-versus-unweighted comparison",
        "",
        f"Last generated: **{timestamp}**.",
        "",
        "## Interpretation and uncertainty convention",
        "",
        summary,
        "",
        "All differences are **weighted minus unweighted**. PSNR and SSIM favor "
        "the weighted pipeline when positive; per-pixel MAE favors it when "
        "negative. Pairing fixes the scenario, sampling law, recovery prompt, "
        "sampling ratio, item, and repeat ID. Rate-specific intervals use the "
        "five repeat IDs as clusters. Across-rate pair summaries first average "
        "the available shared rates within each repeat and then form a 95% "
        "Student-$t$ interval across repeats. The shared rates are "
        "`0.00125`, `0.0025`, `0.005`, and `0.01`.",
        "",
        "Pairing does not mean that the Fourier masks are identical: the "
        "repeat seed couples corresponding draws, but the S500 and regularized "
        "S10000 laws generally produce different sampled indices.",
        "",
        "Because the S500/S10000 artifact, probability regularization, FFT "
        "normalization, sampling masks, and least-squares objective all change, "
        "these are complete-pipeline differences—not an isolated weighting "
        "ablation. LPIPS is omitted until the unweighted LPIPS sidecar has been "
        "computed; PSNR, SSIM, and per-pixel MAE are paired directly from the "
        "saved run artifacts.",
        "",
        "## Sampling-ratio summary",
        "",
        "The `leaves/clusters` column gives the number of paired reconstruction "
        "leaves and the number of independent trial clusters used for the "
        "interval.",
        "",
        markdown_table(
            [
                "Scenario",
                "$m/n$",
                "Leaves/clusters",
                "PSNR U",
                "PSNR W",
                "$\\Delta$PSNR [95% CI]",
                "$\\Delta$SSIM [95% CI]",
                "$\\Delta$MAE [95% CI]",
            ],
            rate_rows,
        ),
    ]
    for scenario, scenario_label in SCENARIOS.items():
        subset = pairs[pairs["scenario"].eq(scenario)]
        rows: list[list[str]] = []
        for _, row in subset.iterrows():
            rows.append(
                [
                    SAMPLING_LABEL_MD[str(row["sampling_condition"])],
                    RECOVERY_LABEL[str(row["reconstruction_condition"])],
                    str(row["rate_counts"]),
                    f"{int(row['paired_leaves'])}/{int(row['trial_clusters'])}",
                    fmt_number(row["psnr_db_unweighted_mean"], 2),
                    fmt_number(row["psnr_db_weighted_mean"], 2),
                    fmt_ci(row, "psnr_db", 3),
                    fmt_ci(row, "ssim", 4),
                    fmt_ci(row, "pixel_mae", 4),
                ]
            )
        sections.extend(
            [
                "",
                f"## {scenario_label}: distribution/recovery pairs",
                "",
                "Rate counts are ordered as "
                "`0.00125 / 0.0025 / 0.005 / 0.01`; the maximum is "
                "`5/5/5/5`. The reported U and W values are paired means over "
                "the same available cells.",
                "",
                markdown_table(
                    [
                        "Sampling law",
                        "Recovery",
                        "Rate counts",
                        "Leaves/trials",
                        "PSNR U",
                        "PSNR W",
                        "$\\Delta$PSNR [95% CI]",
                        "$\\Delta$SSIM [95% CI]",
                        "$\\Delta$MAE [95% CI]",
                    ],
                    rows,
                ),
            ]
        )
        metric_titles = {
            "psnr_db": r"$\Delta$PSNR (dB)",
            "ssim": r"$\Delta$SSIM",
            "pixel_mae": r"$\Delta$ per-pixel MAE",
        }
        for metric in METRICS:
            digits = 3 if metric == "psnr_db" else 4
            matrix_rows: list[list[str]] = []
            for _, pair_row in subset.iterrows():
                cells = []
                for rate in SHARED_RATES:
                    match = rate_pairs[
                        rate_pairs["scenario"].eq(scenario)
                        & rate_pairs["sampling_condition"].eq(
                            pair_row["sampling_condition"]
                        )
                        & rate_pairs["reconstruction_condition"].eq(
                            pair_row["reconstruction_condition"]
                        )
                        & np.isclose(
                            rate_pairs["samp_perc"].astype(float),
                            rate,
                            rtol=0.0,
                            atol=5e-8,
                        )
                    ]
                    if len(match) != 1:
                        raise ValueError(
                            "Expected one rate-specific pair-summary row."
                        )
                    cells.append(fmt_ci(match.iloc[0], metric, digits))
                matrix_rows.append(
                    [
                        SAMPLING_LABEL_MD[str(pair_row["sampling_condition"])],
                        RECOVERY_LABEL[str(pair_row["reconstruction_condition"])],
                        *cells,
                        fmt_ci(pair_row, metric, digits),
                    ]
                )
            sections.extend(
                [
                    "",
                    f"### {scenario_label}: rate-specific {metric_titles[metric]}",
                    "",
                    markdown_table(
                        [
                            "Sampling law",
                            "Recovery",
                            "0.00125",
                            "0.0025",
                            "0.005",
                            "0.01",
                            "Across-rate mean [95% CI]",
                        ],
                        matrix_rows,
                    ),
                ]
            )
    sections.extend(
        [
            "",
            "## Response-ready summary paragraph",
            "",
            "> " + summary,
            "",
            "The source tables are also saved as "
            "[pair-summary CSV](weighted_unweighted_paired_pair_summary.csv) "
            "and [sampling-rate CSV]"
            "(weighted_unweighted_paired_rate_summary.csv). The complete "
            "rate-by-pair values are in [rate-pair CSV]"
            "(weighted_unweighted_paired_rate_pair_summary.csv). The matching "
            "LaTeX fragment is "
            "[weighted_unweighted_paired_numeric_comparison.tex]"
            "(weighted_unweighted_paired_numeric_comparison.tex).",
            "",
        ]
    )
    return "\n".join(sections)


def tex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def tex_ci(row: pd.Series, metric: str, digits: int) -> str:
    mean = float(row[f"{metric}_difference_mean"])
    low = float(row[f"{metric}_difference_ci_low"])
    high = float(row[f"{metric}_difference_ci_high"])
    if not np.isfinite(mean):
        return r"$\mathrm{NA}$"
    if not np.isfinite(low) or not np.isfinite(high):
        return rf"${mean:+.{digits}f}\;[\mathrm{{NA}}]$"
    return rf"${mean:+.{digits}f}\;[{low:+.{digits}f},{high:+.{digits}f}]$"


def latex_summary_paragraph(markdown_summary: str) -> str:
    text = markdown_summary.replace("Student-$t$", r"Student-$t$")
    text = text.replace("95% ", r"95\% ")
    text = text.replace("48 ", "48 ")
    for md, tex in SAMPLING_LABEL_MD.items():
        del md, tex
    # Replace Markdown sampling labels already present in the generated text.
    text = text.replace("**", "")
    return text


def latex_table(headers: list[str], rows: list[list[str]], column_spec: str) -> str:
    lines = [
        r"\begin{center}",
        r"\fontsize{7}{7.7}\selectfont",
        r"\setlength{\tabcolsep}{1.5pt}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}", r"\end{center}"])
    return "\n".join(lines)


def build_latex(
    paired: pd.DataFrame,
    pairs: pd.DataFrame,
    rates: pd.DataFrame,
    rate_pairs: pd.DataFrame,
    overall: pd.DataFrame,
    timestamp: str,
) -> str:
    summary = numerical_summary_paragraph(paired, pairs, overall)
    # Convert the few Markdown-only elements used by the generated paragraph.
    for condition, md_label in SAMPLING_LABEL_MD.items():
        summary = summary.replace(md_label, SAMPLING_LABEL_TEX[condition])
    summary = summary.replace("95% ", r"95\% ")
    summary = summary.replace("Student-$t$", r"Student-$t$")

    rate_rows: list[list[str]] = []
    for _, row in rates.iterrows():
        rate_rows.append(
            [
                tex_escape(str(row["scenario_label"])),
                f"{float(row['samp_perc']):.5f}".rstrip("0"),
                f"{int(row['paired_leaves'])}/{int(row['trial_clusters'])}",
                fmt_number(row["psnr_db_unweighted_mean"], 2),
                fmt_number(row["psnr_db_weighted_mean"], 2),
                tex_ci(row, "psnr_db", 3),
                tex_ci(row, "ssim", 4),
                tex_ci(row, "pixel_mae", 4),
            ]
        )
    sections = [
        "% Generated by scripts/response/build_weighted_unweighted_paired_tables.py",
        "% This is an input-ready LaTeX fragment and uses only standard tabular environments.",
        rf"% Last generated: {timestamp}",
        r"\subsection*{Paired numerical weighted--unweighted comparison}",
        "",
        summary,
        "",
        r"All differences are weighted minus unweighted. Positive PSNR and SSIM "
        r"differences favor the weighted pipeline; negative per-pixel MAE "
        r"differences favor it. Pairing fixes the scenario, sampling law, "
        r"recovery prompt, sampling ratio, item, and repeat identifier. "
        r"Rate-specific intervals use repeat identifiers as clusters. "
        r"Across-rate summaries first average the shared rates within each "
        r"repeat and then use a 95\% Student-$t$ interval across repeats. The "
        r"shared sampling ratios are $0.00125$, $0.0025$, $0.005$, and $0.01$.",
        "",
        r"Pairing does not imply identical Fourier masks: the repeat seed "
        r"couples corresponding draws, but the S500 and regularized S10000 "
        r"laws generally produce different sampled indices.",
        "",
        r"These are complete-pipeline comparisons rather than an isolated "
        r"weighting ablation: the S500/S10000 estimate, probability "
        r"regularization, Fourier normalization, masks, and least-squares "
        r"objective all change. LPIPS is omitted until the unweighted LPIPS "
        r"sidecar is complete.",
        "",
        r"\paragraph{Sampling-ratio summary.}",
        latex_table(
            [
                "Scenario",
                "$m/n$",
                "$N/C$",
                "PSNR U",
                "PSNR W",
                r"$\Delta$PSNR [95\% CI]",
                r"$\Delta$SSIM [95\% CI]",
                r"$\Delta$MAE [95\% CI]",
            ],
            rate_rows,
            "llrrrrll",
        ),
        r"\noindent Here $N/C$ denotes paired reconstruction leaves divided by "
        r"trial clusters.",
    ]
    for scenario, scenario_label in SCENARIOS.items():
        subset = pairs[pairs["scenario"].eq(scenario)]
        rows: list[list[str]] = []
        for _, row in subset.iterrows():
            rows.append(
                [
                    SAMPLING_LABEL_TEX[str(row["sampling_condition"])],
                    tex_escape(RECOVERY_LABEL[str(row["reconstruction_condition"])]),
                    str(row["rate_counts"]),
                    f"{int(row['paired_leaves'])}/{int(row['trial_clusters'])}",
                    fmt_number(row["psnr_db_unweighted_mean"], 2),
                    fmt_number(row["psnr_db_weighted_mean"], 2),
                    tex_ci(row, "psnr_db", 3),
                    tex_ci(row, "ssim", 4),
                    tex_ci(row, "pixel_mae", 4),
                ]
            )
        sections.extend(
            [
                "",
                rf"\paragraph{{{tex_escape(scenario_label)}.}}",
                r"The rate-count column is ordered as "
                r"$0.00125$, $0.0025$, $0.005$, and $0.01$; its maximum is "
                r"\texttt{5/5/5/5}.",
                latex_table(
                    [
                        "Law",
                        "Recovery",
                        "Rate counts",
                        "$N/T$",
                        "PSNR U",
                        "PSNR W",
                        r"$\Delta$PSNR [95\% CI]",
                        r"$\Delta$SSIM [95\% CI]",
                        r"$\Delta$MAE [95\% CI]",
                    ],
                    rows,
                    "llrrrrlll",
                ),
                r"\noindent Here $N/T$ denotes paired reconstruction leaves "
                r"divided by trial clusters.",
            ]
        )
        metric_titles = {
            "psnr_db": r"$\Delta$PSNR (dB)",
            "ssim": r"$\Delta$SSIM",
            "pixel_mae": r"$\Delta$ per-pixel MAE",
        }
        for metric in METRICS:
            digits = 3 if metric == "psnr_db" else 4
            matrix_rows: list[list[str]] = []
            for _, pair_row in subset.iterrows():
                cells = []
                for rate in SHARED_RATES:
                    match = rate_pairs[
                        rate_pairs["scenario"].eq(scenario)
                        & rate_pairs["sampling_condition"].eq(
                            pair_row["sampling_condition"]
                        )
                        & rate_pairs["reconstruction_condition"].eq(
                            pair_row["reconstruction_condition"]
                        )
                        & np.isclose(
                            rate_pairs["samp_perc"].astype(float),
                            rate,
                            rtol=0.0,
                            atol=5e-8,
                        )
                    ]
                    if len(match) != 1:
                        raise ValueError(
                            "Expected one rate-specific pair-summary row."
                        )
                    cells.append(tex_ci(match.iloc[0], metric, digits))
                matrix_rows.append(
                    [
                        SAMPLING_LABEL_TEX[str(pair_row["sampling_condition"])],
                        tex_escape(
                            RECOVERY_LABEL[
                                str(pair_row["reconstruction_condition"])
                            ]
                        ),
                        *cells,
                        tex_ci(pair_row, metric, digits),
                    ]
                )
            sections.extend(
                [
                    "",
                    rf"\noindent\textit{{Rate-specific {metric_titles[metric]}.}}",
                    latex_table(
                        [
                            "Law",
                            "Recovery",
                            "$0.00125$",
                            "$0.0025$",
                            "$0.005$",
                            "$0.01$",
                            "Across rates",
                        ],
                        matrix_rows,
                        "llrrrrr",
                    ),
                ]
            )
    sections.extend(
        [
            "",
            r"\paragraph{Response-ready summary.}",
            summary,
            "",
        ]
    )
    return "\n".join(sections)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paired = paired_rows()
    if paired.empty:
        raise RuntimeError("No weighted/unweighted CS rows could be paired.")
    pairs = pair_summary(paired)
    rates = rate_summary(paired)
    rate_pairs = rate_pair_summary(paired)
    overall = overall_summary(paired)
    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime(
        "%Y-%m-%d %H:%M %Z"
    )

    pairs.to_csv(PAIR_CSV_PATH, index=False)
    rates.to_csv(RATE_CSV_PATH, index=False)
    rate_pairs.to_csv(RATE_PAIR_CSV_PATH, index=False)
    MARKDOWN_PATH.write_text(
        build_markdown(paired, pairs, rates, rate_pairs, overall, timestamp),
        encoding="utf-8",
    )
    LATEX_PATH.write_text(
        build_latex(
            paired,
            pairs,
            rates,
            rate_pairs,
            overall,
            timestamp,
        ),
        encoding="utf-8",
    )
    print(f"paired leaves: {len(paired)}")
    print(f"pair summaries: {len(pairs)}")
    print(f"rate summaries: {len(rates)}")
    print(f"rate-pair summaries: {len(rate_pairs)}")
    print(MARKDOWN_PATH.relative_to(ROOT))
    print(LATEX_PATH.relative_to(ROOT))
    print(PAIR_CSV_PATH.relative_to(ROOT))
    print(RATE_CSV_PATH.relative_to(ROOT))
    print(RATE_PAIR_CSV_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
