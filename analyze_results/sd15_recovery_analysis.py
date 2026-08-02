"""Shared loading and plotting helpers for the SD1.5 recovery-result notebooks."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
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
WEIGHTED_MAIN_SAMPLING_METHODS: tuple[str, ...] = (
    "cs",
    "mcs",
    "inverse_square",
    "vdhh",
)
WEIGHTED_MAIN_RATES: tuple[float, ...] = (
    0.00125,
    0.0025,
    0.005,
    0.01,
    0.025,
)
WEIGHTED_MAIN_RECOVERIES: tuple[str, ...] = (
    "unprompted",
    "daytime_beach",
    "sunset_beach",
    "cat",
)
WEIGHTED_MAIN_DISTRIBUTIONS: tuple[tuple[str, str, int], ...] = (
    ("k0", "cs", 0),
    ("k1_daytime_beach", "cs", 1),
    ("k2_sunset_beach", "cs", 2),
    ("k4_cat", "cs", 3),
    ("mcs", "mcs", 4),
    ("inverse_square", "inverse_square", 5),
    ("vdhh", "vdhh", 6),
)
DEFAULT_ALLOWED_SAMPLING_PERCENTAGES: tuple[float, ...] = (
    0.00015625,
    0.0003125,
    0.000625,
    0.00125,
    0.0025,
    0.005,
    0.01,
)
UNWEIGHTED_MAIN_SAMPLING_METHODS: tuple[str, ...] = (
    "cs",
    "mcs",
    "inverse_square",
)
UNWEIGHTED_MAIN_DISTRIBUTIONS: tuple[tuple[str, str, int], ...] = (
    ("k0", "cs", 0),
    ("k1_daytime_beach", "cs", 1),
    ("k2_sunset_beach", "cs", 2),
    ("k4_cat", "cs", 3),
    ("mcs", "mcs", 4),
    ("inverse_square", "inverse_square", 5),
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
    "lpips": "LPIPS",
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
SWEEP_FIGSIZE_PER_ROW = 4.0
SWEEP_SINGLE_FIGSIZE_HEIGHT = 5.25
SWEEP_SINGLE_LEGEND_Y = 0.985
SWEEP_SINGLE_TOP = 0.76
SWEEP_SINGLE_BOTTOM = 0.28
SWEEP_SINGLE_LEFT = 0.065
SWEEP_SINGLE_RIGHT = 0.995
SWEEP_SINGLE_WSPACE = 0.18
SWEEP_SINGLE_XLABEL_Y = 0.15
SWEEP_LEGEND_Y = 1.08
SWEEP_LINEWIDTH = 2.4
SWEEP_MARKERSIZE = 7.0
SWEEP_MARKER_EDGEWIDTH = 1.3

ZERO_FILLED_METRIC_COLUMNS = {
    "psnr_db": "zero_filled_psnr_db",
    "ssim": "zero_filled_ssim",
    "lpips": "zero_filled_lpips",
    "pixel_mae": "zero_filled_pixel_mae",
}
LPIPS_METRICS_RELATIVE_PATHS = {
    "weighted": Path("results/weighted/metrics/lpips.csv"),
    "unweighted": Path("results/unweighted/metrics/lpips.csv"),
}


def _figure_slug(value: object) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower())
    return slug.strip("_")


def _output_context_slug(output_dir: str | Path) -> str:
    parts = list(Path(output_dir).parts)
    if "figures" in parts:
        parts = parts[parts.index("figures") + 1 :]
    else:
        parts = parts[-2:]
    slug_parts = [_figure_slug(part) for part in parts]
    return "_".join(part for part in slug_parts if part) or "recovery"


def _figure_filename(output_dir: str | Path, *parts: object) -> str:
    slug_parts = [_output_context_slug(output_dir), *(_figure_slug(part) for part in parts)]
    return "_".join(part for part in slug_parts if part) + ".pdf"


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


def weighted_split_tag_group_candidates(
    base_tag: str,
    *,
    distribution_suffixes: Sequence[str] = DEFAULT_DISTRIBUTION_TAG_SUFFIXES,
) -> list[tuple[str, list[str]]]:
    """Build weighted first3/last2, unsplit, and aggregate fallback groups."""

    base = str(base_tag).strip("/")
    suffixes = [str(suffix) for suffix in distribution_suffixes]
    first_tags = [f"{base}/first3_{suffix}" for suffix in suffixes]
    last_tags = [f"{base}/last2_{suffix}" for suffix in suffixes]
    unsplit_tags = [f"{base}/{suffix}" for suffix in suffixes]
    return [
        (base, first_tags + last_tags),
        (base, unsplit_tags),
        (base, [base]),
    ]


def weighted_main_tag_group_candidates(base_tag: str) -> list[tuple[str, list[str]]]:
    """Return the complete split-tag set for weighted CS and baseline runs."""

    base = str(base_tag).strip("/")
    cs_tags: list[str] = []
    for suffix in DEFAULT_DISTRIBUTION_TAG_SUFFIXES:
        cs_tags.extend(
            [
                f"{base}/first3_{suffix}",
                f"{base}/last2_{suffix}",
            ]
        )
    baseline_tags: list[str] = []
    for condition, method, _ in WEIGHTED_MAIN_DISTRIBUTIONS:
        if method == "cs":
            continue
        for recovery in WEIGHTED_MAIN_RECOVERIES:
            baseline_tags.extend(
                [
                    f"{base}/first3_{condition}_recover_{recovery}",
                    f"{base}/last2_{condition}_recover_{recovery}",
                ]
            )
    unsplit_cs = [f"{base}/{suffix}" for suffix in DEFAULT_DISTRIBUTION_TAG_SUFFIXES]
    return [
        (base, cs_tags + baseline_tags),
        (base, unsplit_cs + baseline_tags),
        (base, [base]),
    ]


def unweighted_main_tag_group_candidates(base_tag: str) -> list[tuple[str, list[str]]]:
    """Return the split-tag set for unweighted CS, MCS, and inverse-square."""

    base = str(base_tag).strip("/")
    cs_tags: list[str] = []
    for suffix in DEFAULT_DISTRIBUTION_TAG_SUFFIXES:
        cs_tags.extend(
            [
                f"{base}/first4_{suffix}",
                f"{base}/last3_{suffix}",
            ]
        )
    baseline_tags: list[str] = []
    for condition, method, _ in UNWEIGHTED_MAIN_DISTRIBUTIONS:
        if method == "cs":
            continue
        for recovery in WEIGHTED_MAIN_RECOVERIES:
            baseline_tags.extend(
                [
                    f"{base}/first4_{condition}_recover_{recovery}",
                    f"{base}/last3_{condition}_recover_{recovery}",
                ]
            )
    unsplit_cs = [f"{base}/{suffix}" for suffix in DEFAULT_DISTRIBUTION_TAG_SUFFIXES]
    return [
        (base, cs_tags + baseline_tags),
        (base, unsplit_cs + baseline_tags),
        (base, [base]),
    ]


def validate_unweighted_main_rows(frame: pd.DataFrame) -> None:
    """Reject rows that do not belong to the original unweighted pipeline."""

    if frame.empty:
        return
    expected_method = {
        condition: method
        for condition, method, _ in UNWEIGHTED_MAIN_DISTRIBUTIONS
    }
    unknown_conditions = sorted(
        set(frame["sampling_condition"].astype(str)).difference(expected_method)
    )
    if unknown_conditions:
        raise ValueError(f"Unknown unweighted sampling conditions: {unknown_conditions}")

    methods = frame["sampling_method"].astype(str)
    conditions = frame["sampling_condition"].astype(str)
    mismatched = frame[
        [
            method != expected_method[condition]
            for method, condition in zip(methods, conditions)
        ]
    ]
    if not mismatched.empty:
        raise ValueError(
            "Unweighted sampling method/condition mismatch:\n"
            + mismatched[
                ["sampling_method", "sampling_condition", "source_suite_tag"]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    recoveries = set(frame["reconstruction_condition"].astype(str))
    unknown_recoveries = sorted(recoveries.difference(WEIGHTED_MAIN_RECOVERIES))
    if unknown_recoveries:
        raise ValueError(f"Unknown unweighted recovery conditions: {unknown_recoveries}")

    if not bool(
        (
            pd.to_numeric(frame["weighted_ls"], errors="coerce").fillna(0)
            == 0
        ).all()
    ):
        raise ValueError("Unweighted-main rows must not use weighted least squares.")
    if not bool(
        frame["fft_normalization"].astype(str).str.lower().eq("backward").all()
    ):
        raise ValueError("Unweighted-main rows must use the legacy backward FFT.")
    zeta = pd.to_numeric(
        frame["probability_regularization_zeta"],
        errors="coerce",
    )
    if zeta.isna().any() or not bool(np.isclose(zeta, 0.0, rtol=0.0, atol=0.0).all()):
        raise ValueError("Unweighted-main rows must use zeta=0.")

    cs_rows = frame[methods == "cs"]
    if "ktilde_name" in cs_rows.columns:
        names = cs_rows["ktilde_name"].astype(str)
        if not bool(names.str.contains("S500", regex=False).all()):
            raise ValueError("Unweighted CS rows must use the original S500 K-tilde artifacts.")

    rates = frame["samp_perc"].astype(float).to_numpy()
    allowed = np.asarray(DEFAULT_ALLOWED_SAMPLING_PERCENTAGES, dtype=float)
    valid_rates = np.isclose(
        rates[:, None],
        allowed[None, :],
        rtol=0.0,
        atol=5e-8,
    ).any(axis=1)
    if not bool(valid_rates.all()):
        raise ValueError(
            "Unweighted rows contain unsupported sampling ratios: "
            f"{sorted(set(rates[~valid_rates].tolist()))}"
        )
    repeat_ids = pd.to_numeric(frame["repeat_id"], errors="coerce")
    if repeat_ids.isna().any() or not repeat_ids.between(0, 4).all():
        raise ValueError("Unweighted repeat ids must be integers in [0, 4].")


def unweighted_main_completion_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return observed counts for the six-law, seven-rate unweighted suite."""

    records: list[dict[str, Any]] = []
    for condition, method, rank in UNWEIGHTED_MAIN_DISTRIBUTIONS:
        for recovery in WEIGHTED_MAIN_RECOVERIES:
            for rate in DEFAULT_ALLOWED_SAMPLING_PERCENTAGES:
                for repeat_id in range(5):
                    if frame.empty:
                        observed = 0
                    else:
                        match = frame[
                            (frame["sampling_method"].astype(str) == method)
                            & (frame["sampling_condition"].astype(str) == condition)
                            & (
                                frame["reconstruction_condition"].astype(str)
                                == recovery
                            )
                            & np.isclose(
                                frame["samp_perc"].astype(float),
                                float(rate),
                                rtol=0.0,
                                atol=5e-8,
                            )
                            & (
                                pd.to_numeric(frame["repeat_id"], errors="coerce")
                                == repeat_id
                            )
                        ]
                        observed = int(len(match))
                    records.append(
                        {
                            "sampling_condition": condition,
                            "sampling_method": method,
                            "sampling_rank": int(rank),
                            "reconstruction_condition": recovery,
                            "samp_perc": float(rate),
                            "repeat_id": int(repeat_id),
                            "observed": observed,
                            "expected": 1,
                            "left": max(0, 1 - observed),
                            "complete": observed == 1,
                        }
                    )
    return pd.DataFrame.from_records(records)


def load_unweighted_main_analysis(
    sd15_root: str | Path,
    *,
    base_tag: str,
    output_root: str | Path | None = None,
    include_partial: bool = True,
) -> tuple[RecoveryAnalysis, pd.DataFrame]:
    """Load and audit original CS plus matched unweighted baseline runs."""

    analysis = load_recovery_analysis(
        sd15_root,
        tag_group_candidates=unweighted_main_tag_group_candidates(base_tag),
        sampling_methods=UNWEIGHTED_MAIN_SAMPLING_METHODS,
        allowed_sampling_percentages=DEFAULT_ALLOWED_SAMPLING_PERCENTAGES,
        include_partial=include_partial,
        output_root=output_root,
    )
    rows = analysis.rows.copy()
    if not rows.empty:
        # Historical CS CSVs predate these explicit audit columns; these values
        # are the fixed legacy defaults used to generate those artifacts.
        if "weighted_ls" not in rows.columns:
            rows["weighted_ls"] = 0
        else:
            rows["weighted_ls"] = pd.to_numeric(
                rows["weighted_ls"], errors="coerce"
            ).fillna(0)
        if "fft_normalization" not in rows.columns:
            rows["fft_normalization"] = "backward"
        else:
            rows["fft_normalization"] = rows["fft_normalization"].fillna("backward")
        if "probability_regularization_zeta" not in rows.columns:
            rows["probability_regularization_zeta"] = 0.0
        else:
            rows["probability_regularization_zeta"] = pd.to_numeric(
                rows["probability_regularization_zeta"],
                errors="coerce",
            ).fillna(0.0)
        duplicate_keys = [
            "source_suite_tag",
            "sampling_method",
            "item_id",
            "prompt_sha256",
            "samp_perc",
            "repeat_id",
        ]
        rows = rows.drop_duplicates(
            subset=[column for column in duplicate_keys if column in rows.columns],
            keep="last",
        ).reset_index(drop=True)
        rows = attach_lpips_metrics(
            rows,
            analysis.sd15_root,
            result_namespace="unweighted",
        )
    validate_unweighted_main_rows(rows)
    if rows is not analysis.rows:
        analysis = RecoveryAnalysis(
            sd15_root=analysis.sd15_root,
            active_tag=analysis.active_tag,
            loaded_tags=analysis.loaded_tags,
            output_dir=analysis.output_dir,
            rows=rows,
            mean_table=(
                exp.build_mean_metric_table(rows)
                if not rows.empty
                else pd.DataFrame()
            ),
        )
    return analysis, unweighted_main_completion_table(rows)


def validate_weighted_main_rows(frame: pd.DataFrame) -> None:
    """Reject incompatible rows before the weighted notebooks aggregate them."""

    if frame.empty:
        return
    expected_method = {
        condition: method
        for condition, method, _ in WEIGHTED_MAIN_DISTRIBUTIONS
    }
    unknown_conditions = sorted(
        set(frame["sampling_condition"].astype(str)).difference(expected_method)
    )
    if unknown_conditions:
        raise ValueError(f"Unknown weighted sampling conditions: {unknown_conditions}")

    method_mismatches = frame[
        [
            str(method) != expected_method[str(condition)]
            for method, condition in zip(
                frame["sampling_method"],
                frame["sampling_condition"],
            )
        ]
    ]
    if not method_mismatches.empty:
        bad_columns = [
            column
            for column in ["sampling_method", "sampling_condition", "source_suite_tag"]
            if column in method_mismatches.columns
        ]
        bad = method_mismatches[bad_columns].drop_duplicates()
        raise ValueError(
            "Weighted sampling method/condition mismatch:\n"
            + bad.to_string(index=False)
        )

    bad_recoveries = sorted(
        set(frame["reconstruction_condition"].astype(str)).difference(
            WEIGHTED_MAIN_RECOVERIES
        )
    )
    if bad_recoveries:
        raise ValueError(f"Unknown weighted recovery conditions: {bad_recoveries}")

    if "source_suite_tag" not in frame.columns:
        raise ValueError("Weighted rows must record their source_suite_tag.")
    cs_suffixes = {
        "k0": "sample_k0_unconditioned",
        "k1_daytime_beach": "sample_k1_daytime_beach",
        "k2_sunset_beach": "sample_k2_sunset_beach",
        "k4_cat": "sample_k4_cat",
    }
    invalid_sources: list[str] = []
    for source, method, condition, recovery in zip(
        frame["source_suite_tag"].astype(str),
        frame["sampling_method"].astype(str),
        frame["sampling_condition"].astype(str),
        frame["reconstruction_condition"].astype(str),
    ):
        source_tail = source.rstrip("/").split("/")[-1]
        if method == "cs":
            suffix = cs_suffixes[condition]
            allowed_tails = {
                suffix,
                f"first3_{suffix}",
                f"last2_{suffix}",
            }
        else:
            allowed_tails = {
                f"first3_{condition}_recover_{recovery}",
                f"last2_{condition}_recover_{recovery}",
            }
        if source_tail not in allowed_tails:
            invalid_sources.append(source)
    if invalid_sources:
        raise ValueError(
            "Weighted rows came from incompatible source tags: "
            f"{sorted(set(invalid_sources))}"
        )

    required_operator_columns = {
        "weighted_ls",
        "fft_normalization",
        "probability_regularization_zeta",
    }
    missing_operator_columns = sorted(
        required_operator_columns.difference(frame.columns)
    )
    if missing_operator_columns:
        raise ValueError(
            "Weighted rows are missing operator audit columns: "
            f"{missing_operator_columns}"
        )
    weighted_flags = pd.to_numeric(frame["weighted_ls"], errors="coerce")
    if weighted_flags.isna().any() or not bool((weighted_flags == 1).all()):
        raise ValueError("Weighted-main rows must all use weighted least squares.")
    fft_conventions = frame["fft_normalization"].astype(str).str.lower()
    if not bool((fft_conventions == "ortho").all()):
        raise ValueError("Weighted-main rows must all use the unitary FFT.")
    zeta_values = pd.to_numeric(
        frame["probability_regularization_zeta"],
        errors="coerce",
    ).to_numpy(dtype=float)
    expected_zeta = np.where(
        frame["sampling_method"].astype(str).to_numpy() == "cs",
        0.5,
        0.0,
    )
    if np.any(~np.isfinite(zeta_values)) or not np.allclose(
        zeta_values,
        expected_zeta,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Weighted-main rows use incompatible probability regularization."
        )
    if "sampling_probability_sum" in frame.columns:
        probability_sums = pd.to_numeric(
            frame["sampling_probability_sum"],
            errors="coerce",
        ).to_numpy(dtype=float)
        if np.any(~np.isfinite(probability_sums)) or not np.allclose(
            probability_sums,
            1.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("Weighted-main sampling laws must sum to one.")

    cs_rows = frame[frame["sampling_method"].astype(str) == "cs"]
    if not cs_rows.empty:
        missing_ktilde_columns = sorted(
            {"ktilde_name", "ktilde_max_samples"}.difference(cs_rows.columns)
        )
        if missing_ktilde_columns:
            raise ValueError(
                "Weighted CS rows are missing S10000 audit columns: "
                f"{missing_ktilde_columns}"
            )
        max_samples = pd.to_numeric(
            cs_rows["ktilde_max_samples"],
            errors="coerce",
        )
        artifact_names = cs_rows["ktilde_name"].astype(str)
        if (
            max_samples.isna().any()
            or not bool((max_samples == 10_000).all())
            or not bool(artifact_names.str.contains("S10000", regex=False).all())
        ):
            raise ValueError(
                "Weighted CS rows must use the named S10000 K-tilde artifacts."
            )
    rates = frame["samp_perc"].astype(float).to_numpy()
    allowed = np.asarray(WEIGHTED_MAIN_RATES, dtype=float)
    valid_rates = np.isclose(
        rates[:, None],
        allowed[None, :],
        rtol=0.0,
        atol=5e-8,
    ).any(axis=1)
    if not bool(valid_rates.all()):
        raise ValueError(
            "Weighted rows contain unsupported sampling ratios: "
            f"{sorted(set(rates[~valid_rates].tolist()))}"
        )
    repeat_ids = pd.to_numeric(frame["repeat_id"], errors="coerce")
    repeat_values = repeat_ids.to_numpy(dtype=float)
    if (
        repeat_ids.isna().any()
        or not repeat_ids.between(0, 4).all()
        or not np.allclose(repeat_values, np.rint(repeat_values), rtol=0.0, atol=0.0)
    ):
        raise ValueError("Weighted repeat ids must be integers in [0, 4].")

    duplicate_keys = [
        column
        for column in [
            "source_suite_tag",
            "sampling_method",
            "item_id",
            "prompt_sha256",
            "samp_perc",
            "repeat_id",
        ]
        if column in frame.columns
    ]
    if duplicate_keys and frame.duplicated(subset=duplicate_keys).any():
        raise ValueError("Weighted loader retained duplicate reconstruction rows.")


def weighted_main_completion_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return observed counts for all 700 method/recovery/rate/repeat cells."""

    records: list[dict[str, Any]] = []
    for condition, method, rank in WEIGHTED_MAIN_DISTRIBUTIONS:
        for recovery in WEIGHTED_MAIN_RECOVERIES:
            for rate in WEIGHTED_MAIN_RATES:
                for repeat_id in range(5):
                    if frame.empty:
                        observed = 0
                    else:
                        match = frame[
                            (frame["sampling_method"].astype(str) == method)
                            & (frame["sampling_condition"].astype(str) == condition)
                            & (
                                frame["reconstruction_condition"].astype(str)
                                == recovery
                            )
                            & np.isclose(
                                frame["samp_perc"].astype(float),
                                float(rate),
                                rtol=0.0,
                                atol=5e-8,
                            )
                            & (
                                pd.to_numeric(
                                    frame["repeat_id"],
                                    errors="coerce",
                                )
                                == repeat_id
                            )
                        ]
                        observed = int(len(match))
                    records.append(
                        {
                            "sampling_condition": condition,
                            "sampling_method": method,
                            "sampling_rank": int(rank),
                            "reconstruction_condition": recovery,
                            "samp_perc": float(rate),
                            "repeat_id": int(repeat_id),
                            "observed": observed,
                            "expected": 1,
                            "left": max(0, 1 - observed),
                            "complete": observed == 1,
                        }
                    )
    return pd.DataFrame.from_records(records)


def load_weighted_main_analysis(
    sd15_root: str | Path,
    *,
    base_tag: str,
    output_root: str | Path | None = None,
    include_partial: bool = True,
) -> tuple[RecoveryAnalysis, pd.DataFrame]:
    """Load, validate, and audit one seven-distribution weighted experiment."""

    analysis = load_recovery_analysis(
        sd15_root,
        tag_group_candidates=weighted_main_tag_group_candidates(base_tag),
        sampling_methods=WEIGHTED_MAIN_SAMPLING_METHODS,
        allowed_sampling_percentages=WEIGHTED_MAIN_RATES,
        include_partial=include_partial,
        output_root=output_root,
    )
    rows = analysis.rows
    if not rows.empty:
        duplicate_keys = [
            "source_suite_tag",
            "sampling_method",
            "item_id",
            "prompt_sha256",
            "samp_perc",
            "repeat_id",
        ]
        rows = rows.drop_duplicates(
            subset=duplicate_keys,
            keep="last",
        ).reset_index(drop=True)
        rows = attach_lpips_metrics(rows, analysis.sd15_root)
    validate_weighted_main_rows(rows)
    if rows is not analysis.rows:
        analysis = RecoveryAnalysis(
            sd15_root=analysis.sd15_root,
            active_tag=analysis.active_tag,
            loaded_tags=analysis.loaded_tags,
            output_dir=analysis.output_dir,
            rows=rows,
            mean_table=(
                exp.build_mean_metric_table(rows)
                if not rows.empty
                else pd.DataFrame()
            ),
        )
    completion = weighted_main_completion_table(rows)
    return analysis, completion


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
            # Avoid the expensive global tag-discovery scan performed while
            # formatting a missing-tag error. Weighted notebooks routinely
            # probe 32 split tags and are expected to work during partial runs.
            if not (root / "results" / str(tag)).is_dir():
                continue
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

    resolved_output_root = (
        Path(output_root)
        if output_root is not None
        else root / "results" / "figures"
    )
    output_tag = Path(active_tag)
    if (
        resolved_output_root.name == "figures"
        and output_tag.parts
        and output_tag.parts[0] == resolved_output_root.parent.name
    ):
        output_tag = Path(*output_tag.parts[1:])
    output_dir = resolved_output_root / output_tag
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
    baseline_labels = {
        "mcs": r"$\mu_{\mathrm{MCS}}$",
        "inverse_square": r"$\mu_{\mathrm{IS}}$",
        "vdhh": r"$\mu_{\mathrm{VDHH}}$",
    }
    condition_text = str(condition)
    if condition_text in baseline_labels:
        return baseline_labels[condition_text]
    return rf"${exp._mu_symbol(str(condition), hat=True)}$"


def synchronize_christoffel_y_limits(
    axes: Sequence[Any],
    sampling_conditions: Sequence[Any],
) -> None:
    """Give the four Christoffel panels one scale and each baseline its own."""

    baseline_conditions = {"mcs", "inverse_square", "vdhh"}
    christoffel_axes = [
        ax
        for ax, condition in zip(axes, sampling_conditions)
        if str(condition) not in baseline_conditions and ax.get_visible()
    ]
    if len(christoffel_axes) < 2:
        return
    limits = [ax.get_ylim() for ax in christoffel_axes]
    lower = min(float(limit[0]) for limit in limits)
    upper = max(float(limit[1]) for limit in limits)
    for ax in christoffel_axes:
        ax.set_ylim(lower, upper)


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


def style_sampling_ratio_axis(ax: Any, ticks: Sequence[float]) -> None:
    """Use only explicit decimal labels on the logarithmic sampling-ratio axis."""

    tick_values = sorted({float(value) for value in ticks})
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(FixedFormatter(sampling_tick_labels(tick_values)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.offsetText.set_visible(False)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _lpips_metrics_relative_path(result_namespace: str) -> Path:
    namespace = str(result_namespace).strip().lower()
    try:
        return LPIPS_METRICS_RELATIVE_PATHS[namespace]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported LPIPS result namespace {result_namespace!r}; "
            f"choose one of {sorted(LPIPS_METRICS_RELATIVE_PATHS)}."
        ) from exc


def attach_lpips_metrics(
    frame: pd.DataFrame,
    sd15_root: str | Path,
    *,
    result_namespace: str = "weighted",
    metrics_path: str | Path | None = None,
) -> pd.DataFrame:
    """Join incremental LPIPS values without modifying run artifacts."""

    result = frame.copy()
    for column in ("lpips", "zero_filled_lpips"):
        if column not in result.columns:
            result[column] = np.nan
    sidecar_path = (
        Path(metrics_path)
        if metrics_path is not None
        else Path(sd15_root) / _lpips_metrics_relative_path(result_namespace)
    )
    if not sidecar_path.is_absolute():
        sidecar_path = Path(sd15_root) / sidecar_path
    if result.empty or not sidecar_path.is_file():
        return result

    sidecar = pd.read_csv(sidecar_path)
    required = {"artifact_relpath", "lpips"}
    if not required.issubset(sidecar.columns):
        raise ValueError(
            f"LPIPS sidecar {sidecar_path} is missing {sorted(required.difference(sidecar.columns))}."
        )
    if sidecar["artifact_relpath"].duplicated().any():
        raise ValueError(f"LPIPS sidecar {sidecar_path} contains duplicate artifact paths.")
    sidecar = sidecar.copy()
    sidecar["artifact_relpath"] = sidecar["artifact_relpath"].astype(str)
    artifact_paths = [
        str(
            run_artifact_dir(sd15_root, row)
            .resolve()
            .relative_to(Path(sd15_root).resolve())
        )
        for _, row in result.iterrows()
    ]
    result["artifact_relpath"] = artifact_paths
    metric_columns = [
        column
        for column in ("lpips", "zero_filled_lpips")
        if column in sidecar.columns
    ]
    result = result.drop(columns=metric_columns, errors="ignore").merge(
        sidecar[["artifact_relpath", *metric_columns]],
        on="artifact_relpath",
        how="left",
        validate="many_to_one",
    )
    for column in ("lpips", "zero_filled_lpips"):
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _load_lpips_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if "artifact_relpath" not in frame:
        raise ValueError(f"{path} does not contain artifact_relpath.")
    if frame["artifact_relpath"].duplicated().any():
        raise ValueError(f"{path} contains duplicate artifact paths.")
    return {
        str(row["artifact_relpath"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def _atomic_write_lpips_records(
    path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    from tempfile import NamedTemporaryFile

    columns = [
        "artifact_relpath",
        "lpips",
        "zero_filled_lpips",
        "gt_path",
        "recon_path",
        "computed_at_utc",
        "network",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [records[key] for key in sorted(records)],
        columns=columns,
    )
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".csv",
        prefix=f".{path.stem}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary_path.replace(path)


def _lpips_image_tensor(path: Path, *, torch: Any, device: Any) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32).mul(2.0).sub(1.0)


def resolve_dataset_artifact_path(
    project_root: str | Path,
    stored_path: str | Path,
) -> Path:
    """Resolve a dataset file after its checkout has moved."""

    root = find_sd15_root(project_root)
    stored = Path(stored_path)
    if stored.is_file():
        return stored
    if "datasets" in stored.parts:
        datasets_index = stored.parts.index("datasets")
        relocated = root.joinpath(*stored.parts[datasets_index:])
        if relocated.is_file():
            return relocated
    candidates = list((root / "datasets").glob(f"*/{stored.name}"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not resolve dataset artifact {stored}; "
        f"found {len(candidates)} filename matches under {root / 'datasets'}."
    )


def _resolve_lpips_ground_truth(
    project_root: Path,
    run_dir: Path,
) -> Path:
    item_path = run_dir / "dataset_item.json"
    with item_path.open("r", encoding="utf-8") as handle:
        item = json.load(handle)
    return resolve_dataset_artifact_path(project_root, item["gt_png_path"])


def _resolve_lpips_reconstruction(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("recon_*.png"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one reconstruction PNG in {run_dir}; "
            f"found {len(candidates)}."
        )
    return candidates[0]


def ensure_lpips_metrics(
    sd15_root: str | Path,
    *,
    result_namespace: str = "weighted",
    artifact_roots: Optional[Sequence[str | Path]] = None,
    device: str = "cpu",
    force: bool = False,
    limit: Optional[int] = None,
    checkpoint_every: int = 25,
    verbose: bool = True,
    metrics_path: str | Path | None = None,
) -> pd.DataFrame:
    """Calculate missing LPIPS values used by the recovery notebooks.

    Results are refreshed incrementally in a shared metric table. CPU is the
    default so notebook analysis does not contend with active reconstruction
    jobs for GPU memory.
    """

    from datetime import datetime, timezone

    root = find_sd15_root(sd15_root)
    namespace = str(result_namespace).strip().lower()
    output_path = (
        Path(metrics_path)
        if metrics_path is not None
        else root / _lpips_metrics_relative_path(namespace)
    )
    if not output_path.is_absolute():
        output_path = root / output_path
    records = _load_lpips_records(output_path)
    search_roots = (
        [Path(path) for path in artifact_roots]
        if artifact_roots is not None
        else [root / "results" / namespace]
    )
    run_dirs = sorted(
        {
            path.parent
            for search_root in search_roots
            for path in search_root.glob("**/run_data.npz")
        }
    )
    pending = [
        run_dir
        for run_dir in run_dirs
        if force or str(run_dir.relative_to(root)) not in records
    ]
    if limit is not None:
        pending = pending[: max(0, int(limit))]
    if verbose:
        print(
            f"LPIPS: {len(run_dirs)} reconstruction artifacts discovered; "
            f"{len(records)} measured; {len(pending)} pending."
        )
    if not pending:
        return pd.DataFrame(
            [records[key] for key in sorted(records)]
        )

    try:
        import lpips
        import torch
    except ImportError as exc:
        raise ImportError(
            "LPIPS is required by the recovery notebooks. Install the pinned "
            "requirements and rerun the notebook."
        ) from exc

    device_name = str(device).lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LPIPS device 'cuda' was requested, but CUDA is unavailable.")
    torch_device = torch.device(device_name)
    model = lpips.LPIPS(net="alex").to(torch_device).eval()
    completed_since_write = 0
    with torch.inference_mode():
        for index, run_dir in enumerate(pending, start=1):
            artifact_relpath = str(run_dir.relative_to(root))
            gt_path = _resolve_lpips_ground_truth(root, run_dir)
            recon_path = _resolve_lpips_reconstruction(run_dir)
            zero_filled_path = run_dir / "zero_filled_ifft.png"
            gt = _lpips_image_tensor(
                gt_path,
                torch=torch,
                device=torch_device,
            )
            recon = _lpips_image_tensor(
                recon_path,
                torch=torch,
                device=torch_device,
            )
            if gt.shape != recon.shape:
                raise ValueError(
                    f"LPIPS image-shape mismatch for {run_dir}: "
                    f"{tuple(gt.shape)} versus {tuple(recon.shape)}."
                )
            lpips_value = float(model(gt, recon).reshape(-1)[0].item())
            zero_filled_value = np.nan
            if zero_filled_path.is_file():
                zero_filled = _lpips_image_tensor(
                    zero_filled_path,
                    torch=torch,
                    device=torch_device,
                )
                zero_filled_value = float(
                    model(gt, zero_filled).reshape(-1)[0].item()
                )
            records[artifact_relpath] = {
                "artifact_relpath": artifact_relpath,
                "lpips": lpips_value,
                "zero_filled_lpips": zero_filled_value,
                "gt_path": str(gt_path.relative_to(root)),
                "recon_path": str(recon_path.relative_to(root)),
                "computed_at_utc": datetime.now(timezone.utc).isoformat(),
                "network": "alex",
            }
            completed_since_write += 1
            if verbose:
                print(
                    f"LPIPS [{index:04d}/{len(pending):04d}] "
                    f"{lpips_value:.6f} {artifact_relpath}"
                )
            if completed_since_write >= max(1, int(checkpoint_every)):
                _atomic_write_lpips_records(output_path, records)
                completed_since_write = 0
    _atomic_write_lpips_records(output_path, records)
    return pd.DataFrame([records[key] for key in sorted(records)])


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


def _center_axis_group(
    axes_to_center: Sequence[Any],
    reference_axes: Sequence[Any],
) -> None:
    """Center equally sized axes as one horizontal group within the figure."""

    group = list(axes_to_center)
    references = list(reference_axes)
    if not group or len(references) < 2:
        return
    reference_positions = [ax.get_position() for ax in references]
    width = float(reference_positions[0].width)
    step = float(reference_positions[1].x0 - reference_positions[0].x0)
    group_width = width + step * float(len(group) - 1)
    first_x = 0.5 - 0.5 * group_width
    for index, ax in enumerate(group):
        position = ax.get_position()
        ax.set_position(
            [
                first_x + step * float(index),
                position.y0,
                width,
                position.height,
            ]
        )


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

    if frame.empty or metric not in frame.columns:
        print(f"No rows available for {metric}.")
        return None
    metric_values = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[metric_values.notna()].copy()
    if frame.empty:
        print(f"No computed {metric} values are available.")
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
    num_cases = int(len(sampling_cases))
    num_columns = min(4, max(1, num_cases))
    num_rows = int(np.ceil(float(num_cases) / float(num_columns)))

    with plt.rc_context(exp.SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(
            num_rows,
            num_columns,
            figsize=(
                SWEEP_FIGSIZE_PER_COL * num_columns,
                SWEEP_SINGLE_FIGSIZE_HEIGHT * num_rows,
            ),
            sharey=False,
            constrained_layout=False,
            squeeze=False,
        )
        flat_axes = axes.ravel()
        reverse_pyramid = (
            num_cases in (5, 6) and num_rows == 2 and num_columns == 4
        )
        legend_axis = None
        if reverse_pyramid:
            baseline_count = num_cases - 4
            panel_axes = [
                *list(axes[0, :4]),
                *list(axes[1, 1 : 1 + baseline_count]),
            ]
            axes[1, 0].set_visible(False)
            for unused_ax in axes[1, 1 + baseline_count : 3]:
                unused_ax.set_visible(False)
            legend_axis = axes[1, 3]
            legend_axis.set_axis_off()
        else:
            panel_axes = list(flat_axes[:num_cases])
            for unused_ax in flat_axes[num_cases:]:
                unused_ax.set_visible(False)

        for ax, (_, sampling_case) in zip(panel_axes, sampling_cases.iterrows()):
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
            style_sampling_ratio_axis(ax, ticks)
            ax.set_xlabel("")
            ax.set_title(sampling_mu_label(sampling_case["sampling_condition"]))
            ax.grid(True, which="major", axis="both", alpha=0.28, linestyle="--")
        synchronize_christoffel_y_limits(
            panel_axes,
            sampling_cases["sampling_condition"].tolist(),
        )
        fig.supylabel(
            METRIC_LABELS.get(metric, title_case(metric)),
            fontsize=exp.SD15_PRESENTATION_RC.get("axes.labelsize", 30),
            x=0.012,
        )
        fig.subplots_adjust(
            left=SWEEP_SINGLE_LEFT,
            right=SWEEP_SINGLE_RIGHT,
            bottom=SWEEP_SINGLE_BOTTOM,
            top=SWEEP_SINGLE_TOP if num_rows == 1 else 0.88,
            wspace=SWEEP_SINGLE_WSPACE,
            hspace=0.48 if num_rows > 1 else 0.0,
        )
        if reverse_pyramid and legend_axis is not None:
            _center_axis_group(
                [*panel_axes[4:], legend_axis],
                list(axes[0, :4]),
            )
        fig.supxlabel(
            GLOBAL_SAMPLING_X_LABEL,
            fontsize=exp.SD15_PRESENTATION_RC.get("axes.labelsize", 30),
            y=SWEEP_SINGLE_XLABEL_Y,
        )
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for legend_ax in panel_axes:
            if not legend_ax.get_visible():
                continue
            handles, labels = legend_ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label and label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        if legend_handles:
            legend_handles, legend_labels = legend_zero_filled_last(legend_handles, legend_labels)
            if legend_axis is not None:
                legend_axis.legend(
                    legend_handles,
                    legend_labels,
                    loc="center",
                    ncol=1,
                    frameon=False,
                    fontsize=exp.SD15_PRESENTATION_RC.get("legend.fontsize", 22),
                )
            else:
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

    num_cases = int(len(sampling_cases))
    case_columns = min(4, max(1, num_cases))
    case_rows = int(np.ceil(float(num_cases) / float(case_columns)))
    n_rows = len(metrics) * case_rows
    n_cols = case_columns
    reverse_pyramid = (
        num_cases in (5, 6) and case_rows == 2 and case_columns == 4
    )
    with plt.rc_context(exp.SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(SWEEP_FIGSIZE_PER_COL * n_cols, SWEEP_FIGSIZE_PER_ROW * n_rows),
            sharex=False,
            sharey=False,
            squeeze=False,
            constrained_layout=False,
        )
        for metric_idx, metric in enumerate(metrics):
            band_column = f"{metric}_ci_halfwidth"
            zero_summary = zero_filled_metric_summary(frame, metric)
            metric_panel_axes: list[Any] = []
            metric_sampling_conditions: list[Any] = []
            for case_idx, (_, sampling_case) in enumerate(sampling_cases.iterrows()):
                if reverse_pyramid and case_idx >= 4:
                    case_row = 1
                    col_idx = int(case_idx - 4 + 1)
                else:
                    case_row = int(case_idx // case_columns)
                    col_idx = int(case_idx % case_columns)
                row_idx = int(metric_idx * case_rows + case_row)
                ax = axes[row_idx, col_idx]
                metric_panel_axes.append(ax)
                metric_sampling_conditions.append(sampling_case["sampling_condition"])
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
                style_sampling_ratio_axis(ax, ticks)
                ax.set_xlabel("")
                ax.set_title(sampling_mu_label(sampling_case["sampling_condition"]))
                ax.grid(True, which="major", axis="both", alpha=0.28, linestyle="--")
            synchronize_christoffel_y_limits(
                metric_panel_axes,
                metric_sampling_conditions,
            )
            used_in_last_row = num_cases - ((case_rows - 1) * case_columns)
            if reverse_pyramid:
                last_row = int(metric_idx * case_rows + case_rows - 1)
                axes[last_row, 0].set_visible(False)
                if used_in_last_row == 1:
                    axes[last_row, 2].set_visible(False)
                axes[last_row, 3].set_axis_off()
            elif case_rows > 1 and used_in_last_row < case_columns:
                last_row = int(metric_idx * case_rows + case_rows - 1)
                for col_idx in range(used_in_last_row, case_columns):
                    axes[last_row, col_idx].set_visible(False)

        fig.subplots_adjust(
            left=0.06,
            right=0.99,
            bottom=0.14,
            top=0.95,
            wspace=0.22,
            hspace=0.70,
        )
        if reverse_pyramid:
            for metric_idx, metric in enumerate(metrics):
                first_row = int(metric_idx * case_rows)
                lower_row = first_row + 1
                _center_axis_group(
                    [
                        *[
                            axes[lower_row, col_idx]
                            for col_idx in range(1, 1 + num_cases - 4)
                        ],
                        axes[lower_row, 3],
                    ],
                    list(axes[first_row, :4]),
                )
                block_center = 1.0 - (float(metric_idx) + 0.5) / float(len(metrics))
                fig.text(
                    0.012,
                    block_center,
                    METRIC_LABELS.get(metric, title_case(metric)),
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=exp.SD15_PRESENTATION_RC.get("axes.labelsize", 30),
                )
        else:
            for metric_idx, metric in enumerate(metrics):
                first_row = int(metric_idx * case_rows)
                last_row = int(first_row + case_rows - 1)
                y_center = 0.5 * (
                    axes[first_row, 0].get_position().y1
                    + axes[last_row, 0].get_position().y0
                )
                fig.text(
                    0.012,
                    y_center,
                    METRIC_LABELS.get(metric, title_case(metric)),
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=exp.SD15_PRESENTATION_RC.get("axes.labelsize", 30),
                )

        fig.supxlabel(
            GLOBAL_SAMPLING_X_LABEL,
            fontsize=exp.SD15_PRESENTATION_RC.get("axes.labelsize", 30),
            y=0.018,
        )
        if reverse_pyramid:
            for metric_idx, _ in enumerate(metrics):
                first_row = int(metric_idx * case_rows)
                lower_row = first_row + 1
                legend_handles: list[Any] = []
                legend_labels: list[str] = []
                metric_axes = [
                    *list(axes[first_row, :4]),
                    *[
                        axes[lower_row, col_idx]
                        for col_idx in range(1, 1 + num_cases - 4)
                    ],
                ]
                for legend_source_ax in metric_axes:
                    handles, labels = legend_source_ax.get_legend_handles_labels()
                    for handle, label in zip(handles, labels):
                        if label and label not in legend_labels:
                            legend_handles.append(handle)
                            legend_labels.append(label)
                legend_handles, legend_labels = legend_zero_filled_last(
                    legend_handles,
                    legend_labels,
                )
                legend_axis = axes[lower_row, 3]
                legend_axis.legend(
                    legend_handles,
                    legend_labels,
                    loc="center",
                    ncol=1,
                    frameon=False,
                    fontsize=exp.SD15_PRESENTATION_RC.get("legend.fontsize", 22),
                )
        else:
            legend_handles: list[Any] = []
            legend_labels: list[str] = []
            for legend_ax in axes.ravel():
                if not legend_ax.get_visible():
                    continue
                handles, labels = legend_ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label and label not in legend_labels:
                        legend_handles.append(handle)
                        legend_labels.append(label)
            if legend_handles:
                legend_handles, legend_labels = legend_zero_filled_last(
                    legend_handles,
                    legend_labels,
                )
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
    sweep_metrics: Sequence[str] = ("psnr_db", "ssim", "lpips", "pixel_mae"),
    combined_metrics: Sequence[str] = ("psnr_db", "ssim"),
    combine_sampling_methods: bool = False,
    show: bool = True,
) -> list[Path]:
    """Export the metric PDFs produced by the recovery-result notebooks."""

    outputs: list[Path] = []
    output_root = Path(output_dir)
    if frame.empty:
        return outputs

    groups: list[tuple[str, pd.DataFrame]]
    if combine_sampling_methods:
        groups = [("all_sampling_laws", frame.copy())]
    else:
        groups = [
            (
                str(sampling_method),
                frame[frame["sampling_method"] == sampling_method].copy(),
            )
            for sampling_method in frame["sampling_method"].drop_duplicates().tolist()
        ]

    for sampling_method, subset in groups:
        if subset.empty:
            continue
        for metric in sweep_metrics:
            output = plot_metric_curves(
                subset,
                metric,
                output_path=output_root
                / _figure_filename(
                    output_root,
                    sampling_method,
                    metric,
                    "vs_sampling_ratio_by_recovery_prompt",
                ),
                show=show,
            )
            if output is not None:
                outputs.append(output)
        # An empty sequence explicitly disables the combined figure.  This is
        # useful for diagnostic notebooks that request only individual sweeps.
        if combined_metrics:
            output = plot_combined_metric_curves(
                subset,
                combined_metrics,
                output_path=output_root
                / _figure_filename(
                    output_root,
                    sampling_method,
                    "_".join(str(metric) for metric in combined_metrics),
                    "vs_sampling_ratio_by_recovery_prompt",
                ),
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
    return resolve_dataset_artifact_path(sd15_root, item["gt_png_path"])


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
    panel_width_in: float = 3.6,
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
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.12, wspace=0.02, hspace=0.02)
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
                best_objective = pd.to_numeric(
                    pd.Series([record.get("bp_best_loss", np.nan)]),
                    errors="coerce",
                ).iloc[0]
                objective_line = (
                    f"Objective {float(best_objective):.3e}\n"
                    if pd.notna(best_objective)
                    else ""
                )
                lpips_value = pd.to_numeric(
                    pd.Series([record.get("lpips", np.nan)]),
                    errors="coerce",
                ).iloc[0]
                lpips_line = (
                    f"LPIPS {float(lpips_value):.3f}\n"
                    if pd.notna(lpips_value)
                    else ""
                )
                metric_text = (
                    objective_line
                    + f"PSNR {float(record['psnr_db']):.2f} dB\n"
                    + f"SSIM {float(record['ssim']):.3f}\n"
                    + lpips_line
                    + f"PPMAE {float(record['pixel_mae']):.4f}"
                )
                ax.text(
                    0.02,
                    0.96,
                    metric_text,
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
    sampling_method: Optional[str] = DEFAULT_SAMPLING_METHODS[0],
    sampling_percentage: float = 0.00125,
    panel_width_in: float = 3.6,
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
        condition_rows = frame[
            frame["sampling_condition"].astype(str) == sampling_condition
        ]
        if sampling_method is None:
            condition_methods = condition_rows["sampling_method"].dropna().astype(str).unique().tolist()
            if len(condition_methods) != 1:
                raise ValueError(
                    f"Sampling condition {sampling_condition!r} maps to "
                    f"{condition_methods}; expected exactly one method."
                )
            active_sampling_method = condition_methods[0]
        else:
            active_sampling_method = str(sampling_method)
        output = plot_recovery_grid(
            frame,
            sd15_root=sd15_root,
            sampling_condition=sampling_condition,
            item_id=item_id,
            repeat_id=repeat_id,
            sampling_method=active_sampling_method,
            sampling_percentage=sampling_percentage,
            panel_width_in=panel_width_in,
            panel_height_in=panel_height_in,
            selection_metrics=selection_metrics,
            output_path=Path(output_dir)
            / _figure_filename(
                output_dir,
                active_sampling_method,
                "recovery_image_grid",
                sample_tag(float(sampling_percentage)),
                sampling_condition,
            ),
            show=show,
        )
        if output is not None:
            outputs.append(output)
    return outputs
