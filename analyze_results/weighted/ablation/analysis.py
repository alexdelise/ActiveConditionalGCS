"""Analysis helpers for the two-trial weighted recovery-CFG ablation."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

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
    "unconditioned": {"label": "Unconditioned", "rank": 0, "cfg_scale": np.nan},
    "cfg1": {"label": "CFG 1", "rank": 1, "cfg_scale": 1.0},
    "cfg1p5": {"label": "CFG 1.5", "rank": 2, "cfg_scale": 1.5},
    "cfg3": {"label": "CFG 3", "rank": 3, "cfg_scale": 3.0},
    "cfg5": {"label": "CFG 5", "rank": 4, "cfg_scale": 5.0},
    "cfg7p5": {"label": "CFG 7.5", "rank": 5, "cfg_scale": 7.5},
}

SAMPLING_RATIOS = (0.01, 0.02, 0.03, 0.04, 0.05)
REPEATS = (0, 1)
EXPECTED_ROWS_PER_SCENARIO = len(LAW_INFO) * len(LINE_INFO) * len(SAMPLING_RATIOS) * len(REPEATS)
EXPECTED_ROWS = EXPECTED_ROWS_PER_SCENARIO * len(SCENARIOS)


def case_name(law: str, line: str) -> str:
    """Return the stable case directory for one law and recovery-CFG line."""

    prefix = str(LAW_INFO[law]["prefix"])
    if line == "unconditioned":
        return f"{prefix}__recover_unprompted"
    return f"{prefix}__recover_prompt_sunset_beach_{line}"


def ensure_lpips(scenario: str, *, device: str = "cpu") -> pd.DataFrame:
    """Populate the study-local LPIPS table from saved or legacy artifacts."""

    return recovery.ensure_lpips_metrics(
        PROJECT_ROOT,
        result_namespace="weighted",
        artifact_roots=[RESULT_ROOT / scenario],
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
            case_root = RESULT_ROOT / scenario / law / name
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
    for law, line, ratio, repeat in product(LAW_INFO, LINE_INFO, SAMPLING_RATIOS, REPEATS):
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
    """Render the established TeX-styled PSNR, SSIM, and LPIPS sweep."""

    # With only two trials, standard-deviation bands are more transparent than
    # an unstable confidence interval estimated from one degree of freedom
    return cfgviz.plot_metric_curves(
        rows,
        output_dir=output_dir,
        show=show,
        band="std",
        xscale="linear",
    )


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
