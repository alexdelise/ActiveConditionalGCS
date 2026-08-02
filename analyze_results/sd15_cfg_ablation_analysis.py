"""CFG-scale ablation analysis helpers for the SD1.5 sunset recovery sweep."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


_LOCAL_TEX_ROOT = Path(__file__).resolve().parent / "tex"
_texinputs = os.environ.get("TEXINPUTS", "")
if str(_LOCAL_TEX_ROOT) not in _texinputs.split(os.pathsep):
    os.environ["TEXINPUTS"] = (
        f"{_LOCAL_TEX_ROOT}{os.pathsep}{_texinputs}"
        if _texinputs
        else f"{_LOCAL_TEX_ROOT}{os.pathsep}"
    )
try:
    from IPython import get_ipython
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    if get_ipython() is not None:
        set_matplotlib_formats("svg")
except (ImportError, RuntimeError):
    pass

PRESENTATION_RC: Dict[str, Any] = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "Latin Modern Roman",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "axes.formatter.use_mathtext": True,
    "figure.facecolor": "white",
    "figure.dpi": 200,
    "savefig.facecolor": "white",
    "savefig.dpi": 800,
    "axes.facecolor": "white",
    "axes.linewidth": 1.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 12,
    "axes.labelpad": 8,
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "font.size": 24,
    "legend.fontsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "grid.linewidth": 0.8,
    "lines.linewidth": 3.5,
    "lines.markersize": 9,
}
EXPORT_DPI = 800
DEFAULT_CONFIDENCE_LEVEL = 0.95
SAMPLING_X_LABEL = r"Sampling Ratio $m/n$"
ABLATION_SAMPLING_PERCENTAGES: tuple[float, ...] = (
    0.00015625,
    0.0003125,
    0.000625,
    0.00125,
    0.0025,
    0.005,
    0.01,
)
OLD_ABLATION_SAMPLING_PERCENTAGES: tuple[float, ...] = (0.00125, 0.0025, 0.005, 0.01, 0.025)
WEIGHTED_SAMPLING_PERCENTAGES: tuple[float, ...] = (0.00125, 0.0025, 0.005, 0.01, 0.025)
PANEL_TITLE_FONT = {
    "fontfamily": "serif",
    "fontname": "Computer Modern Roman",
}


BASE_DISTRIBUTIONS: List[Dict[str, Any]] = [
    {
        "key": "k0",
        "label": r"$\widetilde{\mu}_{\mathrm{uc}}$",
        "name": "Unconditioned sampling",
        "rank": 0,
        "prefix": "sample_k0_unconditioned",
    },
    {
        "key": "k1_daytime_beach",
        "label": r"$\widetilde{\mu}_{\mathrm{db}}$",
        "name": "Daytime-beach sampling",
        "rank": 1,
        "prefix": "sample_k1_daytime_beach",
    },
    {
        "key": "k2_sunset_beach",
        "label": r"$\widetilde{\mu}_{\mathrm{sb}}$",
        "name": "Sunset-beach sampling",
        "rank": 2,
        "prefix": "sample_k2_sunset_beach",
    },
    {
        "key": "k4_cat",
        "label": r"$\widetilde{\mu}_{\mathrm{ca}}$",
        "name": "Cat sampling",
        "rank": 4,
        "prefix": "sample_k4_cat",
    },
]

UNWEIGHTED_BASELINE_DISTRIBUTIONS: List[Dict[str, Any]] = [
    {
        "key": "mcs",
        "label": r"$\mu_{\mathrm{MCS}}$",
        "name": "Uniform MCS",
        "rank": 5,
        "prefix": "baseline_mcs",
        "sampling_method": "mcs",
    },
    {
        "key": "inverse_square",
        "label": r"$\mu_{\mathrm{IS}}$",
        "name": "Inverse-square sampling",
        "rank": 6,
        "prefix": "baseline_inverse_square",
        "sampling_method": "inverse_square",
    },
]

EXPERIMENT_SPECS: Dict[str, Dict[str, Any]] = {
    "prompt_matched_in_range": {
        "ablation_root": "unweighted/ablation/prompt_matched/sunset",
        "result_root": "unweighted/prompt_matched/sunset",
        "dataset_name": "sunset_beach_signal_sd15_512x512",
        "split": True,
        "archived": False,
    },
    "prompt_mismatched_in_range": {
        "ablation_root": "unweighted/ablation/prompt_mismatched/sunset",
        "result_root": "unweighted/prompt_mismatched/sunset",
        "dataset_name": "sunset_sandy_coast_signal_sd15_512x512",
        "split": True,
        "archived": False,
    },
    "out_of_range": {
        "ablation_root": "unweighted/ablation/out_of_range/sunset",
        "result_root": "unweighted/out_of_range/sunset",
        "dataset_name": "out_of_range_512x512",
        "split": True,
        "archived": False,
    },
    "weighted_prompt_matched_in_range": {
        "ablation_root": "weighted/ablation/prompt_matched/sunset",
        "result_root": "weighted/prompt_matched/sunset",
        "dataset_name": "sunset_beach_signal_sd15_512x512",
        "split": True,
        "split_names": ("first3", "last2"),
        "archived": False,
        "sampling_percentages": WEIGHTED_SAMPLING_PERCENTAGES,
    },
    "weighted_prompt_mismatched_in_range": {
        "ablation_root": "weighted/ablation/prompt_mismatched/sunset",
        "result_root": "weighted/prompt_mismatched/sunset",
        "dataset_name": "sunset_sandy_coast_signal_sd15_512x512",
        "split": True,
        "split_names": ("first3", "last2"),
        "archived": False,
        "sampling_percentages": WEIGHTED_SAMPLING_PERCENTAGES,
    },
    "weighted_out_of_range": {
        "ablation_root": "weighted/ablation/out_of_range/sunset",
        "result_root": "weighted/out_of_range/sunset",
        "dataset_name": "out_of_range_512x512",
        "split": True,
        "split_names": ("first3", "last2"),
        "archived": False,
        "sampling_percentages": WEIGHTED_SAMPLING_PERCENTAGES,
    },
}


def experiment_sampling_percentages(experiment: str) -> tuple[float, ...]:
    """Return the sampling grid used by one CFG-ablation experiment."""

    key = str(experiment)
    if key not in EXPERIMENT_SPECS:
        raise KeyError(f"Unknown CFG-ablation experiment {experiment!r}; expected one of {sorted(EXPERIMENT_SPECS)}.")
    spec = EXPERIMENT_SPECS[key]
    if "sampling_percentages" in spec:
        return tuple(float(value) for value in spec["sampling_percentages"])
    if bool(spec.get("archived", False)):
        return OLD_ABLATION_SAMPLING_PERCENTAGES
    return ABLATION_SAMPLING_PERCENTAGES


def experiment_distributions(experiment: str = "prompt_matched_in_range") -> List[Dict[str, Any]]:
    """Return the sampling distributions with experiment-specific result tags."""

    key = str(experiment)
    if key not in EXPERIMENT_SPECS:
        raise KeyError(f"Unknown CFG-ablation experiment {experiment!r}; expected one of {sorted(EXPERIMENT_SPECS)}.")
    spec = EXPERIMENT_SPECS[key]
    distributions: List[Dict[str, Any]] = []
    bases = [*BASE_DISTRIBUTIONS, *UNWEIGHTED_BASELINE_DISTRIBUTIONS]
    for base in bases:
        item = dict(base)
        prefix = str(item["prefix"])
        sampling_method = str(item.get("sampling_method", "cs"))
        item["experiment"] = key
        item["dataset_name"] = str(spec["dataset_name"])
        split_names = tuple(spec.get("split_names", ("first4", "last3")))
        item["result_root"] = str(spec["result_root"])
        item["split_names"] = split_names
        if sampling_method == "cs":
            item["new_tag"] = f"{spec['ablation_root']}/{prefix}"
            item["new_tags"] = (
                [f"{spec['ablation_root']}/{split_name}_{prefix}" for split_name in split_names]
                if bool(spec.get("split", False))
                else [item["new_tag"]]
            )
            item["old_tag"] = f"{spec['result_root']}/{prefix}"
            item["old_tags"] = (
                [f"{spec['result_root']}/{split_name}_{prefix}" for split_name in split_names]
                if bool(spec.get("split", False))
                else [item["old_tag"]]
            )
        else:
            item["new_tag"] = f"{spec['ablation_root']}/{sampling_method}_cfg_ablation"
            item["new_tags"] = [
                f"{spec['ablation_root']}/{split_name}_{sampling_method}_{line_key}"
                for split_name in split_names
                for line_key in ("cfg1", "cfg1p5", "cfg3", "cfg5")
            ]
            item["old_tag"] = ""
            item["old_tags"] = []
        distributions.append(item)
    return distributions


DISTRIBUTIONS: List[Dict[str, Any]] = experiment_distributions("prompt_matched_in_range")

LINE_SPECS: List[Dict[str, Any]] = [
    {"key": "unconditioned", "label": "Unconditioned", "rank": 0, "cfg_scale": np.nan},
    {"key": "cfg1", "label": "CFG 1", "rank": 1, "cfg_scale": 1.0},
    {"key": "cfg1p5", "label": "CFG 1.5", "rank": 2, "cfg_scale": 1.5},
    {"key": "cfg3", "label": "CFG 3", "rank": 3, "cfg_scale": 3.0},
    {"key": "cfg5", "label": "CFG 5", "rank": 4, "cfg_scale": 5.0},
    {"key": "cfg7p5", "label": "CFG 7.5", "rank": 5, "cfg_scale": 7.5},
]

METRIC_SPECS: List[tuple[str, str]] = [
    ("psnr_db", "PSNR (dB)"),
    ("ssim", "SSIM"),
    ("lpips", "LPIPS"),
    ("pixel_mae", "Per-Pixel MAE"),
]
PLOT_METRIC_SPECS: List[tuple[str, str]] = METRIC_SPECS[:3]

LINE_COLORS: Dict[str, str] = {
    "unconditioned": "#4C78A8",
    "cfg1": "#54A24B",
    "cfg1p5": "#B279A2",
    "cfg3": "#F58518",
    "cfg5": "#6A3D9A",
    "cfg7p5": "#E45756",
}
LINE_MARKERS: Dict[str, str] = {
    "unconditioned": "o",
    "cfg1": "D",
    "cfg1p5": "v",
    "cfg3": "s",
    "cfg5": "^",
    "cfg7p5": "P",
}
ZERO_FILLED_LABEL = "Zero-Filled"
ZERO_FILLED_COLOR = "#111111"
ZERO_FILLED_MARKER = "x"
SWEEP_LEGEND_Y = 1.08
ZERO_FILLED_METRIC_COLUMNS: Dict[str, str] = {
    "psnr_db": "zero_filled_psnr_db",
    "ssim": "zero_filled_ssim",
    "lpips": "zero_filled_lpips",
    "pixel_mae": "zero_filled_pixel_mae",
}
LPIPS_METRICS_RELATIVE_PATHS = {
    "weighted": Path("results/weighted/metrics/lpips.csv"),
    "unweighted": Path("results/unweighted/metrics/lpips.csv"),
}

RUN_DATA_KEYS = {
    "item_id",
    "prompt_text",
    "prompt_sha256",
    "conditioning_mode",
    "samp_perc",
    "samp_method",
    "m_coeffs",
    "repeat_id",
    "rep_seed",
    "runtime_sec",
    "psnr_db",
    "ssim",
    "pixel_mae",
    "grain",
    "zero_filled_psnr_db",
    "zero_filled_ssim",
    "zero_filled_pixel_mae",
    "zero_filled_grain",
    "recon_num_steps",
    "method",
    "recon_method",
}


def _candidate_roots(start: Path) -> Iterable[Path]:
    for base in [start, *start.parents]:
        yield base
        yield base / "sd1.5"


def find_sd15_root(start: str | Path | None = None) -> Path:
    """Resolve the SD1.5 project root from a notebook or script location."""

    begin = Path.cwd() if start is None else Path(start)
    for candidate in _candidate_roots(begin.resolve()):
        if (candidate / "ktilde" / "unweighted" / "config.json").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not resolve the sd1.5 project root.")


def _npz_value_to_python(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array.copy()


def _load_run_data_row(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        row = {
            key: _npz_value_to_python(payload[key])
            for key in payload.files
            if key in RUN_DATA_KEYS
        }
    row["_run_data_path"] = str(path)
    row["_result_source"] = "run_data_npz"
    return row


def _load_run_data_rows(case_root: Path) -> List[Dict[str, Any]]:
    return [
        _load_run_data_row(path)
        for path in sorted(case_root.glob("*/item_*/samp_*/rep_*/run_data.npz"))
    ]


def _load_compact_npz_rows(case_root: Path) -> List[Dict[str, Any]]:
    npz_path = case_root / "results_cs.npz"
    if not npz_path.is_file():
        return []
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files if key in RUN_DATA_KEYS}
        if not arrays:
            return []
        row_count = len(next(iter(arrays.values())))
        rows: List[Dict[str, Any]] = []
        for idx in range(row_count):
            row = {key: _npz_value_to_python(value[idx]) for key, value in arrays.items()}
            row["_result_source"] = "results_cs_npz"
            rows.append(row)
    return rows


def _line_spec(key: str) -> Dict[str, Any]:
    for spec in LINE_SPECS:
        if spec["key"] == key:
            return dict(spec)
    raise KeyError(f"Unknown line key: {key}")


def _line_specs_from_frame(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return configured recovery lines that have rows in the loaded experiment."""

    if frame.empty or "line_condition" not in frame.columns:
        return list(LINE_SPECS)
    available = {str(value) for value in frame["line_condition"].dropna().unique()}
    return [dict(spec) for spec in LINE_SPECS if str(spec["key"]) in available]


def _new_case_name(distribution: Mapping[str, Any], line_key: str) -> str:
    prefix = str(distribution["prefix"])
    if line_key == "cfg1":
        return f"{prefix}__recover_prompt_sunset_beach_cfg1"
    if line_key == "cfg1p5":
        return f"{prefix}__recover_prompt_sunset_beach_cfg1p5"
    if line_key == "cfg3":
        return f"{prefix}__recover_prompt_sunset_beach_cfg3"
    if line_key == "cfg5":
        return f"{prefix}__recover_prompt_sunset_beach_cfg5"
    raise KeyError(f"No new CFG case for line key {line_key!r}.")


def case_root_candidate_groups(
    sd15_root: str | Path,
    distribution: Mapping[str, Any],
    line_key: str,
) -> List[List[Path]]:
    """Return ordered groups of case roots, combining split roots before fallbacks."""

    root = find_sd15_root(sd15_root)
    prefix = str(distribution["prefix"])
    new_roots = [root / "results" / str(tag) for tag in distribution["new_tags"]]
    sampling_method = str(distribution.get("sampling_method", "cs"))
    if sampling_method != "cs":
        result_root = root / "results" / str(distribution["result_root"])
        split_names = tuple(distribution.get("split_names", ("first4", "last3")))
        if line_key == "unconditioned":
            return [
                [
                    result_root
                    / f"{split_name}_{sampling_method}_recover_unprompted"
                    / f"baseline_{sampling_method}__recover_unprompted"
                    for split_name in split_names
                ]
            ]
        if line_key == "cfg7p5":
            return [
                [
                    result_root
                    / f"{split_name}_{sampling_method}_recover_sunset_beach"
                    / f"baseline_{sampling_method}__recover_sunset_beach"
                    for split_name in split_names
                ]
            ]
        case_name = _new_case_name(distribution, line_key)
        return [
            [
                root
                / "results"
                / str(EXPERIMENT_SPECS[str(distribution["experiment"])]["ablation_root"])
                / f"{split_name}_{sampling_method}_{line_key}"
                / case_name
                for split_name in split_names
            ]
        ]

    old_roots = [root / "results" / str(tag) for tag in distribution["old_tags"]]
    unsplit_new_root = root / "results" / str(distribution["new_tag"])
    unsplit_old_root = root / "results" / str(distribution["old_tag"])

    if line_key == "unconditioned":
        return [
            [new_root / "reference_unconditioned" for new_root in new_roots],
            [unsplit_new_root / "reference_unconditioned"],
            [old_root / f"{prefix}__recover_unprompted" for old_root in old_roots],
            [unsplit_old_root / f"{prefix}__recover_unprompted"],
        ]
    if line_key == "cfg7p5":
        return [
            [new_root / "reference_cfg7p5" for new_root in new_roots],
            [unsplit_new_root / "reference_cfg7p5"],
            [old_root / f"{prefix}__recover_prompt_sunset_beach" for old_root in old_roots],
            [unsplit_old_root / f"{prefix}__recover_prompt_sunset_beach"],
        ]
    case_name = _new_case_name(distribution, line_key)
    return [
        [new_root / case_name for new_root in new_roots],
        [unsplit_new_root / case_name],
    ]


def _sampling_dir_name(value: float) -> str:
    return f"samp_{float(value):.5f}".replace(".", "p")


def _copy_tree_contents(src: Path, dst: Path, *, dry_run: bool = False) -> int:
    """Copy all files under src into dst, preserving existing destination extras."""

    file_count = sum(1 for path in src.rglob("*") if path.is_file())
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return file_count


def sync_main_references(
    sd15_root: str | Path | None = None,
    *,
    experiment: str = "prompt_matched_in_range",
    sampling_percentages: Sequence[float] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Refresh compatible unconditioned and CFG 7.5 references from main runs.

    Current legacy ablations use first4/last3 result tags matching the main
    experiments. Weighted ablations use first3/last2 tags. Archived ablations
    are never modified. The weighted launch scripts perform the stricter
    full-config compatibility check before copying; this helper supports
    notebook dry-runs.
    """

    root = find_sd15_root(sd15_root)
    key = str(experiment)
    if key not in EXPERIMENT_SPECS:
        raise KeyError(f"Unknown CFG-ablation experiment {experiment!r}; expected one of {sorted(EXPERIMENT_SPECS)}.")
    spec = EXPERIMENT_SPECS[key]
    if bool(spec.get("archived", False)):
        return pd.DataFrame(
            [
                {
                    "experiment": key,
                    "distribution_key": str(distribution["key"]),
                    "reference": "",
                    "source_kind": "archive",
                    "sampling_dir": "",
                    "status": "archived_no_sync",
                    "copied_files": 0,
                    "source": "",
                    "destination": str(root / "results" / str(distribution["new_tag"])),
                }
                for distribution in experiment_distributions(key)
            ]
        )
    if sampling_percentages is None:
        sample_values = [float(value) for value in experiment_sampling_percentages(key)]
    else:
        sample_values = [float(value) for value in sampling_percentages]
    records: List[Dict[str, Any]] = []
    references = [
        ("unconditioned", "recover_unprompted", "reference_unconditioned"),
        ("cfg7p5", "recover_prompt_sunset_beach", "reference_cfg7p5"),
    ]

    for distribution in experiment_distributions(key):
        prefix = str(distribution["prefix"])
        copied_any = False
        if str(distribution.get("sampling_method", "cs")) != "cs":
            records.append(
                {
                    "experiment": key,
                    "distribution_key": str(distribution["key"]),
                    "reference": "unconditioned,cfg7p5",
                    "source_kind": "direct_main_baseline",
                    "sampling_dir": "",
                    "status": "loaded_in_place",
                    "copied_files": 0,
                    "source": str(root / "results" / str(distribution["result_root"])),
                    "destination": str(root / "results" / str(spec["ablation_root"])),
                }
            )
            continue

        if bool(spec.get("split", False)):
            tag_roots = [
                (
                    split_name,
                    root / "results" / str(spec["result_root"]) / f"{split_name}_{prefix}",
                    root / "results" / str(spec["ablation_root"]) / f"{split_name}_{prefix}",
                )
                for split_name in tuple(spec.get("split_names", ("first4", "last3")))
            ]
        else:
            tag_roots = [
                (
                    "unsplit",
                    root / "results" / str(spec["result_root"]) / prefix,
                    root / "results" / str(spec["ablation_root"]) / prefix,
                )
            ]

        for source_kind, source_root, target_root in tag_roots:
            for reference, source_suffix, target_name in references:
                source_case = source_root / f"{prefix}__{source_suffix}"
                dest_case = target_root / target_name
                if not source_case.is_dir():
                    continue
                for sample_value in sample_values:
                    sampling_dir = _sampling_dir_name(sample_value)
                    src = source_case / "cs" / "item_000" / sampling_dir
                    if not src.is_dir():
                        continue
                    dst = dest_case / "cs" / "item_000" / sampling_dir
                    copied_files = _copy_tree_contents(src, dst, dry_run=dry_run)
                    copied_any = True
                    records.append(
                        {
                            "experiment": key,
                            "distribution_key": str(distribution["key"]),
                            "reference": reference,
                            "source_kind": source_kind,
                            "sampling_dir": sampling_dir,
                            "status": "would_copy" if dry_run else "copied",
                            "copied_files": int(copied_files),
                            "source": str(src),
                            "destination": str(dst),
                        }
                    )

        if not copied_any:
            records.append(
                {
                    "experiment": key,
                    "distribution_key": str(distribution["key"]),
                    "reference": "",
                    "source_kind": "",
                    "sampling_dir": "",
                    "status": "missing_source",
                    "copied_files": 0,
                    "source": str(root / "results"),
                    "destination": str(root / "results" / str(spec["ablation_root"])),
                }
            )

    return pd.DataFrame(records)


def sync_cfg7p5_references(
    sd15_root: str | Path | None = None,
    *,
    experiment: str = "prompt_matched_in_range",
    sampling_percentages: Sequence[float] | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Compatibility alias that now synchronizes both reusable main references."""

    return sync_main_references(
        sd15_root,
        experiment=experiment,
        sampling_percentages=sampling_percentages,
        dry_run=dry_run,
    )


def load_case_rows(
    sd15_root: str | Path,
    *,
    distribution: Mapping[str, Any],
    line_key: str,
) -> pd.DataFrame:
    """Load one distribution/CFG line from per-run artifacts or compact npz output."""

    line = _line_spec(line_key)
    for case_root_group in case_root_candidate_groups(sd15_root, distribution, line_key):
        group_frames: List[pd.DataFrame] = []
        for case_root in case_root_group:
            rows = _load_run_data_rows(case_root)
            if not rows:
                rows = _load_compact_npz_rows(case_root)
            if not rows:
                continue
            root_frame = pd.DataFrame(rows)
            root_frame["case_root"] = str(case_root)
            group_frames.append(root_frame)
        if not group_frames:
            continue

        frame = pd.concat(group_frames, ignore_index=True, sort=False)
        frame["distribution_key"] = str(distribution["key"])
        frame["distribution_label"] = str(distribution["label"])
        frame["distribution_name"] = str(distribution["name"])
        frame["distribution_rank"] = int(distribution["rank"])
        frame["experiment"] = str(distribution.get("experiment", "prompt_matched_in_range"))
        frame["dataset_name"] = str(distribution.get("dataset_name", EXPERIMENT_SPECS["prompt_matched_in_range"]["dataset_name"]))
        frame["line_condition"] = str(line["key"])
        frame["line_label"] = str(line["label"])
        frame["line_rank"] = int(line["rank"])
        frame["cfg_scale"] = float(line["cfg_scale"]) if np.isfinite(line["cfg_scale"]) else np.nan
        return frame

    return pd.DataFrame()


def _filter_sampling_percentages(frame: pd.DataFrame, sampling_percentages: Sequence[float] | None) -> pd.DataFrame:
    """Keep only requested sampling percentages, using tolerant float matching."""

    if frame.empty or sampling_percentages is None or "samp_perc" not in frame.columns:
        return frame
    allowed = [float(value) for value in sampling_percentages]
    values = frame["samp_perc"].astype(float).to_numpy()
    mask = np.zeros(len(frame), dtype=bool)
    for value in allowed:
        mask |= np.isclose(values, value)
    return frame.loc[mask].copy()


def attach_lpips_metrics(
    frame: pd.DataFrame,
    sd15_root: str | Path,
    *,
    result_namespace: str,
) -> pd.DataFrame:
    """Join the same incremental LPIPS sidecar used by the main notebooks."""

    result = frame.copy()
    for column in ("lpips", "zero_filled_lpips"):
        if column not in result.columns:
            result[column] = np.nan
    namespace = str(result_namespace).strip().lower()
    if namespace not in LPIPS_METRICS_RELATIVE_PATHS:
        raise ValueError(f"Unsupported LPIPS namespace: {result_namespace!r}.")
    sidecar_path = Path(sd15_root) / LPIPS_METRICS_RELATIVE_PATHS[namespace]
    if result.empty or not sidecar_path.is_file() or "_run_data_path" not in result.columns:
        return result

    sidecar = pd.read_csv(sidecar_path)
    required = {"artifact_relpath", "lpips"}
    if not required.issubset(sidecar.columns):
        raise ValueError(
            f"LPIPS sidecar {sidecar_path} is missing "
            f"{sorted(required.difference(sidecar.columns))}."
        )
    if sidecar["artifact_relpath"].duplicated().any():
        raise ValueError(f"LPIPS sidecar {sidecar_path} contains duplicate artifact paths.")

    root = Path(sd15_root).resolve()

    def artifact_relpath(value: Any) -> str:
        if not isinstance(value, str) or not value:
            return ""
        return str(Path(value).resolve().parent.relative_to(root))

    result["artifact_relpath"] = result["_run_data_path"].map(artifact_relpath)
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


def load_cfg_ablation_rows(
    sd15_root: str | Path | None = None,
    *,
    experiment: str = "prompt_matched_in_range",
    sampling_percentages: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Load all available CFG-ablation rows, including copied or original references."""

    root = find_sd15_root(sd15_root)
    frames: List[pd.DataFrame] = []
    for distribution in experiment_distributions(experiment):
        for line in LINE_SPECS:
            frame = load_case_rows(root, distribution=distribution, line_key=str(line["key"]))
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True, sort=False)
    numeric_columns = [
        "item_id",
        "repeat_id",
        "samp_perc",
        "psnr_db",
        "ssim",
        "pixel_mae",
        "grain",
        "zero_filled_psnr_db",
        "zero_filled_ssim",
        "zero_filled_pixel_mae",
        "zero_filled_grain",
        "runtime_sec",
        "cfg_scale",
        "distribution_rank",
        "line_rank",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    selected_sampling_percentages = sampling_percentages
    if selected_sampling_percentages is None:
        selected_sampling_percentages = experiment_sampling_percentages(experiment)
    frame = _filter_sampling_percentages(frame, selected_sampling_percentages)
    if frame.empty:
        return pd.DataFrame()
    dedupe_columns = [
        "distribution_key",
        "line_condition",
        "item_id",
        "samp_perc",
        "repeat_id",
    ]
    frame = frame.drop_duplicates(subset=dedupe_columns, keep="last").sort_values(
        ["distribution_rank", "line_rank", "samp_perc", "repeat_id"],
        kind="stable",
    ).reset_index(drop=True)
    namespace = "weighted" if str(experiment).startswith("weighted_") else "unweighted"
    return attach_lpips_metrics(frame, root, result_namespace=namespace)


def _z_value_for_confidence_level(confidence_level: float) -> float:
    """Return the two-sided normal critical value for a confidence level."""

    level = float(confidence_level)
    if level <= 0.0:
        return 0.0
    level = min(level, 0.999999)
    return float(NormalDist().inv_cdf(0.5 + 0.5 * level))


def _band_suffix(band: str) -> str:
    """Return the metric-column suffix for an uncertainty band mode."""

    band_key = str(band).strip().lower()
    if band_key == "std":
        return "std"
    if band_key == "sem":
        return "sem"
    if band_key == "ci":
        return "ci_halfwidth"
    raise KeyError(f"Unsupported band mode '{band}'. Expected one of: ci, sem, std.")


def build_metric_summary(
    frame: pd.DataFrame,
    metrics: Sequence[str] = tuple(metric for metric, _ in METRIC_SPECS),
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Aggregate means and uncertainty columns over repeats."""

    if frame.empty:
        return pd.DataFrame()
    group_cols = [
        "distribution_key",
        "distribution_label",
        "distribution_name",
        "distribution_rank",
        "line_condition",
        "line_label",
        "line_rank",
        "cfg_scale",
        "samp_perc",
    ]
    grouped = frame.groupby(group_cols, dropna=False, sort=False)[list(metrics)].agg(["mean", "std", "count"]).reset_index()
    columns: List[str] = []
    for column in grouped.columns:
        if isinstance(column, tuple):
            name, stat = column
            columns.append(str(name) if not stat or stat == "mean" else f"{name}_{stat}")
        else:
            columns.append(str(column))
    grouped.columns = columns
    z_value = _z_value_for_confidence_level(confidence_level)
    for metric, _ in METRIC_SPECS:
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        sem_col = f"{metric}_sem"
        ci_col = f"{metric}_ci_halfwidth"
        if std_col in grouped.columns and count_col in grouped.columns:
            grouped[std_col] = grouped[std_col].fillna(0.0)
            grouped[count_col] = grouped[count_col].fillna(0).astype(int)
            grouped[sem_col] = 0.0
            valid = grouped[count_col] > 0
            grouped.loc[valid, sem_col] = grouped.loc[valid, std_col] / np.sqrt(grouped.loc[valid, count_col].astype(float))
            grouped[ci_col] = grouped[sem_col] * z_value
    return grouped.sort_values(["distribution_rank", "line_rank", "samp_perc"], kind="stable").reset_index(drop=True)


def build_zero_filled_metric_summary(
    frame: pd.DataFrame,
    metric: str,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Aggregate zero-filled baseline metrics over unique masks/repeats."""

    source_metric = ZERO_FILLED_METRIC_COLUMNS.get(str(metric))
    empty_columns = [
        "distribution_key",
        "distribution_label",
        "distribution_name",
        "distribution_rank",
        "samp_perc",
        metric,
        f"{metric}_std",
        f"{metric}_count",
        f"{metric}_sem",
        f"{metric}_ci_halfwidth",
    ]
    if frame.empty or source_metric is None or source_metric not in frame.columns:
        return pd.DataFrame(columns=empty_columns)

    keep_columns = [
        column
        for column in [
            "distribution_key",
            "distribution_label",
            "distribution_name",
            "distribution_rank",
            "item_id",
            "repeat_id",
            "samp_perc",
            source_metric,
        ]
        if column in frame.columns
    ]
    baseline = frame[keep_columns].copy()
    baseline[source_metric] = pd.to_numeric(baseline[source_metric], errors="coerce")
    baseline["samp_perc"] = pd.to_numeric(baseline["samp_perc"], errors="coerce")
    baseline = baseline.dropna(subset=["distribution_key", "repeat_id", "samp_perc", source_metric])
    if baseline.empty:
        return pd.DataFrame(columns=empty_columns)
    baseline = baseline.drop_duplicates(
        subset=["distribution_key", "item_id", "repeat_id", "samp_perc"],
        keep="last",
    )
    group_cols = [
        "distribution_key",
        "distribution_label",
        "distribution_name",
        "distribution_rank",
        "samp_perc",
    ]
    grouped = (
        baseline.groupby(group_cols, dropna=False, sort=False)[source_metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .rename(columns={"mean": metric, "std": f"{metric}_std", "count": f"{metric}_count"})
    )
    std_col = f"{metric}_std"
    count_col = f"{metric}_count"
    sem_col = f"{metric}_sem"
    ci_col = f"{metric}_ci_halfwidth"
    grouped[std_col] = grouped[std_col].fillna(0.0)
    grouped[count_col] = grouped[count_col].fillna(0).astype(int)
    grouped[sem_col] = 0.0
    valid = grouped[count_col] > 0
    grouped.loc[valid, sem_col] = grouped.loc[valid, std_col] / np.sqrt(grouped.loc[valid, count_col].astype(float))
    grouped[ci_col] = grouped[sem_col] * _z_value_for_confidence_level(confidence_level)
    return grouped.sort_values(["distribution_rank", "samp_perc"], kind="stable").reset_index(drop=True)


def count_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return available repeat counts by distribution, CFG line, and sampling percentage."""

    if frame.empty:
        return pd.DataFrame()
    table = frame.pivot_table(
        index=["distribution_name", "line_label"],
        columns="samp_perc",
        values="repeat_id",
        aggfunc="nunique",
        fill_value=0,
    )
    return table.sort_index()


def _distributions_from_frame(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Recover ordered distribution metadata from loaded rows."""

    if frame.empty:
        return []
    columns = ["distribution_key", "distribution_label", "distribution_name", "distribution_rank"]
    available = [column for column in columns if column in frame.columns]
    if len(available) != len(columns):
        return sorted(DISTRIBUTIONS, key=lambda item: int(item["rank"]))
    records = (
        frame[columns]
        .drop_duplicates()
        .sort_values("distribution_rank", kind="stable")
        .to_dict(orient="records")
    )
    return [
        {
            "key": str(record["distribution_key"]),
            "label": str(record["distribution_label"]),
            "name": str(record["distribution_name"]),
            "rank": int(record["distribution_rank"]),
        }
        for record in records
    ]


def _sampling_tick_labels(values: Sequence[float]) -> List[str]:
    labels: List[str] = []
    for value in values:
        value = float(value)
        if value < 0.01:
            labels.append(f"{value:.5f}")
        elif value < 0.1:
            labels.append(f"{value:.3f}")
        else:
            labels.append(f"{value:.2f}".rstrip("0").rstrip("."))
    return labels


def _apply_sampling_axis(ax: Any, values: Sequence[float]) -> None:
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    ticks = sorted({float(value) for value in values if np.isfinite(value)})
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(_sampling_tick_labels(ticks)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.tick_params(axis="x", which="major", labelrotation=35)
    for label in ax.get_xticklabels(which="major"):
        label.set_ha("right")
    ax.set_xlabel("")
    ax.grid(True, which="major", axis="both", alpha=0.28, linestyle="--")


def _synchronize_christoffel_y_limits(
    axes: Sequence[Any],
    distributions: Sequence[Dict[str, Any]],
) -> None:
    """Give the four Christoffel panels one scale and each baseline its own."""

    christoffel_axes = [
        ax
        for ax, distribution in zip(axes, distributions)
        if str(distribution.get("sampling_method", "cs")) == "cs" and ax.get_visible()
    ]
    if len(christoffel_axes) < 2:
        return
    limits = [ax.get_ylim() for ax in christoffel_axes]
    lower = min(float(limit[0]) for limit in limits)
    upper = max(float(limit[1]) for limit in limits)
    for ax in christoffel_axes:
        ax.set_ylim(lower, upper)


def _save_figure(fig: Any, output_dir: str | Path | None, stem: str, *, show: bool) -> Dict[str, Path]:
    import matplotlib.pyplot as plt

    outputs: Dict[str, Path] = {}
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{stem}.pdf"
        fig.savefig(path, dpi=EXPORT_DPI, bbox_inches="tight")
        outputs["pdf"] = path
    if show:
        plt.show()
    plt.close(fig)
    return outputs


def _figure_slug(value: object) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower())
    return slug.strip("_")


def _ablation_context_slug(output_dir: str | Path | None) -> str:
    if output_dir is None:
        return "cfg_ablation"
    parts = list(Path(output_dir).parts)
    if "ablation" in parts:
        parts = parts[parts.index("ablation") + 1 :]
    else:
        parts = parts[-2:]
    slug_parts = [_figure_slug(part) for part in parts]
    return "_".join(part for part in slug_parts if part) or "cfg_ablation"


def plot_metric_curves(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    show: bool = True,
    band: str = "ci",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Dict[str, Path]:
    """Plot combined PSNR, SSIM, and LPIPS curves for all distributions."""

    import matplotlib.pyplot as plt

    if frame.empty:
        raise ValueError("No CFG-ablation rows were found.")
    summary = build_metric_summary(frame, confidence_level=confidence_level)
    line_specs = _line_specs_from_frame(frame)
    zero_summaries = {
        metric: build_zero_filled_metric_summary(frame, metric, confidence_level=confidence_level)
        for metric, _ in PLOT_METRIC_SPECS
    }
    distributions = _distributions_from_frame(frame)
    with plt.rc_context(PRESENTATION_RC):
        fig, axes = plt.subplots(
            len(PLOT_METRIC_SPECS),
            len(distributions),
            figsize=(4.9 * len(distributions), 3.7 * len(PLOT_METRIC_SPECS)),
            sharex="col",
            sharey=False,
            squeeze=False,
            constrained_layout=True,
        )
        axes_array = np.asarray(axes, dtype=object)
        band_suffix = _band_suffix(band)

        for col_idx, distribution in enumerate(distributions):
            dist_key = str(distribution["key"])
            dist_subset = summary[summary["distribution_key"] == dist_key]
            for row_idx, (metric, metric_label) in enumerate(PLOT_METRIC_SPECS):
                ax = axes_array[row_idx, col_idx]
                for line in line_specs:
                    line_key = str(line["key"])
                    group = dist_subset[dist_subset["line_condition"] == line_key].sort_values("samp_perc", kind="stable")
                    if group.empty:
                        continue
                    x = group["samp_perc"].to_numpy(dtype=float)
                    y = group[metric].to_numpy(dtype=float)
                    ax.plot(
                        x,
                        y,
                        label=str(line["label"]),
                        color=LINE_COLORS[line_key],
                        marker=LINE_MARKERS[line_key],
                        linewidth=2.4,
                        markersize=7.0,
                        markerfacecolor="white",
                        markeredgewidth=1.3,
                    )
                    band_col = f"{metric}_{band_suffix}"
                    if band_col in group.columns:
                        delta = group[band_col].fillna(0.0).to_numpy(dtype=float)
                        if np.any(delta > 0.0):
                            ax.fill_between(x, y - delta, y + delta, color=LINE_COLORS[line_key], alpha=0.14, linewidth=0)
                zero_summary = zero_summaries.get(metric, pd.DataFrame())
                zero_group = zero_summary[zero_summary["distribution_key"] == dist_key].sort_values("samp_perc", kind="stable")
                if not zero_group.empty:
                    x_zero = zero_group["samp_perc"].to_numpy(dtype=float)
                    y_zero = zero_group[metric].to_numpy(dtype=float)
                    ax.plot(
                        x_zero,
                        y_zero,
                        label=ZERO_FILLED_LABEL,
                        color=ZERO_FILLED_COLOR,
                        linestyle="--",
                        marker=ZERO_FILLED_MARKER,
                        linewidth=2.4,
                        markersize=7.0,
                        markeredgewidth=1.3,
                    )
                    band_col = f"{metric}_{band_suffix}"
                    if band_col in zero_group.columns:
                        delta = zero_group[band_col].fillna(0.0).to_numpy(dtype=float)
                        if np.any(delta > 0.0):
                            ax.fill_between(
                                x_zero,
                                y_zero - delta,
                                y_zero + delta,
                                color=ZERO_FILLED_COLOR,
                                alpha=0.10,
                                linewidth=0,
                            )
                if row_idx == 0:
                    ax.set_title(str(distribution["label"]))
                if col_idx == 0:
                    ax.set_ylabel(metric_label)
                if row_idx == len(PLOT_METRIC_SPECS) - 1:
                    _apply_sampling_axis(ax, dist_subset["samp_perc"].tolist())
                else:
                    ax.grid(True, which="major", axis="both", alpha=0.28, linestyle="--")

        for row_idx, _ in enumerate(PLOT_METRIC_SPECS):
            _synchronize_christoffel_y_limits(
                list(axes_array[row_idx, :]),
                distributions,
            )

        fig.supxlabel(SAMPLING_X_LABEL, fontsize=PRESENTATION_RC["axes.labelsize"])
        handles: List[Any] = []
        labels: List[str] = []
        for ax in axes_array.ravel():
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            for handle, label in zip(ax_handles, ax_labels):
                if label and label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(handles),
                frameon=False,
                bbox_to_anchor=(0.5, SWEEP_LEGEND_Y),
                columnspacing=0.9,
                handletextpad=0.35,
                borderaxespad=0.2,
            )
        context = _ablation_context_slug(output_dir)
        stem = f"cfg_ablation_{context}_psnr_ssim_lpips_vs_sampling_ratio"
        return _save_figure(fig, output_dir, stem, show=show)


def _read_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0


def _panel_title_text(text: str) -> str:
    """Render panel titles in Computer Modern via math roman text."""

    escaped = str(text).replace("\\", r"\backslash ").replace(" ", r"\ ")
    escaped = escaped.replace("-", r"\!-\!")
    return rf"$\mathrm{{{escaped}}}$"


def _format_sampling_tag(value: float) -> str:
    return f"samp_{float(value):.5f}".replace(".", "p")


def _default_panel_sampling_percentage(frame: pd.DataFrame) -> float:
    """Choose the largest sampling percentage shared by available panel columns."""

    available_sets: List[set[float]] = []
    for distribution in _distributions_from_frame(frame):
        for line in _line_specs_from_frame(frame):
            subset = frame[
                (frame["distribution_key"] == str(distribution["key"]))
                & (frame["line_condition"] == str(line["key"]))
            ]
            if subset.empty:
                continue
            available_sets.append({float(value) for value in subset["samp_perc"].dropna().unique()})

    if available_sets:
        common_values = set.intersection(*available_sets)
        if common_values:
            return float(max(common_values))
    return float(np.nanmax(frame["samp_perc"].to_numpy(dtype=float)))


def _best_panel_row(frame: pd.DataFrame, *, distribution_key: str, line_key: str, samp_perc: float) -> Optional[pd.Series]:
    subset = frame[
        (frame["distribution_key"] == distribution_key)
        & (frame["line_condition"] == line_key)
        & np.isclose(frame["samp_perc"].astype(float), float(samp_perc))
    ].copy()
    if subset.empty:
        return None
    sort_cols = [column for column in ["psnr_db", "ssim"] if column in subset.columns]
    if sort_cols:
        subset = subset.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="stable")
    return subset.iloc[0]


def _best_zero_filled_panel_row(frame: pd.DataFrame, *, distribution_key: str, samp_perc: float) -> Optional[pd.Series]:
    subset = frame[
        (frame["distribution_key"] == distribution_key)
        & np.isclose(frame["samp_perc"].astype(float), float(samp_perc))
    ].copy()
    if subset.empty:
        return None
    subset = subset.drop_duplicates(
        subset=["distribution_key", "item_id", "repeat_id", "samp_perc"],
        keep="last",
    )
    sort_cols = [column for column in ["zero_filled_psnr_db", "zero_filled_ssim"] if column in subset.columns]
    if sort_cols:
        subset = subset.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="stable")
    return subset.iloc[0]


def _add_metric_overlay(ax: Any, row: pd.Series) -> None:
    lpips_line = ""
    if "lpips" in row and pd.notna(row["lpips"]):
        lpips_line = f"\nLPIPS {float(row['lpips']):.3f}"
    ax.text(
        0.02,
        0.96,
        f"PSNR {float(row['psnr_db']):.2f} dB\n"
        f"SSIM {float(row['ssim']):.3f}"
        f"{lpips_line}\n"
        f"PPMAE {float(row['pixel_mae']):.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="white",
        bbox={"boxstyle": "square,pad=0.22", "facecolor": (0, 0, 0, 0.46), "edgecolor": "none"},
    )


def _add_zero_filled_metric_overlay(ax: Any, row: pd.Series) -> None:
    lpips_line = ""
    if "zero_filled_lpips" in row and pd.notna(row["zero_filled_lpips"]):
        lpips_line = f"\nLPIPS {float(row['zero_filled_lpips']):.3f}"
    ax.text(
        0.02,
        0.96,
        f"PSNR {float(row['zero_filled_psnr_db']):.2f} dB\n"
        f"SSIM {float(row['zero_filled_ssim']):.3f}"
        f"{lpips_line}\n"
        f"PPMAE {float(row['zero_filled_pixel_mae']):.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="white",
        bbox={"boxstyle": "square,pad=0.22", "facecolor": (0, 0, 0, 0.58), "edgecolor": "none"},
    )


def plot_reconstruction_panel(
    frame: pd.DataFrame,
    *,
    sd15_root: str | Path | None = None,
    samp_perc: float | None = None,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Dict[str, Path]:
    """Plot best reconstruction panels for the selected sampling percentage."""

    import matplotlib.pyplot as plt

    if frame.empty:
        raise ValueError("No CFG-ablation rows were found.")
    root = find_sd15_root(sd15_root)
    if samp_perc is None:
        samp_perc = _default_panel_sampling_percentage(frame)
    dataset_name = EXPERIMENT_SPECS["prompt_matched_in_range"]["dataset_name"]
    if "dataset_name" in frame.columns and not frame["dataset_name"].dropna().empty:
        dataset_name = str(frame["dataset_name"].dropna().iloc[0])
    gt_path = root / "datasets" / dataset_name / "gt_000.png"
    gt_image = _read_image(gt_path) if gt_path.is_file() else None

    distributions = _distributions_from_frame(frame)
    n_rows = len(distributions)
    line_specs = _line_specs_from_frame(frame)
    n_cols = 2 + len(line_specs)
    with plt.rc_context(PRESENTATION_RC):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.0 * n_cols, 3.0 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(
            rf"$\mathrm{{Sampling\ Ratio:}}\ {float(samp_perc):.5f}$",
            fontsize=34,
            **PANEL_TITLE_FONT,
        )

        for row_idx, distribution in enumerate(distributions):
            ax = axes[row_idx, 0]
            if gt_image is not None:
                ax.imshow(gt_image)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(_panel_title_text("Ground Truth"), fontsize=22, pad=6, **PANEL_TITLE_FONT)
            ax.set_ylabel(str(distribution["label"]), rotation=0, labelpad=42, va="center", fontsize=22)

            ax = axes[row_idx, 1]
            selected_zero = _best_zero_filled_panel_row(
                frame,
                distribution_key=str(distribution["key"]),
                samp_perc=float(samp_perc),
            )
            if selected_zero is not None and "_run_data_path" in selected_zero and isinstance(selected_zero["_run_data_path"], str):
                zero_path = Path(selected_zero["_run_data_path"]).parent / "zero_filled_ifft.png"
                if zero_path.is_file():
                    ax.imshow(_read_image(zero_path), cmap="gray")
                    _add_zero_filled_metric_overlay(ax, selected_zero)
                else:
                    ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(_panel_title_text(ZERO_FILLED_LABEL), fontsize=22, pad=6, color=ZERO_FILLED_COLOR, **PANEL_TITLE_FONT)

            for col_offset, line in enumerate(line_specs, start=2):
                line_key = str(line["key"])
                ax = axes[row_idx, col_offset]
                selected = _best_panel_row(
                    frame,
                    distribution_key=str(distribution["key"]),
                    line_key=line_key,
                    samp_perc=float(samp_perc),
                )
                if selected is not None and "_run_data_path" in selected and isinstance(selected["_run_data_path"], str):
                    method = str(selected.get("method", "cs"))
                    recon_path = Path(selected["_run_data_path"]).parent / f"recon_{method}.png"
                    if recon_path.is_file():
                        ax.imshow(_read_image(recon_path))
                        _add_metric_overlay(ax, selected)
                    else:
                        ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
                else:
                    ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_idx == 0:
                    ax.set_title(_panel_title_text(str(line["label"])), fontsize=22, pad=6, **PANEL_TITLE_FONT)

    context = _ablation_context_slug(output_dir)
    stem = f"cfg_ablation_{context}_reconstruction_panel_{_format_sampling_tag(float(samp_perc))}"
    return _save_figure(fig, output_dir, stem, show=show)
