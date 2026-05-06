"""SD1.5 beach-conditioning analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


SD15_PRESENTATION_RC: Dict[str, Any] = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
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
SD15_PLOT_EXCLUDED_ROLES: set[str] = set()
SD15_EXPORT_DPI = 800

SD15_X_LABEL = r"Sampling Ratio $m/n$"
SD15_METRIC_LABELS = {
    "psnr_db": r"$\mathrm{PSNR\ (dB)}$",
    "ssim": r"$\mathrm{SSIM}$",
    "pixel_mae": r"$\mathrm{Per\!-\!Pixel\ MAE}$",
    "grain": r"$\mathrm{Grain}$",
    "runtime_sec": r"$\mathrm{Runtime\ (s)}$",
}
SD15_RECON_COLORS = ["#4C78A8", "#54A24B", "#E45756", "#6A3D9A", "#F58518"]
SD15_RECON_MARKERS = ["o", "D", "s", "^", "P"]
SD15_PRIOR_COLORS = {
    "k0": "#4C78A8",
    "k1_daytime_beach": "#54A24B",
    "k2_sunset_beach": "#E45756",
    "k4_cat": "#6A3D9A",
}
SD15_LAMBDA_CMAP_COLORS = ["#16324F", "#3A7CA5", "#8DC7B4", "#DDEBD6", "#FCFBF7"]
SD15_PENALTY_CMAP_COLORS = ["#FFF7ED", "#F8D19A", "#EE8F63", "#C8556D", "#5B1746"]
SD15_DISTRIBUTION_CMAP_COLORS = ["#16324F", "#3A7CA5", "#8DC7B4", "#DDEBD6", "#FCFBF7"]
SD15_LAMBDA_STYLE_RECON_COLORS = ["#16324F", "#3A7CA5", "#6DB7AD", "#9CCB8B", "#CDBE7A"]
DC_METHOD_LABELS = {
    "diffusion_backprop": "Diffusion Backprop",
}
SAMPLING_METHOD_LABELS = {
    "cs": "CS (Christoffel)",
    "mcs": "MCS",
}
REGRESSION_ROW_COLUMNS = [
    "dc_method",
    "dc_method_label",
    "sampling_method",
    "sampling_method_label",
    "suite_case",
    "suite_case_description",
    "sampling_condition",
    "sampling_label",
    "sampling_rank",
    "reconstruction_condition",
    "reconstruction_label",
    "recon_rank",
    "ktilde_name",
    "run_tag",
    "item_id",
    "repeat_id",
    "samp_perc",
    "psnr_db",
    "ssim",
    "pixel_mae",
    "grain",
    "runtime_sec",
    "_result_source",
]
MEAN_METRIC_COLUMNS = [
    "dc_method",
    "dc_method_label",
    "sampling_method",
    "sampling_method_label",
    "sampling_condition",
    "sampling_label",
    "sampling_rank",
    "reconstruction_condition",
    "reconstruction_label",
    "recon_rank",
    "samp_perc",
    "psnr_db",
    "ssim",
    "pixel_mae",
    "grain",
    "runtime_sec",
]
SUMMARY_METRICS = ["psnr_db", "ssim", "pixel_mae", "grain", "runtime_sec"]
DEFAULT_CONFIDENCE_LEVEL = 0.95
UNPROMPTED_DISPLAY_LABEL = "Unconditioned"


def _candidate_roots(start: Path) -> Iterable[Path]:
    for base in [start, *start.parents]:
        yield base
        yield base / "sd1.5"


def find_sd15_root(start: str | Path | None = None) -> Path:
    """Resolve the SD1.5 project root from a cwd or notebook location."""

    begin = Path.cwd() if start is None else Path(start)
    for candidate in _candidate_roots(begin.resolve()):
        if (candidate / "ktilde" / "config.json").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not resolve the sd1.5 project root from the current working directory.")


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a UTF-8 JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _npz_value_to_python(value: np.ndarray) -> Any:
    """Convert one npz field into a scalar when possible, preserving trace arrays."""

    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.copy()


def load_run_data_npz(path: str | Path) -> Dict[str, Any]:
    """Load one completed per-run ``run_data.npz`` artifact."""

    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as payload:
        return {str(key): _npz_value_to_python(payload[key]) for key in payload.files}


def discover_regression_tags(sd15_root: str | Path) -> List[str]:
    """Return top-level result tags that contain conditioning-suite manifests."""

    root = find_sd15_root(sd15_root)
    results_root = root / "results"
    if not results_root.is_dir():
        return []

    tags: set[str] = set()
    for manifest_path in results_root.rglob("suite_manifest.json"):
        relative_parent = manifest_path.parent.relative_to(results_root)
        pieces = relative_parent.parts
        if len(pieces) >= 2:
            tags.add(pieces[0])
        elif pieces:
            tags.add(pieces[0])
    return sorted(tags)


def _partial_sampling_methods(case_root: Path) -> List[str]:
    """Return sampler folders that already contain at least one per-run artifact."""

    methods: List[str] = []
    for method_root in sorted(path for path in case_root.iterdir() if path.is_dir()):
        if any(method_root.glob("item_*/samp_*/rep_*/run_data.npz")):
            methods.append(method_root.name)
    return methods


def _load_partial_run_frame(case_root: Path, sampling_method: str) -> pd.DataFrame:
    """Load available per-run artifacts for one sampler under one case root."""

    method_root = case_root / str(sampling_method)
    rows: List[Dict[str, Any]] = []
    for npz_path in sorted(method_root.glob("item_*/samp_*/rep_*/run_data.npz")):
        row = load_run_data_npz(npz_path)
        row.setdefault("method", str(sampling_method))
        row["_result_source"] = "run_data_npz"
        row["_run_data_path"] = str(npz_path)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _drop_duplicate_run_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per reconstruction, preferring per-run artifacts over stale CSV rows."""

    if frame.empty:
        return frame
    keys = [
        "run_tag",
        "sampling_method",
        "item_id",
        "prompt_sha256",
        "samp_perc",
        "repeat_id",
    ]
    subset = [key for key in keys if key in frame.columns]
    if not subset:
        return frame.reset_index(drop=True)
    return frame.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)


def _attach_regression_metadata(
    frame: pd.DataFrame,
    *,
    dc_method: str,
    sampling_method: str,
    case: Mapping[str, Any],
    case_tag: str,
) -> pd.DataFrame:
    """Attach suite metadata expected by the plotting helpers."""

    frame = frame.copy()
    frame["dc_method"] = dc_method
    frame["dc_method_label"] = DC_METHOD_LABELS.get(dc_method, dc_method)
    frame["sampling_method"] = sampling_method
    frame["sampling_method_label"] = SAMPLING_METHOD_LABELS.get(sampling_method, sampling_method)
    frame["suite_case"] = str(case.get("name", ""))
    frame["suite_case_description"] = str(case.get("description", ""))
    frame["sampling_condition"] = str(case.get("sampling_condition", case.get("ktilde_name", "")))
    frame["sampling_label"] = str(case.get("sampling_label", case.get("sampling_condition", "")))
    frame["sampling_rank"] = int(case.get("sampling_rank", 0))
    frame["reconstruction_condition"] = str(case.get("reconstruction_condition", ""))
    frame["reconstruction_label"] = _display_reconstruction_label(
        frame["reconstruction_condition"].iloc[0],
        case.get("reconstruction_label", case.get("reconstruction_condition", "")),
    )
    frame["recon_rank"] = int(case.get("recon_rank", 0))
    frame["ktilde_name"] = str(case.get("ktilde_name", ""))
    frame["run_tag"] = case_tag
    return frame


def load_ktilde_catalog(sd15_root: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load the raw k-tilde config JSON so analysis can access labels and roles."""

    root = find_sd15_root(sd15_root)
    payload = load_json(root / "ktilde" / "config.json")
    return {str(name): dict(entry) for name, entry in dict(payload.get("ktilde", {})).items()}


def load_ktilde_npz_for_analysis(path: str | Path) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a k-tilde archive without importing the torch-dependent runtime module."""

    with np.load(str(path), allow_pickle=False) as payload:
        k_tilde = payload["K_tilde"].astype(np.float64)
        probabilities = payload["prob"].astype(np.float64)
        metadata = json.loads(str(payload["meta"]))
    return k_tilde, probabilities, metadata


def ktilde_unitary_energy_scale(metadata: Mapping[str, Any], probabilities: np.ndarray) -> float:
    """Return the H*W factor converting stored FFT energies to the unitary convention."""

    height = int(metadata.get("height", 0) or 0)
    width = int(metadata.get("width", 0) or 0)
    if height > 0 and width > 0:
        return float(height * width)
    return float(np.asarray(probabilities, dtype=np.float64).size)


def load_ktilde_bank(
    sd15_root: str | Path,
    *,
    names: Optional[Sequence[str]] = None,
    skip_missing: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Load configured k-tilde artifacts plus their analysis metadata."""

    root = find_sd15_root(sd15_root)
    catalog = load_ktilde_catalog(root)
    ordered_names = [str(name) for name in (names or catalog.keys())]

    bank: Dict[str, Dict[str, Any]] = {}
    for name in ordered_names:
        if name not in catalog:
            raise KeyError(f"Unknown k-tilde '{name}'.")
        artifact_path = root / "ktilde" / f"{name}.npz"
        if not artifact_path.is_file():
            if skip_missing:
                continue
            raise FileNotFoundError(f"K-tilde artifact not found: {artifact_path}")
        k_tilde, probabilities, metadata = load_ktilde_npz_for_analysis(artifact_path)
        probabilities_np = np.asarray(probabilities, dtype=np.float64)
        k_tilde_raw = np.asarray(k_tilde, dtype=np.float64)
        fft_energy_scale = ktilde_unitary_energy_scale(metadata, probabilities_np)
        raw_entry = catalog[name]
        bank[str(raw_entry.get("role", name))] = {
            "name": name,
            "role": str(raw_entry.get("role", name)),
            "label": str(raw_entry.get("label", name)),
            "description": str(raw_entry.get("description", "")),
            "sampling_rank": int(raw_entry.get("sampling_rank", len(bank))),
            "christoffel_law": str(raw_entry.get("christoffel_law", "")),
            "prompt": str(raw_entry.get("prompt", "")),
            "prompt_bank": list(raw_entry.get("prompt_bank", [])),
            "artifact_path": artifact_path,
            "metadata": metadata,
            "K_tilde": k_tilde_raw / fft_energy_scale,
            "K_tilde_raw": k_tilde_raw,
            "fft_energy_scale": fft_energy_scale,
            "fft_energy_convention": "unitary",
            "probabilities": probabilities_np,
        }
    return dict(sorted(bank.items(), key=lambda item: int(item[1]["sampling_rank"])))


def empirical_lambda_self(k_tilde_num: np.ndarray, mu_sampling: np.ndarray) -> float:
    """Return max K_c / mu_cs over the support of a sampling prior."""

    k_flat = np.asarray(k_tilde_num, dtype=np.float64).reshape(-1)
    mu_flat = np.asarray(mu_sampling, dtype=np.float64).reshape(-1)
    ratio = np.zeros_like(k_flat)
    pos = mu_flat > 0.0
    ratio[pos] = k_flat[pos] / mu_flat[pos]
    if np.any(~pos):
        return float("inf")
    return float(ratio.max(initial=0.0))


def empirical_kappa(k_tilde_num: np.ndarray) -> float:
    """Return the empirical kappa surrogate sum K_c."""

    return float(np.asarray(k_tilde_num, dtype=np.float64).sum())


def build_lambda_tables(
    sd15_root: str | Path,
    *,
    names: Optional[Sequence[str]] = None,
    skip_missing: bool = False,
) -> Dict[str, Any]:
    """Build unitary-FFT lambda/kappa tables for the configured SD1.5 k-tilde bank."""

    root = find_sd15_root(sd15_root)
    catalog = load_ktilde_catalog(root)
    requested_names = [str(name) for name in (names or catalog.keys())]
    bank = load_ktilde_bank(root, names=requested_names, skip_missing=skip_missing)
    loaded_names = {str(info["name"]) for info in bank.values()}
    missing_names = [name for name in requested_names if name not in loaded_names]
    if not bank:
        raise FileNotFoundError("No k-tilde artifacts were available for SD1.5 lambda analysis.")

    lambda_rows: List[Dict[str, Any]] = []
    kappa_rows: List[Dict[str, Any]] = []
    for numerator_key, numerator_info in bank.items():
        kappa_hat = empirical_kappa(numerator_info["K_tilde"])
        kappa_rows.append(
            {
                "numerator_class": numerator_key,
                "label": numerator_info["label"],
                "kappa_hat": kappa_hat,
            }
        )
        for sampling_key, sampling_info in bank.items():
            lambda_hat = empirical_lambda_self(numerator_info["K_tilde"], sampling_info["probabilities"])
            mismatch_penalty = lambda_hat / kappa_hat if kappa_hat > 0.0 and np.isfinite(lambda_hat) else np.inf
            lambda_rows.append(
                {
                    "numerator_class": numerator_key,
                    "numerator_label": numerator_info["label"],
                    "sampling_prior": sampling_key,
                    "sampling_label": sampling_info["label"],
                    "lambda_hat": lambda_hat,
                    "kappa_hat": kappa_hat,
                    "mismatch_penalty": mismatch_penalty,
                }
            )

    kappa_df = pd.DataFrame(kappa_rows)
    lambda_df = pd.DataFrame(lambda_rows)
    lambda_table = lambda_df.pivot(index="numerator_class", columns="sampling_prior", values="lambda_hat")
    penalty_table = lambda_df.pivot(index="numerator_class", columns="sampling_prior", values="mismatch_penalty")
    matched_check = lambda_df[lambda_df["numerator_class"] == lambda_df["sampling_prior"]].copy()
    matched_check["abs_lambda_minus_kappa"] = (matched_check["lambda_hat"] - matched_check["kappa_hat"]).abs()
    return {
        "catalog": catalog,
        "bank": bank,
        "requested_names": requested_names,
        "missing_names": missing_names,
        "fft_energy_convention": "unitary",
        "fft_energy_scale": {role: float(info["fft_energy_scale"]) for role, info in bank.items()},
        "kappa_df": kappa_df,
        "lambda_df": lambda_df,
        "lambda_table": lambda_table,
        "penalty_table": penalty_table,
        "matched_check": matched_check,
    }


def _make_linear_cmap(name: str, colors: Sequence[str]):
    """Create a small custom colormap."""

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(colors))


def _mu_subscript(role: str) -> str:
    """Return informative prior subscripts for plot labels."""

    role_text = str(role).strip()
    subscript_map = {
        "k0": "uc",
        "k1_daytime_beach": "db",
        "k2_sunset_beach": "sb",
        "k4_cat": "ca",
    }
    return subscript_map.get(role_text, role_text)


def _prior_descriptor(info: Mapping[str, Any]) -> str:
    """Return the human-readable prior description without the leading k-label."""

    role_text = str(info.get("role", "")).strip()
    descriptor_map = {
        "k0": "Unconditioned",
        "k1_daytime_beach": "Daytime Beach",
        "k2_sunset_beach": "Sunset Beach",
        "k4_cat": "Cat",
    }
    if role_text in descriptor_map:
        return descriptor_map[role_text]

    label_text = str(info.get("label", "")).strip()
    if "(" in label_text and label_text.endswith(")"):
        raw_descriptor = label_text[label_text.find("(") + 1 : -1].strip()
        return " ".join(token.capitalize() for token in raw_descriptor.split())

    description = str(info.get("description", "")).strip()
    if description:
        cleaned = description.replace("Prompt-conditioned Christoffel prior for ", "")
        cleaned = cleaned.replace("Unconditioned Christoffel prior built from ", "")
        cleaned = cleaned.rstrip(".")
        return " ".join(token.capitalize() for token in cleaned.split())

    fallback = str(info.get("role", "")).replace("_", " ").strip()
    return " ".join(token.capitalize() for token in fallback.split())


def _mu_symbol(role: str, *, hat: bool = False) -> str:
    """Return a TeX mu symbol with the right subscript."""

    subscript = _class_symbol(role)
    if hat:
        return rf"\widetilde{{\mu}}_{{{subscript}}}"
    return rf"\mu_{{{subscript}}}"


def _class_symbol(role: str) -> str:
    """Return a TeX class symbol with the right subscript."""

    subscript = _mu_subscript(role)
    return rf"\mathrm{{c}}_{{\mathrm{{{subscript}}}}}"


def _family_symbol(role: str) -> str:
    """Return the family symbol used inside self-difference numerators."""

    return rf"\mathbb{{F}}_{{{_class_symbol(role)}}}"


def _self_numerator_label(info: Mapping[str, Any], *, multiline: bool = True) -> str:
    """Return the empirical self-difference numerator label."""

    del multiline
    family = _family_symbol(str(info.get("role", "")))
    return rf"$\widetilde{{K}}({family}-{family})$"


def _class_plot_label(info: Mapping[str, Any], *, multiline: bool = True) -> str:
    """Return the plot label for one numerator class."""

    del multiline
    return rf"${_class_symbol(str(info.get('role', '')))}$"


def _mu_plot_label(info: Mapping[str, Any], *, multiline: bool = True, hat: bool = True) -> str:
    """Return the plot label for one prior using mu notation."""

    del multiline
    return rf"${_mu_symbol(str(info.get('role', '')), hat=hat)}$"


def _ordered_prior_infos(
    tables: Mapping[str, Any],
    *,
    include_excluded: bool = False,
) -> List[Dict[str, Any]]:
    """Return bank entries in the configured sampling order."""

    bank = dict(tables["bank"])
    preferred_plot_order = {
        "k0": 0,
        "k2_sunset_beach": 1,
        "k1_daytime_beach": 2,
        "k4_cat": 3,
    }
    return [
        dict(info)
        for _, info in sorted(
            bank.items(),
            key=lambda item: (
                int(preferred_plot_order.get(str(item[1].get("role", "")), 10_000)),
                int(item[1].get("sampling_rank", 0)),
            ),
        )
        if include_excluded or str(info.get("role", "")) not in SD15_PLOT_EXCLUDED_ROLES
    ]


def _ordered_heatmap_table(tables: Mapping[str, Any], table_key: str) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Return an ordered heatmap table with mu-based labels."""

    table = pd.DataFrame(tables[table_key])
    ordered_infos = [info for info in _ordered_prior_infos(tables) if str(info["role"]) in table.index]
    ordered_roles = [str(info["role"]) for info in ordered_infos]
    ordered_table = table.loc[ordered_roles, ordered_roles].copy().T
    row_labels = {str(info["role"]): _self_numerator_label(info, multiline=False) for info in ordered_infos}
    col_labels = {str(info["role"]): _mu_plot_label(info, multiline=False, hat=True) for info in ordered_infos}
    ordered_table.index = [col_labels[role] for role in ordered_roles]
    ordered_table.columns = [row_labels[role] for role in ordered_roles]
    return ordered_table, ordered_infos


def _annotation_scale_power(values: np.ndarray) -> int:
    """Choose a clean power-of-ten scaling for displayed annotation values."""

    finite_positive = np.asarray(values, dtype=float)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return 0
    max_value = float(finite_positive.max())
    if max_value < 1.0e4:
        return 0
    return max(0, 3 * int(np.floor(np.log10(max_value) / 3.0)))


def _format_heatmap_number(value: float) -> str:
    """Format a heatmap annotation with paper-friendly precision."""

    if not np.isfinite(value):
        return r"$\infty$"
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return "0"
    if magnitude >= 100.0:
        return f"{value:.0f}"
    if magnitude >= 10.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if magnitude >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if magnitude >= 1.0e-2:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _format_scientific_annotation(value: float) -> str:
    """Format a heatmap annotation compactly using scientific notation when helpful."""

    if not np.isfinite(value):
        return r"$\infty$"
    magnitude = abs(float(value))
    if magnitude == 0.0:
        return "0"
    if magnitude >= 1.0e4 or magnitude < 1.0e-2:
        return f"{value:.2e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return _format_heatmap_number(value)


def format_plain_number(value: Any) -> str:
    """Format numeric values for notebook table displays without scientific notation."""

    try:
        return _format_heatmap_number(float(value))
    except (TypeError, ValueError):
        return str(value)


def _scaled_quantity_label(base_label: str, scale_power: int) -> str:
    """Append a shared scientific-scale multiplier to a display label when needed."""

    if int(scale_power) == 0:
        return base_label
    return f"{base_label} " + rf"$(\times 10^{{{int(scale_power)}}})$"


def _prepare_log_heatmap(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return renderable values plus a stable positive range for log-scaled heatmaps."""

    numeric = np.asarray(table, dtype=float)
    finite_positive = numeric[np.isfinite(numeric) & (numeric > 0.0)]
    if finite_positive.size == 0:
        raise ValueError("Heatmap values must contain at least one finite positive entry.")

    plot_values = numeric.copy()
    finite_max = float(finite_positive.max())
    positive_floor = float(finite_positive.min())

    if np.isinf(plot_values).any():
        plot_values[np.isinf(plot_values)] = finite_max * 1.25
    plot_values[np.isnan(plot_values) | (plot_values <= 0.0)] = positive_floor

    vmin = positive_floor
    vmax = float(np.max(plot_values))
    if np.isclose(vmin, vmax):
        vmax = vmin * 1.01
    return numeric, plot_values, vmin, vmax


def _draw_publication_heatmap(
    ax,
    table: pd.DataFrame,
    *,
    cmap,
    colorbar_label: str,
    annotation_scale_power: int = 0,
    annotation_fontsize: float = 16.0,
    tick_labelsize: Optional[float] = None,
    show_ylabels: bool = True,
    x_rotation: float = 34.0,
) -> None:
    """Draw a publication-style log heatmap with compact annotations."""

    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import LogFormatterMathtext

    raw_values, plot_values, vmin, vmax = _prepare_log_heatmap(table)
    scale_factor = float(10 ** int(annotation_scale_power))
    annotation_values = raw_values / scale_factor
    image = ax.imshow(
        plot_values,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect="equal",
        interpolation="nearest",
    )
    n_rows, n_cols = raw_values.shape
    ax.set_xticks(np.arange(n_cols), labels=list(table.columns))
    ax.set_yticks(np.arange(n_rows))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    if show_ylabels:
        ax.set_yticklabels(list(table.index))
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="x", rotation=x_rotation, length=0, pad=9)
    ax.tick_params(axis="y", length=0, pad=7)
    if tick_labelsize is not None:
        ax.tick_params(axis="x", labelsize=tick_labelsize)
        ax.tick_params(axis="y", labelsize=tick_labelsize)
    for label in ax.get_xticklabels():
        if abs(float(x_rotation)) < 1.0e-9:
            label.set_horizontalalignment("center")
        else:
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax.text(
                col_idx,
                row_idx,
                _format_heatmap_number(annotation_values[row_idx, col_idx]),
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                fontweight="semibold",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": (1.0, 1.0, 1.0, 0.8),
                    "edgecolor": "none",
                },
            )

    for diag_idx in range(min(n_rows, n_cols)):
        ax.add_patch(
            Rectangle(
                (diag_idx - 0.5, diag_idx - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#111827",
                linewidth=2.1,
            )
        )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.ax.tick_params(labelsize=12, width=0.8, length=3)
    colorbar.set_label(colorbar_label)
    colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    colorbar.outline.set_linewidth(0.8)


def plot_lambda_heatmap(
    tables: Mapping[str, Any],
    *,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot the empirical compatibility matrix as a standalone figure."""

    import matplotlib.pyplot as plt

    lambda_table, _ = _ordered_heatmap_table(tables, "lambda_table")
    cmap = _make_linear_cmap("sd15_lambda", SD15_LAMBDA_CMAP_COLORS)
    label = r"$\widetilde{\Lambda}(c_r,c_r,c_s)$"

    with plt.rc_context(SD15_PRESENTATION_RC):
        fig, ax = plt.subplots(figsize=(7.2, 6.8), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.02, hspace=0.02)
        scale_power = _annotation_scale_power(lambda_table.to_numpy(dtype=float))
        _draw_publication_heatmap(
            ax,
            lambda_table,
            cmap=cmap,
            colorbar_label=label,
            annotation_scale_power=scale_power,
            tick_labelsize=14.0,
            show_ylabels=True,
            x_rotation=0.0,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        return _save_plot(fig, output_path, show=show)


def plot_penalty_heatmap(
    tables: Mapping[str, Any],
    *,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot the relative lambda blow-up as a standalone figure."""

    import matplotlib.pyplot as plt

    penalty_table, _ = _ordered_heatmap_table(tables, "penalty_table")
    cmap = _make_linear_cmap("sd15_penalty", SD15_PENALTY_CMAP_COLORS)

    with plt.rc_context(SD15_PRESENTATION_RC):
        fig, ax = plt.subplots(figsize=(7.2, 6.8), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.02, hspace=0.02)
        _draw_publication_heatmap(
            ax,
            penalty_table,
            cmap=cmap,
            colorbar_label=r"$\widetilde{\Lambda}'(c_r,c_s)=\widetilde{\lambda}(c_r,c_r,c_s)/\widetilde{\kappa}(c_r)$",
            annotation_scale_power=0,
            show_ylabels=True,
            x_rotation=0.0,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        return _save_plot(fig, output_path, show=show)


def plot_kappa_bar(
    tables: Mapping[str, Any],
    *,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot the empirical kappa surrogate as a standalone bar chart."""

    import matplotlib.pyplot as plt

    ordered_infos = _ordered_prior_infos(tables)
    ordered_roles = [str(info["role"]) for info in ordered_infos]
    kappa_df = pd.DataFrame(tables["kappa_df"]).set_index("numerator_class").loc[ordered_roles].reset_index()
    kappa_df["display_label"] = [_self_numerator_label(info, multiline=False) for info in ordered_infos]
    bar_colors = [
        SD15_PRIOR_COLORS.get(str(info["role"]), SD15_RECON_COLORS[idx % len(SD15_RECON_COLORS)])
        for idx, info in enumerate(ordered_infos)
    ]

    with plt.rc_context(SD15_PRESENTATION_RC):
        fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.02, hspace=0.02)
        scale_power = _annotation_scale_power(kappa_df["kappa_hat"].to_numpy(dtype=float))
        scale_factor = float(10 ** scale_power)
        scaled_values = kappa_df["kappa_hat"].to_numpy(dtype=float) / scale_factor
        bars = ax.bar(
            np.arange(len(kappa_df)),
            scaled_values,
            color=bar_colors,
            edgecolor="#1F2937",
            linewidth=1.0,
            width=0.72,
        )
        ax.set_xticks(np.arange(len(kappa_df)), labels=list(kappa_df["display_label"]))
        ax.margins(x=0.12)
        ax.tick_params(axis="x", rotation=0, length=0, pad=10, labelsize=11)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
            label.set_rotation_mode("default")
        ax.set_ylabel(_scaled_quantity_label(r"Baseline Complexity $\widetilde{\kappa}(c)$", scale_power))
        ax.set_xlabel("Christoffel Function")
        ax.grid(True, axis="y", alpha=0.24, linestyle=(0, (3, 3)))
        ax.set_axisbelow(True)
        ymax = float(scaled_values.max()) if scaled_values.size else 1.0
        ax.set_ylim(0.0, ymax * 1.18 if ymax > 0.0 else 1.0)
        for bar, value in zip(bars, scaled_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ymax * 0.03,
                _format_heatmap_number(float(value)),
                ha="center",
                va="bottom",
                fontsize=17.0,
                color="#111827",
            )
        return _save_plot(fig, output_path, show=show)


def _reshape_centered_distribution(info: Mapping[str, Any]) -> np.ndarray:
    """Return the 2D centered probability image for one prior."""

    probabilities = np.asarray(info["probabilities"], dtype=np.float64)
    metadata = dict(info.get("metadata", {}))
    height = int(metadata.get("height", int(np.sqrt(probabilities.size))))
    width = int(metadata.get("width", int(probabilities.size // max(height, 1))))
    return probabilities.reshape(height, width)


def distribution_log_limits(tables: Mapping[str, Any], *, roles: Optional[Sequence[str]] = None) -> tuple[float, float]:
    """Choose shared display limits for the per-prior mu visualizations."""

    role_filter = {str(role) for role in (roles or [])}
    lows: List[float] = []
    highs: List[float] = []
    for info in _ordered_prior_infos(tables, include_excluded=bool(role_filter)):
        role = str(info["role"])
        if role_filter and role not in role_filter:
            continue
        centered = _reshape_centered_distribution(info)
        positive = centered[centered > 0.0]
        if positive.size == 0:
            continue
        log_values = np.log10(positive)
        lows.append(float(np.min(log_values)))
        highs.append(float(np.max(log_values)))
    if not lows or not highs:
        return (-18.0, 0.0)
    vmin = float(min(lows))
    vmax = float(max(highs))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return (vmin, vmax)


def plot_sampling_distribution(
    tables: Mapping[str, Any],
    *,
    role: str,
    output_path: Optional[str | Path] = None,
    show: bool = False,
    limits: Optional[tuple[float, float]] = None,
) -> Optional[Path]:
    """Plot one centered sampling distribution as a standalone figure."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterMathtext

    info = dict(tables["bank"][str(role)])
    centered = _reshape_centered_distribution(info)
    vmin, vmax = limits or distribution_log_limits(tables, roles=[str(role)])
    norm_vmin = float(10 ** vmin)
    norm_vmax = float(10 ** vmax)
    cmap = _make_linear_cmap("sd15_distribution", SD15_DISTRIBUTION_CMAP_COLORS)

    with plt.rc_context(SD15_PRESENTATION_RC):
        fig, ax = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.02, hspace=0.02)
        image = ax.imshow(
            centered,
            cmap=cmap,
            norm=LogNorm(vmin=norm_vmin, vmax=norm_vmax),
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_title(_mu_plot_label(info, multiline=False, hat=True), pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        colorbar.ax.tick_params(labelsize=12, width=0.8, length=3)
        colorbar.outline.set_linewidth(0.8)
        return _save_plot(fig, output_path, show=show)


def plot_sampling_distribution_row(
    tables: Mapping[str, Any],
    *,
    roles: Sequence[str],
    output_path: Optional[str | Path] = None,
    show: bool = False,
    limits: Optional[tuple[float, float]] = None,
) -> Optional[Path]:
    """Plot centered sampling distributions on a 2x2 grid with a shared colorbar."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterMathtext

    ordered_roles = [str(role) for role in roles]
    infos = [dict(tables["bank"][role]) for role in ordered_roles]
    vmin, vmax = limits or distribution_log_limits(tables, roles=ordered_roles)
    norm = LogNorm(vmin=float(10 ** vmin), vmax=float(10 ** vmax))

    with plt.rc_context(SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10.0, 9.2),
            constrained_layout=True,
        )
        axes = np.asarray(axes, dtype=object).reshape(-1)
        fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.02, hspace=0.02)
        image = None
        for ax, info in zip(axes, infos):
            centered = _reshape_centered_distribution(info)
            image = ax.imshow(
                centered,
                cmap=_make_linear_cmap("sd15_distribution_row", SD15_DISTRIBUTION_CMAP_COLORS),
                norm=norm,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_title(_mu_plot_label(info, multiline=False, hat=True), pad=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax in axes[len(infos) :]:
            ax.set_visible(False)
        assert image is not None
        colorbar = fig.colorbar(image, ax=list(axes), fraction=0.05, pad=0.02, shrink=0.96, aspect=28)
        colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        colorbar.ax.tick_params(labelsize=13, width=0.9, length=4)
        colorbar.outline.set_linewidth(0.8)
        return _save_plot(fig, output_path, show=show)


def export_lambda_figure_set(
    tables: Mapping[str, Any],
    *,
    output_dir: str | Path,
    show: bool = False,
    file_format: str = "pdf",
    include_kappa: bool = True,
) -> Dict[str, Path]:
    """Save standalone lambda and mu figures for the current SD1.5 bank."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = str(file_format).lstrip(".")
    outputs: Dict[str, Path] = {}
    outputs["lambda_heatmap"] = plot_lambda_heatmap(
        tables,
        output_path=root / f"sd15_lambda_heatmap.{suffix}",
        show=show,
    )
    outputs["penalty_heatmap"] = plot_penalty_heatmap(
        tables,
        output_path=root / f"sd15_lambda_blowup_heatmap.{suffix}",
        show=show,
    )
    if include_kappa:
        outputs["kappa_bar"] = plot_kappa_bar(
            tables,
            output_path=root / f"sd15_kappa_bar.{suffix}",
            show=show,
        )

    limits = distribution_log_limits(tables)
    for info in _ordered_prior_infos(tables):
        role = str(info["role"])
        outputs[f"distribution_{role}"] = plot_sampling_distribution(
            tables,
            role=role,
            output_path=root / f"sd15_sampling_distribution_{role}.{suffix}",
            show=show,
            limits=limits,
        )
    outputs["distribution_row_uc_sb_db_ca"] = plot_sampling_distribution_row(
        tables,
        roles=["k0", "k2_sunset_beach", "k1_daytime_beach", "k4_cat"],
        output_path=root / f"sd15_sampling_distribution_row_uc_sb_db_ca.{suffix}",
        show=show,
        limits=limits,
    )
    return outputs


def plot_lambda_summary(
    tables: Mapping[str, Any],
    *,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Backwards-compatible composite summary plot."""

    import matplotlib.pyplot as plt

    lambda_table, ordered_infos = _ordered_heatmap_table(tables, "lambda_table")
    penalty_table, _ = _ordered_heatmap_table(tables, "penalty_table")
    ordered_roles = [str(info["role"]) for info in ordered_infos]
    kappa_df = pd.DataFrame(tables["kappa_df"]).set_index("numerator_class").loc[ordered_roles].reset_index()
    kappa_df["display_label"] = [_self_numerator_label(info, multiline=False) for info in ordered_infos]
    bar_colors = [
        SD15_PRIOR_COLORS.get(str(info["role"]), SD15_RECON_COLORS[idx % len(SD15_RECON_COLORS)])
        for idx, info in enumerate(ordered_infos)
    ]
    with plt.rc_context(SD15_PRESENTATION_RC):
        fig = plt.figure(figsize=(18.0, 6.4), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.04, wspace=0.04, hspace=0.02)
        grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.95])
        axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
        lambda_scale_power = _annotation_scale_power(lambda_table.to_numpy(dtype=float))
        _draw_publication_heatmap(
            axes[0],
            lambda_table,
            cmap=_make_linear_cmap("sd15_lambda", SD15_LAMBDA_CMAP_COLORS),
            colorbar_label=r"$\widetilde{\Lambda}(c_r,c_r,c_s)$",
            annotation_scale_power=lambda_scale_power,
            tick_labelsize=12.0,
            show_ylabels=True,
            x_rotation=0.0,
        )
        axes[0].set_xlabel("")
        axes[0].set_ylabel("")

        _draw_publication_heatmap(
            axes[1],
            penalty_table,
            cmap=_make_linear_cmap("sd15_penalty", SD15_PENALTY_CMAP_COLORS),
            colorbar_label=r"$\widetilde{\Lambda}'(c_r,c_s)=\widetilde{\lambda}(c_r,c_r,c_s)/\widetilde{\kappa}(c_r)$",
            annotation_scale_power=0,
            show_ylabels=False,
            x_rotation=0.0,
        )
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")

        kappa_scale_power = _annotation_scale_power(kappa_df["kappa_hat"].to_numpy(dtype=float))
        kappa_scale_factor = float(10 ** kappa_scale_power)
        scaled_kappa_values = kappa_df["kappa_hat"].to_numpy(dtype=float) / kappa_scale_factor
        bars = axes[2].bar(
            np.arange(len(kappa_df)),
            scaled_kappa_values,
            color=bar_colors,
            edgecolor="#1F2937",
            linewidth=1.0,
            width=0.72,
        )
        axes[2].set_xticks(np.arange(len(kappa_df)), labels=list(kappa_df["display_label"]))
        axes[2].margins(x=0.12)
        axes[2].tick_params(axis="x", rotation=0, length=0, pad=10, labelsize=11)
        for label in axes[2].get_xticklabels():
            label.set_horizontalalignment("center")
            label.set_rotation_mode("default")
        axes[2].set_ylabel(_scaled_quantity_label(r"Baseline Complexity $\widetilde{\kappa}(c)$", kappa_scale_power))
        axes[2].set_xlabel("Christoffel Function")
        axes[2].grid(True, axis="y", alpha=0.24, linestyle=(0, (3, 3)))
        axes[2].set_axisbelow(True)
        ymax = float(scaled_kappa_values.max()) if scaled_kappa_values.size else 1.0
        axes[2].set_ylim(0.0, ymax * 1.18 if ymax > 0.0 else 1.0)
        for bar, value in zip(bars, scaled_kappa_values):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ymax * 0.03,
                _format_heatmap_number(float(value)),
                ha="center",
                va="bottom",
                fontsize=17.0,
                color="#111827",
            )
        return _save_plot(fig, output_path, show=show)


def load_regression_rows(
    sd15_root: str | Path,
    *,
    tag: str,
    dc_methods: Optional[Sequence[str]] = None,
    sampling_methods: Optional[Sequence[str]] = None,
    include_partial: bool = True,
) -> pd.DataFrame:
    """Load per-run regression rows from a tagged conditioning sweep.

    When ``include_partial`` is true, this also scans completed per-run
    ``run_data.npz`` artifacts so notebooks can show progress before the last
    aggregate ``results_<sampler>.csv`` files have been written.
    """

    root = find_sd15_root(sd15_root)
    dc_method_filter = {str(item) for item in (dc_methods or [])}
    sampling_filter = {str(item) for item in (sampling_methods or [])}
    rows: List[pd.DataFrame] = []
    tag_root = root / "results" / str(tag)
    if not tag_root.is_dir():
        available = discover_regression_tags(root)
        hint = f" Available tags: {', '.join(available)}." if available else ""
        raise FileNotFoundError(f"Results tag not found: {tag_root}.{hint}")

    method_roots = sorted(path for path in tag_root.iterdir() if path.is_dir() and (path / "suite_manifest.json").is_file())
    if not method_roots and (tag_root / "suite_manifest.json").is_file():
        method_roots = [tag_root]

    for method_root in method_roots:
        manifest = load_json(method_root / "suite_manifest.json")
        dc_method = str(manifest.get("dc_method", method_root.name))
        if dc_method_filter and dc_method not in dc_method_filter:
            continue
        for case in list(manifest.get("cases", [])):
            case_tag = str(case.get("tag", "")).strip()
            if not case_tag:
                continue
            case_root = root / "results" / case_tag
            if not case_root.is_dir():
                continue

            csv_by_method = {
                csv_path.stem.replace("results_", ""): csv_path for csv_path in sorted(case_root.glob("results_*.csv"))
            }
            method_names = set(csv_by_method)
            if include_partial:
                method_names.update(_partial_sampling_methods(case_root))

            for sampling_method in sorted(method_names):
                if sampling_filter and sampling_method not in sampling_filter:
                    continue
                frame_parts: List[pd.DataFrame] = []

                csv_path = csv_by_method.get(sampling_method)
                if csv_path is not None:
                    csv_frame = pd.read_csv(csv_path)
                    if not csv_frame.empty:
                        csv_frame["_result_source"] = "results_csv"
                        frame_parts.append(csv_frame)

                if include_partial:
                    partial_frame = _load_partial_run_frame(case_root, sampling_method)
                    if not partial_frame.empty:
                        frame_parts.append(partial_frame)

                if not frame_parts:
                    continue

                frame = pd.concat(frame_parts, ignore_index=True, sort=False)
                frame = _attach_regression_metadata(
                    frame,
                    dc_method=dc_method,
                    sampling_method=sampling_method,
                    case=case,
                    case_tag=case_tag,
                )
                frame = _drop_duplicate_run_rows(frame)
                rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=REGRESSION_ROW_COLUMNS)

    frame = pd.concat(rows, ignore_index=True)
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
        "sampling_rank",
        "recon_rank",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def sweep_rows_to_dataframe(rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Normalize row records into a pandas DataFrame."""

    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    elif not rows:
        frame = pd.DataFrame()
    else:
        frame = pd.DataFrame(list(rows))
    for column in [
        "psnr_db",
        "ssim",
        "pixel_mae",
        "grain",
        "runtime_sec",
        "zero_filled_psnr_db",
        "zero_filled_ssim",
        "zero_filled_pixel_mae",
        "zero_filled_grain",
    ]:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _display_reconstruction_label(condition: Any, label: Any) -> str:
    """Normalize displayed reconstruction labels across notebooks."""

    if str(condition) == "unprompted":
        return UNPROMPTED_DISPLAY_LABEL
    return str(label)


def _z_value_for_confidence_level(confidence_level: float) -> float:
    """Return the two-sided normal critical value for a confidence level."""

    level = float(confidence_level)
    if level <= 0.0:
        return 0.0
    level = min(level, 0.999999)
    return float(NormalDist().inv_cdf(0.5 + 0.5 * level))


def _metric_band_column(metric: str, band_mode: str) -> str:
    """Return the summary column name used for uncertainty bands."""

    if str(band_mode) == "std":
        return f"{metric}_std"
    if str(band_mode) == "sem":
        return f"{metric}_sem"
    if str(band_mode) == "ci":
        return f"{metric}_ci_halfwidth"
    raise KeyError(f"Unsupported uncertainty band mode '{band_mode}'.")


def _aggregate_metric_statistics(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Aggregate means plus std/SEM/CI columns for one or more metrics."""

    if frame.empty:
        return pd.DataFrame(columns=list(group_cols))

    grouped = (
        frame.groupby(list(group_cols), dropna=False, sort=False)[list(metric_cols)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    flattened_columns: list[str] = []
    for column in grouped.columns:
        if isinstance(column, tuple):
            name, stat = column
            if stat == "":
                flattened_columns.append(str(name))
            elif stat == "mean":
                flattened_columns.append(str(name))
            else:
                flattened_columns.append(f"{name}_{stat}")
        else:
            flattened_columns.append(str(column))
    grouped.columns = flattened_columns

    z_value = _z_value_for_confidence_level(confidence_level)
    for metric in metric_cols:
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        sem_col = f"{metric}_sem"
        ci_col = f"{metric}_ci_halfwidth"
        grouped[std_col] = grouped[std_col].fillna(0.0)
        grouped[count_col] = grouped[count_col].fillna(0).astype(int)
        grouped[sem_col] = 0.0
        valid = grouped[count_col] > 0
        grouped.loc[valid, sem_col] = grouped.loc[valid, std_col] / np.sqrt(grouped.loc[valid, count_col].astype(float))
        grouped[ci_col] = grouped[sem_col] * z_value
    return grouped


def _validate_single_panel_subset(frame: pd.DataFrame) -> None:
    """Make sure one plot call corresponds to one DC method and sampling method."""

    if frame.empty:
        return
    if frame["dc_method"].nunique(dropna=True) > 1:
        raise ValueError("Filter to one dc_method before calling this plotting helper.")
    if frame["sampling_method"].nunique(dropna=True) > 1:
        raise ValueError("Filter to one sampling_method before calling this plotting helper.")


def build_metric_summary_table(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Aggregate sweep metrics over all trials with uncertainty columns."""

    frame = sweep_rows_to_dataframe(rows)
    if frame.empty:
        return pd.DataFrame(columns=MEAN_METRIC_COLUMNS)
    summary = _aggregate_metric_statistics(
        frame,
        group_cols=[
            "dc_method",
            "dc_method_label",
            "sampling_method",
            "sampling_method_label",
            "sampling_condition",
            "sampling_label",
            "sampling_rank",
            "reconstruction_condition",
            "reconstruction_label",
            "recon_rank",
            "samp_perc",
        ],
        metric_cols=SUMMARY_METRICS,
        confidence_level=confidence_level,
    )
    if "reconstruction_condition" in summary.columns and "reconstruction_label" in summary.columns:
        summary["reconstruction_label"] = [
            _display_reconstruction_label(condition, label)
            for condition, label in zip(summary["reconstruction_condition"], summary["reconstruction_label"])
        ]
    return summary.sort_values(
        ["dc_method", "sampling_method", "sampling_rank", "recon_rank", "samp_perc"],
        kind="stable",
    ).reset_index(drop=True)


def build_mean_metric_table(rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Average metrics over items and repeats for one conditioning sweep."""

    summary = build_metric_summary_table(rows, confidence_level=DEFAULT_CONFIDENCE_LEVEL)
    if summary.empty:
        return pd.DataFrame(columns=MEAN_METRIC_COLUMNS)
    return summary[MEAN_METRIC_COLUMNS].copy()


def build_prior_delta_table(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    baseline_sampling: str = "k0",
    conditioned_sampling: Optional[str] = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Compute sampling-prior deltas relative to a baseline prior."""

    frame = sweep_rows_to_dataframe(rows)
    if frame.empty:
        return pd.DataFrame()
    _validate_single_panel_subset(frame)

    available = frame["sampling_condition"].drop_duplicates().tolist()
    if str(baseline_sampling) not in available:
        raise KeyError(f"Baseline sampling condition '{baseline_sampling}' was not found.")
    if conditioned_sampling is None:
        conditioned_sampling = next((name for name in available if str(name) != str(baseline_sampling)), None)
    if conditioned_sampling is None:
        return pd.DataFrame()

    baseline_df = frame[frame["sampling_condition"] == str(baseline_sampling)][
        [
            "item_id",
            "repeat_id",
            "reconstruction_condition",
            "reconstruction_label",
            "recon_rank",
            "samp_perc",
            "psnr_db",
            "ssim",
            "pixel_mae",
            "grain",
        ]
    ].rename(
        columns={
            "psnr_db": "psnr_db_baseline",
            "ssim": "ssim_baseline",
            "pixel_mae": "pixel_mae_baseline",
            "grain": "grain_baseline",
        }
    )
    conditioned_df = frame[frame["sampling_condition"] == str(conditioned_sampling)][
        [
            "item_id",
            "repeat_id",
            "reconstruction_condition",
            "reconstruction_label",
            "recon_rank",
            "samp_perc",
            "psnr_db",
            "ssim",
            "pixel_mae",
            "grain",
        ]
    ].rename(
        columns={
            "psnr_db": "psnr_db_conditioned",
            "ssim": "ssim_conditioned",
            "pixel_mae": "pixel_mae_conditioned",
            "grain": "grain_conditioned",
        }
    )
    merged = conditioned_df.merge(
        baseline_df,
        on=["item_id", "repeat_id", "reconstruction_condition", "reconstruction_label", "recon_rank", "samp_perc"],
        how="inner",
    )
    merged["delta_psnr_db"] = merged["psnr_db_conditioned"] - merged["psnr_db_baseline"]
    merged["delta_ssim"] = merged["ssim_conditioned"] - merged["ssim_baseline"]
    merged["delta_pixel_mae"] = merged["pixel_mae_conditioned"] - merged["pixel_mae_baseline"]
    merged["delta_grain"] = merged["grain_conditioned"] - merged["grain_baseline"]

    summary = _aggregate_metric_statistics(
        merged,
        group_cols=["reconstruction_condition", "reconstruction_label", "recon_rank", "samp_perc"],
        metric_cols=["delta_psnr_db", "delta_ssim", "delta_pixel_mae", "delta_grain"],
        confidence_level=confidence_level,
    )
    summary["baseline_sampling"] = str(baseline_sampling)
    summary["conditioned_sampling"] = str(conditioned_sampling)
    baseline_match = frame[frame["sampling_condition"] == str(baseline_sampling)]["sampling_label"]
    conditioned_match = frame[frame["sampling_condition"] == str(conditioned_sampling)]["sampling_label"]
    summary["baseline_sampling_label"] = str(baseline_match.iloc[0]) if not baseline_match.empty else str(baseline_sampling)
    summary["conditioned_sampling_label"] = (
        str(conditioned_match.iloc[0]) if not conditioned_match.empty else str(conditioned_sampling)
    )
    summary["reconstruction_label"] = [
        _display_reconstruction_label(condition, label)
        for condition, label in zip(summary["reconstruction_condition"], summary["reconstruction_label"])
    ]
    return summary.sort_values(["recon_rank", "samp_perc"], kind="stable").reset_index(drop=True)


def build_reconstruction_delta_table(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    baseline_reconstruction: Optional[str] = "unprompted",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Compute reconstruction-prompt deltas relative to a baseline prompt."""

    frame = sweep_rows_to_dataframe(rows)
    if frame.empty:
        return pd.DataFrame()
    _validate_single_panel_subset(frame)

    available = frame["reconstruction_condition"].drop_duplicates().tolist()
    if not available:
        return pd.DataFrame()
    if baseline_reconstruction is None:
        baseline_reconstruction = str(available[0])
    if str(baseline_reconstruction) not in available:
        raise KeyError(f"Baseline reconstruction condition '{baseline_reconstruction}' was not found.")

    baseline_df = frame[frame["reconstruction_condition"] == str(baseline_reconstruction)][
        [
            "item_id",
            "repeat_id",
            "sampling_condition",
            "sampling_label",
            "sampling_rank",
            "samp_perc",
            "psnr_db",
            "ssim",
            "pixel_mae",
            "grain",
        ]
    ].rename(
        columns={
            "psnr_db": "psnr_db_baseline",
            "ssim": "ssim_baseline",
            "pixel_mae": "pixel_mae_baseline",
            "grain": "grain_baseline",
        }
    )
    frames: List[pd.DataFrame] = []
    for reconstruction_condition in available:
        if str(reconstruction_condition) == str(baseline_reconstruction):
            continue
        recon_df = frame[frame["reconstruction_condition"] == str(reconstruction_condition)][
            [
                "item_id",
                "repeat_id",
                "sampling_condition",
                "sampling_label",
                "sampling_rank",
                "samp_perc",
                "reconstruction_condition",
                "psnr_db",
                "ssim",
                "pixel_mae",
                "grain",
                "reconstruction_label",
                "recon_rank",
            ]
        ].rename(
            columns={
                "psnr_db": "psnr_db_prompt",
                "ssim": "ssim_prompt",
                "pixel_mae": "pixel_mae_prompt",
                "grain": "grain_prompt",
            }
        )
        merged = recon_df.merge(
            baseline_df,
            on=["item_id", "repeat_id", "sampling_condition", "sampling_label", "sampling_rank", "samp_perc"],
            how="inner",
        )
        merged["delta_psnr_db"] = merged["psnr_db_prompt"] - merged["psnr_db_baseline"]
        merged["delta_ssim"] = merged["ssim_prompt"] - merged["ssim_baseline"]
        merged["delta_pixel_mae"] = merged["pixel_mae_prompt"] - merged["pixel_mae_baseline"]
        merged["delta_grain"] = merged["grain_prompt"] - merged["grain_baseline"]
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    merged_frame = pd.concat(frames, ignore_index=True)
    summary = _aggregate_metric_statistics(
        merged_frame,
        group_cols=[
            "sampling_condition",
            "sampling_label",
            "sampling_rank",
            "samp_perc",
            "reconstruction_condition",
            "reconstruction_label",
            "recon_rank",
        ],
        metric_cols=["delta_psnr_db", "delta_ssim", "delta_pixel_mae", "delta_grain"],
        confidence_level=confidence_level,
    )
    summary["baseline_reconstruction"] = str(baseline_reconstruction)
    summary["reconstruction_label"] = [
        _display_reconstruction_label(condition, label)
        for condition, label in zip(summary["reconstruction_condition"], summary["reconstruction_label"])
    ]
    return summary.sort_values(
        ["sampling_rank", "recon_rank", "samp_perc"],
        kind="stable",
    ).reset_index(drop=True)


def _save_plot(fig, outpath: Optional[str | Path], *, show: bool = False) -> Optional[Path]:
    """Save and optionally show a matplotlib figure."""

    if outpath is not None:
        file_path = Path(outpath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path, dpi=SD15_EXPORT_DPI, bbox_inches="tight")
    else:
        file_path = None
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return file_path


def _sampling_tick_labels(x_values: Sequence[float]) -> List[str]:
    """Format log-scale sampling-fraction tick labels."""

    labels: List[str] = []
    for value in x_values:
        if value < 0.01:
            labels.append(f"{value:0.5f}")
        elif value < 0.1:
            labels.append(f"{value:0.3f}")
        else:
            labels.append(f"{value:0.2f}".rstrip("0").rstrip("."))
    return labels


def _apply_sampling_axis(ax, x_values: Sequence[float]) -> None:
    """Apply the presentation-style sampling axis formatting."""

    ticks = sorted({float(x) for x in x_values})
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(_sampling_tick_labels(ticks), rotation=35, ha="right")
    ax.set_xlabel(SD15_X_LABEL)
    ax.grid(True, which="major", axis="both", alpha=0.28, linestyle="--")


def plot_metric_by_sampling_prior(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    metric: str = "psnr_db",
    metric_limits: Optional[tuple[float, float]] = None,
    show_confidence_intervals: bool = True,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    band_mode: str = "ci",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot mean reconstruction quality by sampling prior and reconstruction condition."""

    import matplotlib.pyplot as plt

    summary_table = build_metric_summary_table(rows, confidence_level=confidence_level)
    if summary_table.empty:
        raise ValueError("Sweep table is empty; there is nothing to plot.")
    _validate_single_panel_subset(summary_table)

    sampling_cases = (
        summary_table[["sampling_condition", "sampling_label", "sampling_rank"]]
        .drop_duplicates()
        .sort_values("sampling_rank", kind="stable")
    )
    recon_cases = (
        summary_table[["reconstruction_condition", "reconstruction_label", "recon_rank"]]
        .drop_duplicates()
        .sort_values("recon_rank", kind="stable")
    )
    band_column = _metric_band_column(metric, band_mode)

    with plt.rc_context(SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(
            1,
            len(sampling_cases),
            figsize=(7.8 * len(sampling_cases), 5.8),
            sharey=True,
            constrained_layout=True,
        )
        axes_array = np.atleast_1d(axes)
        for ax, (_, sampling_case) in zip(axes_array, sampling_cases.iterrows()):
            subset = summary_table[summary_table["sampling_condition"] == sampling_case["sampling_condition"]]
            for recon_idx, (_, recon_case) in enumerate(recon_cases.iterrows()):
                group = subset[subset["reconstruction_condition"] == recon_case["reconstruction_condition"]].sort_values(
                    "samp_perc",
                    kind="stable",
                )
                if group.empty:
                    continue
                ax.plot(
                    group["samp_perc"],
                    group[metric],
                    label=recon_case["reconstruction_label"],
                    color=SD15_RECON_COLORS[recon_idx % len(SD15_RECON_COLORS)],
                    marker=SD15_RECON_MARKERS[recon_idx % len(SD15_RECON_MARKERS)],
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                )
                if show_confidence_intervals and band_column in group.columns:
                    band = group[band_column].fillna(0.0).to_numpy(dtype=float)
                    if np.any(band > 0.0):
                        x = group["samp_perc"].to_numpy(dtype=float)
                        y = group[metric].to_numpy(dtype=float)
                        ax.fill_between(x, y - band, y + band, color=SD15_RECON_COLORS[recon_idx % len(SD15_RECON_COLORS)], alpha=0.16, linewidth=0)
            ax.set_title(str(sampling_case["sampling_label"]))
            _apply_sampling_axis(ax, subset["samp_perc"].tolist())
            if metric_limits is not None:
                ax.set_ylim(*metric_limits)

        axes_array[0].set_ylabel(SD15_METRIC_LABELS[metric])
        axes_array[-1].legend(loc="best", frameon=True)
        return _save_plot(fig, output_path, show=show)


def plot_sampling_delta(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    metric: str = "psnr_db",
    baseline_sampling: str = "k0",
    conditioned_sampling: Optional[str] = None,
    show_confidence_intervals: bool = True,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    band_mode: str = "ci",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot conditioned-prior minus baseline-prior deltas."""

    import matplotlib.pyplot as plt

    delta_table = build_prior_delta_table(
        rows,
        baseline_sampling=baseline_sampling,
        conditioned_sampling=conditioned_sampling,
        confidence_level=confidence_level,
    )
    if delta_table.empty:
        raise ValueError("Need at least two sampling priors to plot a prior delta.")

    recon_cases = (
        delta_table[["reconstruction_condition", "reconstruction_label", "recon_rank"]]
        .drop_duplicates()
        .sort_values("recon_rank", kind="stable")
    )
    if metric == "psnr_db":
        y_column = "delta_psnr_db"
        y_label = r"$\Delta \mathrm{PSNR}\ \mathrm{(conditioned - baseline)}$"
    elif metric == "ssim":
        y_column = "delta_ssim"
        y_label = r"$\Delta \mathrm{SSIM}\ \mathrm{(conditioned - baseline)}$"
    elif metric == "pixel_mae":
        y_column = "delta_pixel_mae"
        y_label = r"$\Delta \mathrm{Per\!-\!Pixel\ MAE}\ \mathrm{(conditioned - baseline)}$"
    elif metric == "grain":
        y_column = "delta_grain"
        y_label = r"$\Delta \mathrm{Grain}\ \mathrm{(conditioned - baseline)}$"
    else:
        raise KeyError(f"Unsupported metric '{metric}'.")
    band_column = _metric_band_column(y_column, band_mode)

    conditioned_label = str(delta_table["conditioned_sampling_label"].iloc[0])
    baseline_label = str(delta_table["baseline_sampling_label"].iloc[0])
    with plt.rc_context(SD15_PRESENTATION_RC):
        fig, ax = plt.subplots(figsize=(10.6, 5.8), constrained_layout=True)
        for recon_idx, (_, recon_case) in enumerate(recon_cases.iterrows()):
            group = delta_table[delta_table["reconstruction_condition"] == recon_case["reconstruction_condition"]].sort_values(
                "samp_perc",
                kind="stable",
            )
            ax.plot(
                group["samp_perc"],
                group[y_column],
                label=str(recon_case["reconstruction_label"]),
                color=SD15_RECON_COLORS[recon_idx % len(SD15_RECON_COLORS)],
                marker=SD15_RECON_MARKERS[recon_idx % len(SD15_RECON_MARKERS)],
                markerfacecolor="white",
                markeredgewidth=1.5,
            )
            if show_confidence_intervals and band_column in group.columns:
                band = group[band_column].fillna(0.0).to_numpy(dtype=float)
                if np.any(band > 0.0):
                    x = group["samp_perc"].to_numpy(dtype=float)
                    y = group[y_column].to_numpy(dtype=float)
                    ax.fill_between(x, y - band, y + band, color=SD15_RECON_COLORS[recon_idx % len(SD15_RECON_COLORS)], alpha=0.16, linewidth=0)
        ax.axhline(0.0, color="#6B7280", linewidth=1.5, linestyle=":")
        _apply_sampling_axis(ax, delta_table["samp_perc"].tolist())
        ax.set_ylabel(y_label)
        ax.set_title(f"{conditioned_label} - {baseline_label}")
        ax.legend(loc="best", frameon=True)
        return _save_plot(fig, output_path, show=show)


def plot_reconstruction_delta(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    metric: str = "psnr_db",
    baseline_reconstruction: Optional[str] = "unprompted",
    show_confidence_intervals: bool = True,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    band_mode: str = "ci",
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot reconstruction-condition gains relative to a baseline prompt."""

    import matplotlib.pyplot as plt

    delta_table = build_reconstruction_delta_table(
        rows,
        baseline_reconstruction=baseline_reconstruction,
        confidence_level=confidence_level,
    )
    if delta_table.empty:
        raise ValueError("Need at least two reconstruction conditions to plot a prompt delta.")

    sampling_cases = (
        delta_table[["sampling_condition", "sampling_label", "sampling_rank"]]
        .drop_duplicates()
        .sort_values("sampling_rank", kind="stable")
    )
    recon_cases = (
        delta_table[["reconstruction_condition", "reconstruction_label", "recon_rank"]]
        .drop_duplicates()
        .sort_values("recon_rank", kind="stable")
    )
    if metric == "psnr_db":
        y_column = "delta_psnr_db"
        y_label = r"$\Delta \mathrm{PSNR}\ \mathrm{vs\ baseline\ reconstruction}$"
    elif metric == "ssim":
        y_column = "delta_ssim"
        y_label = r"$\Delta \mathrm{SSIM}\ \mathrm{vs\ baseline\ reconstruction}$"
    elif metric == "pixel_mae":
        y_column = "delta_pixel_mae"
        y_label = r"$\Delta \mathrm{Per\!-\!Pixel\ MAE}\ \mathrm{vs\ baseline\ reconstruction}$"
    elif metric == "grain":
        y_column = "delta_grain"
        y_label = r"$\Delta \mathrm{Grain}\ \mathrm{vs\ baseline\ reconstruction}$"
    else:
        raise KeyError(f"Unsupported metric '{metric}'.")
    band_column = _metric_band_column(y_column, band_mode)

    with plt.rc_context(SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(
            1,
            len(sampling_cases),
            figsize=(7.8 * len(sampling_cases), 5.8),
            sharey=True,
            constrained_layout=True,
        )
        axes_array = np.atleast_1d(axes)
        for ax, (_, sampling_case) in zip(axes_array, sampling_cases.iterrows()):
            subset = delta_table[delta_table["sampling_condition"] == sampling_case["sampling_condition"]]
            for recon_idx, (_, recon_case) in enumerate(recon_cases.iterrows()):
                group = subset[subset["reconstruction_condition"] == recon_case["reconstruction_condition"]].sort_values(
                    "samp_perc",
                    kind="stable",
                )
                if group.empty:
                    continue
                ax.plot(
                    group["samp_perc"],
                    group[y_column],
                    label=str(recon_case["reconstruction_label"]),
                    color=SD15_RECON_COLORS[recon_idx % len(SD15_RECON_COLORS)],
                    marker=SD15_RECON_MARKERS[recon_idx % len(SD15_RECON_MARKERS)],
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                )
                if show_confidence_intervals and band_column in group.columns:
                    band = group[band_column].fillna(0.0).to_numpy(dtype=float)
                    if np.any(band > 0.0):
                        x = group["samp_perc"].to_numpy(dtype=float)
                        y = group[y_column].to_numpy(dtype=float)
                        ax.fill_between(x, y - band, y + band, color=SD15_RECON_COLORS[recon_idx % len(SD15_RECON_COLORS)], alpha=0.16, linewidth=0)
            ax.axhline(0.0, color="#6B7280", linewidth=1.5, linestyle=":")
            ax.set_title(str(sampling_case["sampling_label"]))
            _apply_sampling_axis(ax, subset["samp_perc"].tolist())

        axes_array[0].set_ylabel(y_label)
        axes_array[-1].legend(loc="best", frameon=True)
        return _save_plot(fig, output_path, show=show)
