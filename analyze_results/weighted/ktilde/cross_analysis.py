"""Loading, validation, and plotting for ordered cross-class K-tilde artifacts."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ZETA = 0.5
CROSS_LAWS = (
    (
        "uc",
        "",
        "sunset beach",
        "Ktilde_SD15__fft__cross_k0unconditioned_minus_k2sunsetbeach_512x512_S10000_ns20",
        r"$\mathbb F_{c_{\mathrm{uc}}}-\mathbb F_{c_{\mathrm{sb}}}$",
        r"$\widetilde{\mu}_{c_{\mathrm{uc}},c_{\mathrm{sb}}}$",
    ),
    (
        "sb",
        "sunset beach",
        "sunset beach",
        "Ktilde_SD15__fft__k2sunsetbeach_512x512_S10000_ns20",
        r"$\mathbb F_{c_{\mathrm{sb}}}-\mathbb F_{c_{\mathrm{sb}}}$",
        r"$\widetilde{\mu}_{c_{\mathrm{sb}},c_{\mathrm{sb}}}$",
    ),
    (
        "db",
        "daytime beach",
        "sunset beach",
        "Ktilde_SD15__fft__cross_k1daytimebeach_minus_k2sunsetbeach_512x512_S10000_ns20",
        r"$\mathbb F_{c_{\mathrm{db}}}-\mathbb F_{c_{\mathrm{sb}}}$",
        r"$\widetilde{\mu}_{c_{\mathrm{db}},c_{\mathrm{sb}}}$",
    ),
    (
        "ca",
        "cat",
        "sunset beach",
        "Ktilde_SD15__fft__cross_k4cat_minus_k2sunsetbeach_512x512_S10000_ns20",
        r"$\mathbb F_{c_{\mathrm{ca}}}-\mathbb F_{c_{\mathrm{sb}}}$",
        r"$\widetilde{\mu}_{c_{\mathrm{ca}},c_{\mathrm{sb}}}$",
    ),
)


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository from a notebook or module path."""

    origin = Path(start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "ktilde" / "weighted" / "config_cross_class_s10000.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the ActiveConditionalGCS repository.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
ANALYSIS_ROOT = PROJECT_ROOT / "analyze_results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import sd15_conditioning_experiment as shared_analysis
import sd15_cfg_ablation_analysis as plot_style


CONFIG_PATH = PROJECT_ROOT / "ktilde" / "weighted" / "config_cross_class_s10000.json"
ARTIFACT_ROOT = PROJECT_ROOT / "ktilde" / "weighted"
FIGURE_ROOT = PROJECT_ROOT / "results" / "weighted" / "ktilde" / "figures"
SELF_CONFIG_PATH = PROJECT_ROOT / "ktilde" / "weighted" / "config_convergence.json"
OOD_ANALYSIS_PATH = PROJECT_ROOT / "analyze_results" / "weighted" / "out_of_range" / "analysis.py"
PROMPT_MATCHED_ANALYSIS_PATH = (
    PROJECT_ROOT / "analyze_results" / "weighted" / "prompt_matched" / "analysis.py"
)

SAMPLING_LABELS = {
    "k0": r"$\widetilde{\mu}_{c_{\mathrm{uc}}}$",
    "k1_daytime_beach": r"$\widetilde{\mu}_{c_{\mathrm{db}}}$",
    "k2_sunset_beach": r"$\widetilde{\mu}_{c_{\mathrm{sb}}}$",
    "k4_cat": r"$\widetilde{\mu}_{c_{\mathrm{ca}}}$",
}

SAMPLING_ORDER = ("k0", "k2_sunset_beach", "k1_daytime_beach", "k4_cat")
RECOVERY_ORDER = ("unprompted", "sunset_beach", "daytime_beach", "cat")
NUMERATOR_TICK_LABELS = {
    "unprompted": r"$\widetilde K(\mathbb F_{c_{\mathrm{uc}}}-\mathbb F_{c_{\mathrm{sb}}})$",
    "daytime_beach": r"$\widetilde K(\mathbb F_{c_{\mathrm{db}}}-\mathbb F_{c_{\mathrm{sb}}})$",
    "sunset_beach": r"$\widetilde K(\mathbb F_{c_{\mathrm{sb}}}-\mathbb F_{c_{\mathrm{sb}}})$",
    "cat": r"$\widetilde K(\mathbb F_{c_{\mathrm{ca}}}-\mathbb F_{c_{\mathrm{sb}}})$",
}

RECOVERY_NUMERATORS = {
    "unprompted": ("cross", "uc", r"$\widetilde K(\mathbb F_{c_{\mathrm{uc}}}-\mathbb F_{c_{\mathrm{sb}}})$"),
    "daytime_beach": ("cross", "db", r"$\widetilde K(\mathbb F_{c_{\mathrm{db}}}-\mathbb F_{c_{\mathrm{sb}}})$"),
    "sunset_beach": ("self", "k2_sunset_beach", r"$\widetilde K(\mathbb F_{c_{\mathrm{sb}}}-\mathbb F_{c_{\mathrm{sb}}})$"),
    "cat": ("cross", "ca", r"$\widetilde K(\mathbb F_{c_{\mathrm{ca}}}-\mathbb F_{c_{\mathrm{sb}}})$"),
}

PERFORMANCE_METRICS = ("bp_best_loss", "psnr_db", "ssim", "lpips", "pixel_mae")


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one K-tilde artifact without enabling pickled arrays."""

    with np.load(path, allow_pickle=False) as payload:
        k_tilde = np.asarray(payload["K_tilde"], dtype=np.float64).reshape(-1)
        probability = np.asarray(payload["prob"], dtype=np.float64).reshape(-1)
        metadata = json.loads(str(payload["meta"]))
    return k_tilde, probability, metadata


def load_bank(*, zeta: float = ZETA) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Load available cross-class laws and report missing or incompatible jobs."""

    from src.ktilde import regularize_sampling_probabilities

    if not 0.0 <= float(zeta) < 1.0:
        raise ValueError("zeta must lie in [0, 1).")
    bank: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for rank, (key, first_prompt, second_prompt, name, class_label, law_label) in enumerate(CROSS_LAWS):
        # The sunset/sunset entry is the self-difference special case already
        # used by the weighted reconstruction study
        path = ARTIFACT_ROOT / f"{name}.npz"
        row: dict[str, Any] = {
            "rank": rank,
            "key": key,
            "first_prompt": first_prompt,
            "second_prompt": second_prompt,
            "artifact_name": name,
            "artifact_path": str(path),
            "status": "missing",
        }
        if not path.is_file():
            rows.append(row)
            continue
        k_tilde, raw_probability, metadata = _load_npz(path)
        if str(metadata.get("prompt")) != first_prompt:
            raise ValueError(f"First prompt mismatch in {path}.")
        if key == "sb":
            if not bool(metadata.get("pair_same_prompt", False)):
                raise ValueError(f"Self-pair metadata mismatch in {path}.")
        elif str(metadata.get("pair_prompt")) != second_prompt:
            raise ValueError(f"Second prompt mismatch in {path}.")
        if int(metadata.get("max_samples", -1)) != 10000:
            raise ValueError(f"Secant budget mismatch in {path}.")
        if int(metadata.get("seed", -1)) != 12345:
            raise ValueError(f"Seed mismatch in {path}.")
        if float(metadata.get("guidance_scale", -1.0)) != 7.5:
            raise ValueError(f"Sampling CFG mismatch in {path}.")
        if not np.all(np.isfinite(k_tilde)) or np.any(k_tilde < 0.0):
            raise ValueError(f"Invalid K-tilde values in {path}.")
        if not np.all(np.isfinite(raw_probability)) or np.any(raw_probability < 0.0):
            raise ValueError(f"Invalid probabilities in {path}.")
        if not np.isclose(raw_probability.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"Probability law in {path} does not sum to one.")
        probability = regularize_sampling_probabilities(raw_probability, float(zeta))
        height = int(metadata["height"])
        width = int(metadata["width"])
        bank[key] = {
            **row,
            "class_label": class_label,
            "law_label": law_label,
            "K_tilde_raw": k_tilde,
            "raw_probability": raw_probability,
            "probability": probability,
            "height": height,
            "width": width,
            "metadata": metadata,
            "zeta": float(zeta),
        }
        row.update(
            {
                "status": "complete",
                "raw_probability_min": float(raw_probability.min()),
                "raw_probability_max": float(raw_probability.max()),
                "regularized_probability_min": float(probability.min()),
                "regularized_probability_max": float(probability.max()),
                "probability_floor": float(zeta) / raw_probability.size,
            }
        )
        rows.append(row)
    return bank, pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def _load_study_analysis(path: Path, module_name: str):
    """Load a notes-to-self study helper from its explicit path."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing weighted-study analysis helper: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ood_rows() -> pd.DataFrame:
    """Load the completed 600-row weighted out-of-range reconstruction study."""

    rows = _load_study_analysis(
        OOD_ANALYSIS_PATH,
        "weighted_out_of_range_analysis",
    ).load_rows(include_partial=True)
    if len(rows) != 600:
        raise ValueError(f"Expected 600 completed out-of-range rows, found {len(rows)}.")
    return rows


def load_prompt_matched_rows() -> pd.DataFrame:
    """Load all currently completed rows from the prompt-matched study."""

    return _load_study_analysis(
        PROMPT_MATCHED_ANALYSIS_PATH,
        "weighted_prompt_matched_analysis",
    ).load_rows(include_partial=True)


def balanced_prompt_matched_rows(rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Retain rate/trial pairs completed by all 16 Christoffel combinations."""

    frame = load_prompt_matched_rows() if rows is None else rows.copy()
    if frame.empty:
        return frame
    frame = frame[frame["sampling_condition"].astype(str).isin(SAMPLING_ORDER)].copy()
    cell_columns = ["sampling_condition", "reconstruction_condition"]
    pair_columns = ["samp_perc", "repeat_id"]
    expected_cells = len(SAMPLING_ORDER) * len(RECOVERY_ORDER)
    counts = (
        frame.groupby(pair_columns, dropna=False)[cell_columns]
        .apply(lambda group: len(group.drop_duplicates()))
        .rename("completed_cells")
        .reset_index()
    )
    complete_pairs = counts[counts["completed_cells"] == expected_cells][pair_columns]
    if complete_pairs.empty:
        return frame.iloc[0:0].copy()
    return frame.merge(complete_pairs, on=pair_columns, how="inner", validate="many_to_one")


def compatibility_table(
    cross_bank: dict[str, dict[str, Any]],
    *,
    probability_regularization_zeta: float = ZETA,
) -> pd.DataFrame:
    """Evaluate cross/self compatibility against four Christoffel laws.

    The sampling laws use the same zeta regularization as the weighted
    reconstruction study. Baseline laws are intentionally excluded from this
    cross/self Christoffel analysis.
    """

    zeta = float(probability_regularization_zeta)
    self_bank = shared_analysis.load_ktilde_bank(
        PROJECT_ROOT,
        config_path=SELF_CONFIG_PATH,
        probability_regularization_zeta=zeta,
    )
    if not self_bank:
        raise FileNotFoundError("No self-class S10000 Christoffel artifacts were found.")
    shape_entry = next(iter(self_bank.values()))
    height = int(shape_entry["metadata"]["height"])
    width = int(shape_entry["metadata"]["width"])
    n = height * width

    sampling_laws = {
        key: np.asarray(self_bank[key]["probabilities"], dtype=np.float64).reshape(-1)
        for key in SAMPLING_ORDER
    }

    rows: list[dict[str, Any]] = []
    for recovery_condition, (source, key, numerator_label) in RECOVERY_NUMERATORS.items():
        if source == "cross":
            if key not in cross_bank:
                continue
            numerator_entry = cross_bank[key]
            numerator = np.asarray(numerator_entry["K_tilde_raw"], dtype=np.float64).reshape(-1) / n
        else:
            numerator_entry = self_bank[key]
            numerator = np.asarray(numerator_entry["K_tilde"], dtype=np.float64).reshape(-1)
        kappa_hat = shared_analysis.empirical_kappa(numerator)
        for sampling_condition, probability in sampling_laws.items():
            lambda_hat = shared_analysis.empirical_lambda_self(numerator, probability)
            penalty = lambda_hat / kappa_hat if kappa_hat > 0.0 else np.inf
            rows.append(
                {
                    "reconstruction_condition": recovery_condition,
                    "numerator_source": source,
                    "numerator_key": key,
                    "numerator_label": numerator_label,
                    "sampling_condition": sampling_condition,
                    "sampling_label": SAMPLING_LABELS[sampling_condition],
                    "lambda_hat": lambda_hat,
                    "kappa_hat": kappa_hat,
                    "mismatch_penalty": penalty,
                    "sampling_probability_min": float(probability.min()),
                    "sampling_probability_max": float(probability.max()),
                    "christoffel_zeta": zeta,
                }
            )
    return pd.DataFrame(rows)


def plot_lambda_matrix(
    frame: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot the absolute four-by-four cross/self compatibility matrix."""

    import matplotlib.pyplot as plt

    if frame.empty:
        raise ValueError("No cross/self compatibility values are available.")
    table = frame.pivot(
        index="reconstruction_condition",
        columns="sampling_condition",
        values="lambda_hat",
    ).reindex(index=RECOVERY_ORDER, columns=SAMPLING_ORDER)
    table.index = [NUMERATOR_TICK_LABELS[key] for key in RECOVERY_ORDER]
    table.columns = [SAMPLING_LABELS[key] for key in SAMPLING_ORDER]
    values = table.to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size != values.size:
        raise ValueError("The compatibility matrix must contain 16 finite positive values.")
    cmap = shared_analysis._make_linear_cmap(
        "cross_class_lambda",
        shared_analysis.SD15_LAMBDA_CMAP_COLORS,
    )
    with plt.rc_context(shared_analysis.SD15_PRESENTATION_RC):
        fig, axis = plt.subplots(figsize=(7.2, 6.8), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.02, hspace=0.02)
        shared_analysis._draw_publication_heatmap(
            axis,
            table,
            cmap=cmap,
            colorbar_label=r"$\widetilde{\Lambda}$",
            annotation_scale_power=shared_analysis._annotation_scale_power(values),
            tick_labelsize=14.0,
            show_ylabels=True,
            x_rotation=0.0,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        path = None
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
    return path


def performance_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed reconstructions with 95% Student-t intervals."""

    from scipy.stats import t as student_t

    records: list[dict[str, Any]] = []
    group_columns = ["sampling_condition", "reconstruction_condition"]
    for keys, group in rows.groupby(group_columns, sort=False):
        record: dict[str, Any] = dict(zip(group_columns, keys))
        record["trial_rate_count"] = int(len(group))
        for metric in PERFORMANCE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            count = int(values.size)
            mean = float(values.mean()) if count else np.nan
            std = float(values.std(ddof=1)) if count > 1 else np.nan
            sem = std / np.sqrt(count) if count > 1 else np.nan
            critical = float(student_t.ppf(0.975, count - 1)) if count > 1 else np.nan
            halfwidth = critical * sem if count > 1 else np.nan
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_ci_lower"] = mean - halfwidth if count > 1 else np.nan
            record[f"{metric}_ci_upper"] = mean + halfwidth if count > 1 else np.nan
        records.append(record)
    return pd.DataFrame(records)


def compatibility_performance_analysis(
    cross_bank: dict[str, dict[str, Any]],
    *,
    rows: pd.DataFrame | None = None,
    probability_regularization_zeta: float = ZETA,
) -> dict[str, pd.DataFrame]:
    """Compare compatibility predictions with completed reconstruction metrics."""

    from scipy.stats import spearmanr

    reconstruction_rows = load_ood_rows() if rows is None else rows.copy()
    compatibility = compatibility_table(
        cross_bank,
        probability_regularization_zeta=probability_regularization_zeta,
    )
    performance = performance_summary(reconstruction_rows)
    merged = compatibility.merge(
        performance,
        on=["sampling_condition", "reconstruction_condition"],
        how="inner",
        validate="one_to_one",
    )

    rank_records: list[dict[str, Any]] = []
    for recovery_condition, group in merged.groupby("reconstruction_condition", sort=False):
        for metric in PERFORMANCE_METRICS:
            ascending = metric in {"bp_best_loss", "lpips", "pixel_mae"}
            ranked = group.sort_values(f"{metric}_mean", ascending=ascending).reset_index(drop=True)
            predicted = group.sort_values("mismatch_penalty").reset_index(drop=True)
            rank_records.append(
                {
                    "reconstruction_condition": recovery_condition,
                    "metric": metric,
                    "predicted_best_sampling": str(predicted.iloc[0]["sampling_condition"]),
                    "observed_best_sampling": str(ranked.iloc[0]["sampling_condition"]),
                    "prediction_matches": bool(
                        predicted.iloc[0]["sampling_condition"] == ranked.iloc[0]["sampling_condition"]
                    ),
                    "predicted_best_penalty": float(predicted.iloc[0]["mismatch_penalty"]),
                    "observed_best_mean": float(ranked.iloc[0][f"{metric}_mean"]),
                }
            )

    correlation_records: list[dict[str, Any]] = []
    scopes = [("all", merged)] + [
        (str(recovery), group)
        for recovery, group in merged.groupby("reconstruction_condition", sort=False)
    ]
    for scope, group in scopes:
        for metric in PERFORMANCE_METRICS:
            # Spearman correlation operates on ranks, so use the compatibility
            # penalty directly rather than applying an unnecessary log transform
            x = np.asarray(group["mismatch_penalty"], dtype=float)
            y = np.asarray(group[f"{metric}_mean"], dtype=float)
            result = spearmanr(x, y)
            raw_rho = float(result.statistic)
            # Positive agreement means that a larger penalty predicts worse performance
            agreement_rho = -raw_rho if metric in {"psnr_db", "ssim"} else raw_rho
            correlation_records.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "count": int(len(group)),
                    "spearman_rho_raw": raw_rho,
                    "spearman_rho_agreement": agreement_rho,
                    "p_value": float(result.pvalue),
                }
            )
    return {
        "compatibility": compatibility,
        "performance": performance,
        "merged": merged,
        "best_match": pd.DataFrame(rank_records),
        "correlations": pd.DataFrame(correlation_records),
    }


def plot_cross_class_laws(
    bank: dict[str, dict[str, Any]],
    *,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot the three cross-class laws and sunset self-difference law."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterMathtext

    available = [entry["probability"] for entry in bank.values()]
    if not available:
        print("No completed ordered cross-class artifacts are available yet.")
        return None
    positive_minima = [float(values[values > 0.0].min()) for values in available]
    norm = LogNorm(
        vmin=min(positive_minima),
        vmax=max(float(values.max()) for values in available),
    )
    cmap = shared_analysis._make_linear_cmap(
        "cross_class_distribution",
        shared_analysis.SD15_DISTRIBUTION_CMAP_COLORS,
    )
    with plt.rc_context(shared_analysis.SD15_PRESENTATION_RC):
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.2), constrained_layout=True)
        axes = np.asarray(axes, dtype=object).reshape(-1)
        fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.02, hspace=0.02)
        image = None
        for axis, (key, _, _, _, _, law_label) in zip(axes, CROSS_LAWS):
            entry = bank.get(key)
            if entry is None:
                axis.text(0.5, 0.5, "In Progress", ha="center", va="center", transform=axis.transAxes)
                axis.set_facecolor("#f1f1f1")
            else:
                values = entry["probability"].reshape(entry["height"], entry["width"])
                image = axis.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
            axis.set_title(law_label, pad=10)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        if image is not None:
            colorbar = fig.colorbar(
                image,
                ax=list(axes),
                fraction=0.05,
                pad=0.02,
                shrink=0.96,
                aspect=28,
            )
            colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
            colorbar.ax.tick_params(labelsize=13, width=0.9, length=4)
            colorbar.outline.set_linewidth(0.8)
        path = None
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
    return path
