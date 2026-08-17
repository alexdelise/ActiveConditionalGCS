"""Analysis helpers for the sampling-side CFG K-tilde ablation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ZETA = 0.5
PROMPTS = (
    ("daytime_beach", "daytime beach", "Daytime Beach", r"$\widetilde{\mu}_{\mathrm{db}}$"),
    ("sunset_beach", "sunset beach", "Sunset Beach", r"$\widetilde{\mu}_{\mathrm{sb}}$"),
    ("cat", "cat", "Cat", r"$\widetilde{\mu}_{\mathrm{ca}}$"),
)
CFG_VALUES = (1.0, 3.0, 5.0, 7.5)


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository from a notebook or module path."""

    origin = Path(start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "ktilde" / "weighted" / "config_convergence.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the ActiveConditionalGCS repository.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
ANALYSIS_ROOT = PROJECT_ROOT / "analyze_results"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
import sd15_conditioning_experiment as shared_analysis
import sd15_cfg_ablation_analysis as cfg_style

NEW_CONFIG = PROJECT_ROOT / "ktilde" / "weighted" / "config_cfg_ablation_s10000.json"
REFERENCE_CONFIG = PROJECT_ROOT / "ktilde" / "weighted" / "config_convergence.json"
ARTIFACT_ROOT = PROJECT_ROOT / "ktilde" / "weighted"
FIGURE_ROOT = PROJECT_ROOT / "results" / "weighted" / "ktilde" / "figures"


def _catalog(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return {str(k): dict(v) for k, v in json.load(handle)["ktilde"].items()}


def _cfg_from_entry(entry: dict[str, Any]) -> float:
    return float(entry["guidance_scale"])


def _prompt_key(prompt: str) -> str:
    matches = [key for key, text, _, _ in PROMPTS if text == str(prompt)]
    if len(matches) != 1:
        raise KeyError(f"Unsupported CFG-ablation prompt {prompt!r}.")
    return matches[0]


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        k_tilde = np.asarray(payload["K_tilde"], dtype=np.float64).reshape(-1)
        probability = np.asarray(payload["prob"], dtype=np.float64).reshape(-1)
        metadata = json.loads(str(payload["meta"]))
    return k_tilde, probability, metadata


def load_bank(*, zeta: float = ZETA) -> tuple[dict[tuple[str, float], dict[str, Any]], pd.DataFrame]:
    """Load all available CFG laws and return an explicit completion table."""

    if not 0.0 <= float(zeta) < 1.0:
        raise ValueError("zeta must lie in [0, 1).")
    specifications: list[tuple[str, dict[str, Any], Path]] = []
    for name, entry in _catalog(NEW_CONFIG).items():
        specifications.append((name, entry, ARTIFACT_ROOT / f"{name}.npz"))
    for name, entry in _catalog(REFERENCE_CONFIG).items():
        if str(entry.get("prompt", "")):
            specifications.append((name, entry, ARTIFACT_ROOT / f"{name}.npz"))

    bank: dict[tuple[str, float], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for name, entry, path in specifications:
        prompt_key = _prompt_key(str(entry["prompt"]))
        cfg = _cfg_from_entry(entry)
        row = {
            "prompt": prompt_key,
            "prompt_text": str(entry["prompt"]),
            "sampling_cfg": cfg,
            "artifact_name": name,
            "artifact_path": str(path),
            "status": "missing",
        }
        if not path.is_file():
            rows.append(row)
            continue
        k_tilde, raw_probability, metadata = _load_npz(path)
        if not np.all(np.isfinite(k_tilde)) or np.any(k_tilde < 0.0):
            raise ValueError(f"Invalid K-tilde values in {path}.")
        if not np.all(np.isfinite(raw_probability)) or np.any(raw_probability < 0.0):
            raise ValueError(f"Invalid probability values in {path}.")
        if not np.isclose(float(raw_probability.sum()), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(f"Probability law in {path} does not sum to one.")
        n = int(raw_probability.size)
        probability = (1.0 - float(zeta)) * raw_probability + float(zeta) / n
        height = int(metadata["height"])
        width = int(metadata["width"])
        key = (prompt_key, cfg)
        bank[key] = {
            **row,
            "metadata": metadata,
            "K_tilde_raw": k_tilde,
            "K_tilde_unitary": k_tilde / float(height * width),
            "raw_probability": raw_probability,
            "probability": probability,
            "height": height,
            "width": width,
            "zeta": float(zeta),
        }
        row.update(
            {
                "status": "complete",
                "raw_probability_min": float(raw_probability.min()),
                "raw_probability_max": float(raw_probability.max()),
                "regularized_probability_min": float(probability.min()),
                "regularized_probability_max": float(probability.max()),
                "probability_floor": float(zeta) / n,
                "zero_count": int(np.count_nonzero(raw_probability == 0.0)),
            }
        )
        rows.append(row)
    completion = pd.DataFrame(rows).sort_values(["prompt", "sampling_cfg"]).reset_index(drop=True)
    return bank, completion


def relative_l2_table(bank: dict[tuple[str, float], dict[str, Any]]) -> pd.DataFrame:
    """Compare every available estimate with its prompt-matched CFG 7.5 estimate."""

    rows: list[dict[str, Any]] = []
    for prompt_key, _, prompt_label, _ in PROMPTS:
        reference = bank.get((prompt_key, 7.5))
        if reference is None:
            continue
        reference_values = reference["K_tilde_raw"]
        denominator = float(np.linalg.norm(reference_values))
        for cfg in CFG_VALUES:
            current = bank.get((prompt_key, cfg))
            if current is None:
                continue
            error = float(np.linalg.norm(current["K_tilde_raw"] - reference_values) / denominator)
            rows.append({"prompt": prompt_label, "sampling_cfg": cfg, "relative_l2_error": error})
    return pd.DataFrame(rows)


def lambda_table(bank: dict[tuple[str, float], dict[str, Any]]) -> pd.DataFrame:
    """Compute absolute unitary-scaled compatibility values within each prompt."""

    rows: list[dict[str, Any]] = []
    for prompt_key, _, prompt_label, _ in PROMPTS:
        for numerator_cfg in CFG_VALUES:
            numerator = bank.get((prompt_key, numerator_cfg))
            if numerator is None:
                continue
            for sampling_cfg in CFG_VALUES:
                sampling = bank.get((prompt_key, sampling_cfg))
                if sampling is None:
                    continue
                value = float(np.max(numerator["K_tilde_unitary"] / sampling["probability"]))
                rows.append(
                    {
                        "prompt": prompt_label,
                        "numerator_cfg": numerator_cfg,
                        "sampling_cfg": sampling_cfg,
                        "lambda_tilde": value,
                        "zeta": float(sampling["zeta"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_distribution_grid(
    bank: dict[tuple[str, float], dict[str, Any]],
    *,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot the three prompts by four sampling CFG values with shared scaling."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterMathtext

    available = [info["probability"] for info in bank.values()]
    if not available:
        raise ValueError("No CFG K-tilde artifacts are available.")
    positive_minima = [float(values[values > 0.0].min()) for values in available if np.any(values > 0.0)]
    if not positive_minima:
        raise ValueError("Sampling laws must contain positive probabilities.")
    vmin = min(positive_minima)
    vmax = max(float(values.max()) for values in available)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = shared_analysis._make_linear_cmap(
        "cfg_ktilde_distribution",
        shared_analysis.SD15_DISTRIBUTION_CMAP_COLORS,
    )
    cmap.set_bad(shared_analysis.SD15_DISTRIBUTION_CMAP_COLORS[0])
    cmap.set_under(shared_analysis.SD15_DISTRIBUTION_CMAP_COLORS[0])
    with plt.rc_context(cfg_style.PRESENTATION_RC):
        fig, axes = plt.subplots(3, 4, figsize=(16.5, 10.8), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.03, wspace=0.05, hspace=0.02)
        image = None
        for row, (prompt_key, _, _, sampling_label) in enumerate(PROMPTS):
            for col, cfg in enumerate(CFG_VALUES):
                ax = axes[row, col]
                info = bank.get((prompt_key, cfg))
                if info is None:
                    ax.text(0.5, 0.5, "In Progress", ha="center", va="center", transform=ax.transAxes)
                    ax.set_facecolor("#f1f1f1")
                else:
                    values = info["probability"].reshape(info["height"], info["width"])
                    image = ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(
                        rf"Sampling CFG $g={cfg:g}$",
                        fontsize=22,
                        pad=6,
                        **cfg_style.PANEL_TITLE_FONT,
                    )
                if col == 0:
                    ax.set_ylabel(
                        sampling_label,
                        fontsize=26,
                        rotation=0,
                        labelpad=34,
                        va="center",
                        **cfg_style.PANEL_TITLE_FONT,
                    )
                for spine in ax.spines.values():
                    spine.set_visible(False)
        if image is not None:
            colorbar = fig.colorbar(
                image,
                ax=axes.ravel().tolist(),
                fraction=0.025,
                pad=0.02,
                shrink=0.96,
                aspect=28,
            )
            colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
            colorbar.ax.tick_params(labelsize=cfg_style.PRESENTATION_RC["ytick.labelsize"], width=0.9, length=4)
            colorbar.outline.set_linewidth(0.8)
            colorbar.set_label(r"Sampling Probability $\widetilde{\mu}_{c,g}$")
        path = None
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        # Render while the inline backend is configured for SVG, then close the
        # Agg-backed live figure before Jupyter's post-cell redraw hook runs
        if show:
            plt.show()
        plt.close(fig)
    return path


def plot_lambda_panels(
    frame: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot one absolute compatibility matrix per prompt."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterMathtext

    if frame.empty:
        raise ValueError("No compatibility values are available.")
    values = frame["lambda_tilde"].to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("Compatibility values must contain finite positive entries.")
    vmin = float(np.min(positive))
    vmax = float(np.max(positive))
    if np.isclose(vmin, vmax):
        vmax = vmin * 1.01
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = shared_analysis._make_linear_cmap(
        "cfg_ktilde_lambda",
        shared_analysis.SD15_LAMBDA_CMAP_COLORS,
    )
    with plt.rc_context(cfg_style.PRESENTATION_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.02)
        image = None
        for ax, (_, _, prompt_label, _) in zip(axes, PROMPTS):
            subset = frame[frame["prompt"] == prompt_label]
            table = subset.pivot(index="numerator_cfg", columns="sampling_cfg", values="lambda_tilde")
            table = table.reindex(index=CFG_VALUES, columns=CFG_VALUES)
            plot_values = table.to_numpy(dtype=float)
            plot_values = np.where(np.isfinite(plot_values), plot_values, np.nan)
            image = ax.imshow(plot_values, cmap=cmap, norm=norm)
            ax.set_title(prompt_label)
            ax.set_xticks(range(4), [f"{cfg:g}" for cfg in CFG_VALUES])
            ax.set_yticks(range(4), [f"{cfg:g}" for cfg in CFG_VALUES])
            ax.set_xlabel("Sampling CFG")
            ax.set_ylabel("Numerator CFG")
            for row in range(4):
                for col in range(4):
                    value = table.iloc[row, col]
                    if np.isinf(value):
                        annotation = r"$\infty$"
                    elif np.isfinite(value):
                        annotation = shared_analysis._format_heatmap_number(float(value))
                    else:
                        continue
                    if np.isfinite(value) or np.isinf(value):
                        ax.text(
                            col,
                            row,
                            annotation,
                            ha="center",
                            va="center",
                            fontsize=13,
                            fontweight="semibold",
                            color="#111827",
                            bbox={
                                "boxstyle": "round,pad=0.17",
                                "facecolor": (1.0, 1.0, 1.0, 0.8),
                                "edgecolor": "none",
                            },
                        )
        if image is not None:
            colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, aspect=28)
            colorbar.set_label(r"$\widetilde{\Lambda}$")
            colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
            colorbar.ax.tick_params(labelsize=cfg_style.PRESENTATION_RC["ytick.labelsize"], width=0.9, length=4)
            colorbar.outline.set_linewidth(0.8)
        path = None
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        # Match the established reconstruction notebooks: show the SVG inline
        # and close the figure so Agg never requests dvipng after the cell ends
        if show:
            plt.show()
        plt.close(fig)
    return path
