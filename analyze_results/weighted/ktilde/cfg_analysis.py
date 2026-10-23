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
    ("daytime_beach", "daytime beach", "Daytime Beach", r"$\widetilde{\mu}_{\mathrm{c}_{\mathrm{db}}}$"),
    ("sunset_beach", "sunset beach", "Sunset Beach", r"$\widetilde{\mu}_{\mathrm{c}_{\mathrm{sb}}}$"),
    ("cat", "cat", "Cat", r"$\widetilde{\mu}_{\mathrm{c}_{\mathrm{ca}}}$"),
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
    with plt.rc_context(shared_analysis.KTILDE_PRESENTATION_RC):
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
                        rf"CFG {cfg:g}",
                        fontsize=shared_analysis.KTILDE_PRESENTATION_RC["axes.titlesize"],
                        pad=6,
                        **cfg_style.PANEL_TITLE_FONT,
                    )
                if col == 0:
                    ax.set_ylabel(
                        sampling_label,
                        fontsize=shared_analysis.KTILDE_PRESENTATION_RC["axes.labelsize"],
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
            shared_analysis.style_ktilde_colorbar(colorbar)
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
    """Plot compatibility with sampling CFG in rows and recovery CFG in columns."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
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
    with plt.rc_context(shared_analysis.KTILDE_PRESENTATION_RC):
        fig = plt.figure(figsize=(26.0, 22.0))
        # Center the lower panel between two closely spaced upper panels
        panel_width = 0.36
        panel_height = panel_width * 26.0 / 22.0
        positions = ((0.065, 0.555), (0.585, 0.555), (0.325, 0.04))
        axes = [fig.add_axes((left, bottom, panel_width, panel_height))
                for left, bottom in positions]
        image = None
        for ax, (left, bottom), (prompt_key, _, prompt_label, _) in zip(axes, positions, PROMPTS):
            subset = frame[frame["prompt"] == prompt_label]
            table = subset.pivot(index="sampling_cfg", columns="numerator_cfg", values="lambda_tilde")
            table = table.reindex(index=CFG_VALUES, columns=CFG_VALUES)
            plot_values = table.to_numpy(dtype=float)
            plot_values = np.where(np.isfinite(plot_values), plot_values, np.nan)
            image = ax.imshow(plot_values, cmap=cmap, norm=norm, interpolation="nearest")
            prompt_symbol = {"daytime_beach": "db", "sunset_beach": "sb", "cat": "ca"}[prompt_key]
            prompt = rf"c_{{\mathrm{{{prompt_symbol}}}}}"
            label_size = 32.0
            cfg_labels = [rf"$\mathrm{{CFG}}\ {cfg:g}$" for cfg in CFG_VALUES]
            ax.set_xticks(range(4), cfg_labels)
            ax.set_yticks(range(4), cfg_labels)
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(
                rf"$\widetilde K(\mathbb F_{{{prompt}}}-\mathbb F_{{{prompt}}})$",
                fontsize=label_size, labelpad=26,
            )
            ax.set_ylabel(
                rf"$\widetilde\mu_{{{prompt}}}$",
                fontsize=label_size, labelpad=30, rotation=0, va="center",
            )
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=9)
            ax.tick_params(axis="x", labelsize=label_size,
                           labelrotation=0)
            ax.tick_params(axis="y", labelsize=label_size,
                           labelrotation=90, pad=23)
            for tick in ax.get_yticklabels():
                tick.set_ha("center")
                tick.set_va("center")
            ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.6)
            ax.tick_params(which="minor", bottom=False, left=False)
            for row in range(4):
                for col in range(4):
                    value = table.iloc[row, col]
                    if np.isinf(value):
                        annotation = r"$\infty$"
                    elif np.isfinite(value):
                        annotation = f"{value:.0f}"
                    else:
                        continue
                    if np.isfinite(value) or np.isinf(value):
                        ax.text(
                            col,
                            row,
                            annotation,
                            ha="center",
                            va="center",
                            fontsize=26.0,
                            fontweight="semibold",
                            color="#111827",
                            bbox={
                                "boxstyle": "round,pad=0.17",
                                "facecolor": (1.0, 1.0, 1.0, 0.8),
                                "edgecolor": "none",
                            },
                        )
            for index in range(4):
                ax.add_patch(Rectangle((index - 0.5, index - 0.5), 1, 1,
                                       fill=False, edgecolor="#111827", linewidth=2.1))
            # Match the shorter, centered color bars of the other heatmaps
            colorbar_height = 0.92 * panel_height
            colorbar_bottom = bottom + (panel_height - colorbar_height) / 2
            colorbar_axis = fig.add_axes((left + panel_width + 0.012, colorbar_bottom, 0.0156, colorbar_height))
            colorbar = fig.colorbar(image, cax=colorbar_axis)
            recovery_prompt = rf"{prompt}^{{(\mathrm{{CFG}}_r)}}"
            sampling_prompt = rf"{prompt}^{{(\mathrm{{CFG}}_s)}}"
            colorbar.set_label(
                rf"$\widetilde\Lambda\left({recovery_prompt},{recovery_prompt},{sampling_prompt}\right)$",
                labelpad=12,
            )
            colorbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
            shared_analysis.style_ktilde_colorbar(colorbar)
            colorbar.ax.yaxis.label.set_fontsize(label_size)
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
