"""Generate the manifests and output-free notebooks for this isolated study."""

from __future__ import annotations

import json
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "weighted" / "ablation"
NOTEBOOK_ROOT = PROJECT_ROOT / "analyze_results" / "weighted" / "ablation"

SCENARIOS = {
    "prompt_matched": {
        "title": "Prompt-Matched In-Range",
        "dataset": "sunset_beach_signal_sd15_512x512",
    },
    "prompt_mismatched": {
        "title": "Prompt-Mismatched In-Range",
        "dataset": "sunset_sandy_coast_signal_sd15_512x512",
    },
    "out_of_range": {
        "title": "Out-of-Range",
        "dataset": "out_of_range_512x512",
    },
}

LAWS = {
    "k0": {
        "condition": "k0",
        "label": r"$\widetilde{\mu}_{c_{\mathrm{uc}}}$",
        "prefix": "sample_k0_unconditioned",
        "artifact": "Ktilde_SD15__fft__k0_512x512_S10000_ns20",
    },
    "k1": {
        "condition": "k1_daytime_beach",
        "label": r"$\widetilde{\mu}_{c_{\mathrm{db}}}$",
        "prefix": "sample_k1_daytime_beach",
        "artifact": "Ktilde_SD15__fft__k1daytimebeach_512x512_S10000_ns20",
    },
    "k2": {
        "condition": "k2_sunset_beach",
        "label": r"$\widetilde{\mu}_{c_{\mathrm{sb}}}$",
        "prefix": "sample_k2_sunset_beach",
        "artifact": "Ktilde_SD15__fft__k2sunsetbeach_512x512_S10000_ns20",
    },
    "k4": {
        "condition": "k4_cat",
        "label": r"$\widetilde{\mu}_{c_{\mathrm{ca}}}$",
        "prefix": "sample_k4_cat",
        "artifact": "Ktilde_SD15__fft__k4cat_512x512_S10000_ns20",
    },
}

LINES = {
    "unconditioned": {"prompt": "", "cfg": 1.0, "label": "Unconditioned", "rank": 0},
    "cfg1": {"prompt": "sunset beach", "cfg": 1.0, "label": "CFG 1", "rank": 1},
    "cfg1p5": {"prompt": "sunset beach", "cfg": 1.5, "label": "CFG 1.5", "rank": 2},
    "cfg3": {"prompt": "sunset beach", "cfg": 3.0, "label": "CFG 3", "rank": 3},
    "cfg5": {"prompt": "sunset beach", "cfg": 5.0, "label": "CFG 5", "rank": 4},
    "cfg7p5": {"prompt": "sunset beach", "cfg": 7.5, "label": "CFG 7.5", "rank": 5},
}


def write_json(path: Path, payload: object) -> None:
    """Write one deterministically formatted JSON asset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def base_config(dataset: str) -> dict[str, object]:
    """Return the validated weighted/unitary ablation configuration."""

    return {
        "dataset": {"name": dataset},
        "gen_recon": {"eta": 0.0, "guidance_scale": 1.0, "num_steps": 20},
        "image": {"height": 512, "width": 512},
        "ktilde": {"name": LAWS["k0"]["artifact"]},
        "optim": {
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "adam_stepsize": 100.0,
            "early_stop_loss": 1e-8,
            "log_every": 10,
            "lr_scale": 0.1,
            "nb_epochs": 2000,
        },
        "output": {
            "plot_images": False,
            "save_images": True,
            "save_json": True,
            "save_mat": False,
            "save_npz": True,
        },
        "reconstruction": {"prompt": "", "prompts": None},
        "reconstruction_solver": {
            "backproj_init_strength": 1.0,
            "checkpoint_denoiser": True,
            "early_stop_min_rel_improvement": 0.0,
            "early_stop_patience": 0,
            "grad_clip": 0.0,
            "init_from_meas_backproj": True,
            "latent_l2_penalty": 0.0,
            "learning_rate": 0.1,
            "log_every": 5,
            "loss_reduction": "measurement_sum_channel_mean",
            "lr_decay_start_iteration": 400,
            "lr_min_factor": 0.01,
            "lr_schedule": "cosine_decay",
            "lr_warmup_iterations": 0,
            "normalize_grad": False,
            "outer_iterations": 2000,
            "sigma_y": 0.0,
            "trace_save_every": 5,
        },
        "repro": {"cudnn_benchmark": False, "deterministic": False, "seed": 12345},
        "runtime": {
            "attention_slicing": "auto",
            "gradient_checkpointing": True,
            "torch_dtype": "float16",
        },
        "sampling": {
            "fft_normalization": "ortho",
            "methods_enabled": {"cs": True, "mcs": False, "inverse_square": False},
            "probability_regularization_zeta": 0.5,
            "weighted_ls": True,
        },
        "sweep": {
            "repeats_per_setting": 2,
            "sampling_perc_list": [0.01, 0.02, 0.03, 0.04, 0.05],
            "save_per_run_artifacts": True,
        },
    }


def suite_config(scenario: str, law: str) -> dict[str, object]:
    """Return all six recovery controls for one fixed CFG-7.5 sampling law."""

    info = LAWS[law]
    cases = []
    for line, line_info in LINES.items():
        suffix = "recover_unprompted" if line == "unconditioned" else f"recover_prompt_sunset_beach_{line}"
        cases.append(
            {
                "name": f"{info['prefix']}__{suffix}",
                "description": (
                    f"{SCENARIOS[scenario]['title']} recovery with fixed {law} CFG-7.5 "
                    f"Christoffel sampling and the {line_info['label']} recovery control."
                ),
                "sampling_condition": info["condition"],
                "sampling_label": info["label"],
                "sampling_rank": list(LAWS).index(law),
                "reconstruction_condition": line,
                "reconstruction_label": line_info["label"],
                "recon_rank": line_info["rank"],
                "overrides": {
                    "gen_recon": {"guidance_scale": line_info["cfg"]},
                    "ktilde": {"name": info["artifact"]},
                    "reconstruction": {"prompt": line_info["prompt"], "prompts": None},
                },
            }
        )
    return {
        "base_config": f"configs/weighted/ablation/{scenario}/base.json",
        "tag": law,
        "use_dc_presets": False,
        "cases": cases,
    }


def notebook_cell(cell_type: str, source: str) -> dict[str, object]:
    """Build one clean notebook cell."""

    cell: dict[str, object] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def notebook(scenario: str) -> dict[str, object]:
    """Return one output-free notebook using the established TeX plotting helper."""

    title = SCENARIOS[scenario]["title"]
    cells = [
        notebook_cell(
            "markdown",
            f"# SD1.5 {title} Weighted Recovery-CFG Ablation\n\n"
            "This two-trial study fixes each Christoffel sampling distribution at its "
            "CFG-7.5 S10000 estimate and varies only the recovery conditioning. All runs "
            "use the weighted unitary Fourier operator, $\\zeta=1/2$, and the 2,000-step "
            "main weighted learning-rate schedule.\n",
        ),
        notebook_cell(
            "code",
            "from pathlib import Path\n"
            "import importlib.util\n"
            "import os\n\n"
            "cwd = Path.cwd().resolve()\n"
            "candidates = [cwd / 'analyze_results/weighted/ablation']\n"
            "candidates.extend(parent / 'analyze_results/weighted/ablation' for parent in (cwd, *cwd.parents))\n"
            "STUDY_ROOT = next((path for path in candidates if (path / 'analysis.py').is_file()), None)\n"
            "if STUDY_ROOT is None:\n"
            "    raise FileNotFoundError('Could not locate analyze_results/weighted/ablation/analysis.py')\n"
            "spec = importlib.util.spec_from_file_location('weighted_cfg_ablation_analysis', STUDY_ROOT / 'analysis.py')\n"
            "diagnostic = importlib.util.module_from_spec(spec)\n"
            "assert spec.loader is not None\n"
            "spec.loader.exec_module(diagnostic)\n\n"
            f"SCENARIO = '{scenario}'\n"
            "OUTPUT_DIR = diagnostic.RESULT_ROOT / SCENARIO / 'figures'\n"
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT_DIR / '.matplotlib'))\n"
            "Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)\n"
            "diagnostic.PROJECT_ROOT, STUDY_ROOT, OUTPUT_DIR\n",
        ),
        notebook_cell(
            "markdown",
            "## Load Results\n\n"
            "LPIPS is stored during new reconstructions and is also filled incrementally "
            "for any compatible legacy artifact. The completion table contains every "
            "expected sampling-law, recovery-CFG, ratio, and trial cell.\n",
        ),
        notebook_cell(
            "code",
            "LPIPS_TABLE = diagnostic.ensure_lpips(SCENARIO, device='cpu')\n"
            "ROWS = diagnostic.load_rows(SCENARIO)\n"
            "COMPLETION = diagnostic.completion_table(ROWS)\n"
            "print(f'Loaded {len(ROWS)} / {diagnostic.EXPECTED_ROWS_PER_SCENARIO} reconstructions')\n"
            "display(diagnostic.count_table(ROWS))\n"
            "display(COMPLETION.groupby(['sampling_law', 'recovery_line'], as_index=False)[['observed', 'expected', 'left']].sum())\n"
            "display(ROWS[['distribution_key', 'line_condition', 'samp_perc', 'repeat_id', 'psnr_db', 'ssim', 'lpips', 'pixel_mae']].head())\n",
        ),
        notebook_cell(
            "markdown",
            "## Metric Curves\n\n"
            "The TeX-styled figure matches the established ablation layout. Solid lines "
            "show two-trial arithmetic means and shading shows one sample standard "
            "deviation; with only two trials, this is more transparent than presenting a "
            "nominal confidence interval based on one degree of freedom.\n",
        ),
        notebook_cell(
            "code",
            "METRIC_OUTPUTS = diagnostic.plot_metric_curves(ROWS, output_dir=OUTPUT_DIR, show=True)\n"
            "METRIC_OUTPUTS\n",
        ),
        notebook_cell(
            "markdown",
            "## Reconstruction Panel\n\n"
            "Each row is one fixed CFG-7.5 sampling law. Within each recovery column, the "
            "displayed trial minimizes LPIPS, with PSNR used to break ties. Change "
            "`PANEL_RATIO` to any value in the 1--5% grid.\n",
        ),
        notebook_cell(
            "code",
            "PANEL_RATIO = 0.01\n"
            "PANEL_OUTPUTS = diagnostic.plot_reconstruction_panel(\n"
            "    ROWS,\n"
            "    sampling_ratio=PANEL_RATIO,\n"
            "    output_dir=OUTPUT_DIR,\n"
            "    show=True,\n"
            ")\n"
            "PANEL_OUTPUTS\n",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "testWH", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Refresh every generated study asset."""

    for scenario, scenario_info in SCENARIOS.items():
        config_root = CONFIG_ROOT / scenario
        write_json(config_root / "base.json", base_config(str(scenario_info["dataset"])))
        for law in LAWS:
            write_json(config_root / f"{law}_suite.json", suite_config(scenario, law))
        write_json(
            NOTEBOOK_ROOT / f"{scenario}_cfg_ablation.ipynb",
            notebook(scenario),
        )


if __name__ == "__main__":
    main()
