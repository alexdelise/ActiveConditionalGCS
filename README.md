# Active Learning for Conditional Generative Compressed Sensing Code Repository

## Setup

Use Python 3.11 or 3.12 with a CUDA-capable PyTorch install.
`requirements.txt` lists the direct Python dependencies needed for the code and analysis notebooks.
```bash
cd ActiveConditionalGCS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHON_BIN="$(pwd)/.venv/bin/python"
```

All shell launchers honor `PYTHON_BIN`; set it to the interpreter for the
environment that has these requirements installed.

The SD1.5 weights are not vendored in this folder. The first dataset, K-tilde,
or reconstruction command downloads `stable-diffusion-v1-5/stable-diffusion-v1-5`
through Hugging Face Diffusers, so the machine must be allowed to download the
model or have it available in the Hugging Face cache. The model page lists the
Stable Diffusion v1.5 weights under the CreativeML Open RAIL-M license; see
`THIRD_PARTY_NOTICES.md` for the model notice. If needed, authenticate with
`huggingface-cli login` and set `HF_HOME` or `HF_HUB_CACHE` to a persistent
cache directory.

The analysis helpers use Matplotlib with `text.usetex=True` for paper-style
figures. Install a TeX distribution before running the notebooks that export
PDF figures.

## File Tree

```text
📦 ActiveConditionalGCS/
│
├── 📄 README.md                         ← This guide: setup, naming, output layout, and complete run commands
├── 📄 requirements.txt                  ← Direct Python dependencies for the paper runs
├── 📄 THIRD_PARTY_NOTICES.md            ← Third-party attribution notes
├── 📄 build_dataset.py                  ← Builds/validates dataset artifacts from `datasets/config.json`
├── 📄 build_ktilde.py                   ← Builds/validates Christoffel/K-tilde artifacts
├── 📄 run_ktilde_convergence.py         ← Reruns Algorithm 1 and records relative L2 convergence
├── 📄 run_conditioning_regression.py    ← Suite runner used by all experiment scripts
├── 📄 run_cs.py                         ← Single-config Christoffel/K-tilde sampler runner
├── 📄 run_mcs.py                        ← Optional MCS baseline runner, not used by paper scripts
├── 📄 run_all.py                        ← Single-config runner for enabled samplers
├── 📄 run_suite.py                      ← Generic suite runner retained for compatibility
│
├── 📁 src/                              ← Core reconstruction package
│   ├── config.py                        ← Config dataclasses, loaders, and sampler aliases
│   ├── constants.py                     ← Device/constants shared by the codebase
│   ├── datasets.py                      ← Dataset artifact loading helpers
│   ├── diffusion.py                     ← SD1.5 pipeline loading, prompt encoding, denoising, VAE helpers
│   ├── fft.py                           ← Partial Fourier operators
│   ├── ktilde.py                        ← K-tilde probability loading/build helpers
│   ├── metrics.py                       ← PSNR/SSIM/image formatting utilities
│   ├── reconstruction.py                ← Diffusion backpropagation reconstruction loop
│   ├── runner.py                        ← Sweep execution, artifact writing, resume/skip behavior
│   ├── sampling.py                      ← Christoffel/K-tilde sampling masks plus measurement operator
│   ├── utils.py                         ← JSON, reproducibility, environment, and CUDA cleanup helpers
│   └── README.md
│
├── 📁 configs/                          ← Base configs and suite manifests
│   ├── prompt_matched/sunset/
│   ├── prompt_mismatched/sunset/
│   ├── out_of_range/sunset/
│   ├── ablation/<experiment>/sunset/
│   ├── *_suite.json                     ← Per-prior suite manifests launched by scripts
│   └── README.md
│
├── 📁 datasets/                         ← Dataset artifacts and index files
│   ├── config.json
│   ├── sunset_beach_signal_sd15_512x512/
│   ├── sunset_sandy_coast_signal_sd15_512x512/
│   ├── out_of_range_512x512/
│   └── README.md
│
├── 📁 ktilde/                           ← Precomputed Christoffel/K-tilde artifacts
│   ├── config.json
│   ├── config_cfg_ablation.json
│   ├── config_convergence.json
│   ├── Ktilde_SD15__fft__*.npz
│   └── README.md
│
├── 📁 scripts/                          ← Parallel launch scripts and aggregate wrappers
│   ├── prompt_matched/sunset/            ← Full, first4, last3, and aggregate prompt-matched runs
│   ├── prompt_mismatched/sunset/         ← Full, first4, last3, and aggregate mismatched runs
│   ├── out_of_range/sunset/              ← Full, first4, last3, and aggregate OOD runs
│   ├── ablation/<experiment>/sunset/     ← CFG-ablation launch scripts
│   ├── ablation/ktilde/sunset/           ← Compatibility aliases for older CFG K-tilde commands
│   ├── ktilde/main/                       ← Four primary S500 paper-prior builders
│   ├── ktilde/cfg_ablation/               ← Nine CFG-ablation S500 prior builders
│   ├── ktilde/convergence/                ← Independent S10000 build and convergence launchers
│   └── README.md
│
├── 📁 analyze_results/                  ← Analysis helpers and notebooks for paper figures
│   ├── sd15_conditioning_experiment.py
│   ├── sd15_cfg_ablation_analysis.py
│   ├── sd15_*_results*.ipynb
│   ├── sd15_*_cfg_ablation.ipynb
│   └── README.md
│
├── 📁 readmepics/                       ← Expected-output figures copied from completed original runs
└── 📁 results/                          ← Ignored local outputs, organized by experiment family
    ├── prompt_matched/sunset/
    ├── prompt_matched_old/sunset/         ← Archived pre-fix prompt-matched outputs
    ├── prompt_mismatched/sunset/
    ├── out_of_range/sunset/
    ├── ablation/<experiment>/sunset/
    └── analysis/<experiment>/sunset/
```

## Naming

| Family Path | Manuscript Name | Dataset | Meaning |
| --- | --- | --- | --- |
| `prompt_matched` | Prompt matched in range | `sunset_beach_signal_sd15_512x512` | Signal and recovery prompt match in range. |
| `prompt_mismatched` | Prompt mismatched in range | `sunset_sandy_coast_signal_sd15_512x512` | Signal is in range but the recovery prompt is mismatched. |
| `out_of_range` | Out of range | `out_of_range_512x512` | External out-of-range sunset image. |

| Prior | Meaning |
| --- | --- |
| `k0_unconditioned` | Unconditioned K-tilde prior. |
| `k1_daytime_beach` | Daytime beach prompt prior. |
| `k2_sunset_beach` | Sunset beach prompt prior. |
| `k4_cat` | Cat prompt prior. |

## K-Tilde Rebuild Commands

Every checked-in k-tilde used by the paper has a corresponding independent
launcher under `scripts/ktilde/`. All launchers reuse an existing artifact
after validating its metadata; pass `--force` to regenerate it.

Rebuild or validate the four primary CFG 7.5, S500 paper priors:

```bash
bash scripts/ktilde/main/build_all.sh
```

Rebuild or validate all nine conditioned CFG 1, 3, and 5 S500 ablation priors:

```bash
bash scripts/ktilde/cfg_ablation/build_all.sh
```

Rebuild or validate all 13 checked-in paper priors sequentially:

```bash
bash scripts/ktilde/build_all_paper.sh
```

Each aggregate is composed of per-artifact launchers in the same directory, so
the jobs can also be scheduled independently. The historical
`scripts/ablation/ktilde/sunset/build_cfg*_conditioned_prompts.sh` commands
remain as compatibility aliases for the canonical CFG aggregate launchers.

## K-Tilde Convergence Commands

The convergence experiment has two phases for each of the four paper prompts.
First, build the final 10,000-iteration reference under `ktilde/`. Then rerun
the same deterministic estimator and record
`||K_tilde_iteration - K_tilde_final||_2 / ||K_tilde_final||_2`. The rerun
saves only the compact error trace and metadata under
`results/analysis/ktilde_convergence/`; it does not save intermediate k-tildes.
Each measurement job also streams the measured relative L2 error every 10
iterations.

Each command below is an independent job. A measurement job requires only its
matching reference build to have completed.

Build the four final references:

```bash
bash scripts/ktilde/convergence/build_k0_unconditioned.sh
bash scripts/ktilde/convergence/build_k1_daytime_beach.sh
bash scripts/ktilde/convergence/build_k2_sunset_beach.sh
bash scripts/ktilde/convergence/build_k4_cat.sh
```

Measure the four convergence traces:

```bash
bash scripts/ktilde/convergence/measure_k0_unconditioned.sh
bash scripts/ktilde/convergence/measure_k1_daytime_beach.sh
bash scripts/ktilde/convergence/measure_k2_sunset_beach.sh
bash scripts/ktilde/convergence/measure_k4_cat.sh
```

Sequential aggregate wrappers are also available:

```bash
bash scripts/ktilde/convergence/build_all.sh
bash scripts/ktilde/convergence/measure_all.sh
```

Pass `--force` to a build or measurement launcher to replace its existing
artifact. The convergence runner validates that the final rerun matches its
saved S10000 reference.

After all four measurement jobs finish, run the early convergence section in
`analyze_results/sd15_ktilde_lambda_comparison.ipynb`. It displays the 2x2
panel and saves the panel plus four individual PDFs under
`results/analysis/ktilde_convergence/figures/`. The panel is saved as
`sd15_ktilde_convergence_grid.pdf`; the standalone files follow
`sd15_ktilde_convergence_<role>.pdf`.

## Main Experiment Commands
We provide commands to perform runs split by sampling distribution for parallel efficiency. Each program launched by one of the command belows requires about 5GB VRAM, so all four runs, corresponding to each sampling distribution, can fit on a 24GB VRAM GPU like the NVIDIA RTX A5000 GPUs we use in the paper. 

### Prompt Matched In Range

This experiment uses the same seven sampling percentages and optimizer settings
as the other main experiments. The complete sweep can be split into the first
four and last three sampling percentages for parallel scheduling.

Split by sampling distribution for the complete first4+last3 sweep:

```bash
bash scripts/prompt_matched/sunset/first4_last3_sample_k0_unconditioned.sh
bash scripts/prompt_matched/sunset/first4_last3_sample_k1_daytime_beach.sh
bash scripts/prompt_matched/sunset/first4_last3_sample_k2_sunset_beach.sh
bash scripts/prompt_matched/sunset/first4_last3_sample_k4_cat.sh
```

Run all four distributions for the complete first4+last3 sweep (not recommended):

```bash
bash scripts/prompt_matched/sunset/first4_last3_all.sh
```

Schedule the first four sampling rates separately:

```bash
bash scripts/prompt_matched/sunset/first4_sample_k0_unconditioned.sh
bash scripts/prompt_matched/sunset/first4_sample_k1_daytime_beach.sh
bash scripts/prompt_matched/sunset/first4_sample_k2_sunset_beach.sh
bash scripts/prompt_matched/sunset/first4_sample_k4_cat.sh
```

Schedule the last three sampling rates separately:

```bash
bash scripts/prompt_matched/sunset/last3_sample_k0_unconditioned.sh
bash scripts/prompt_matched/sunset/last3_sample_k1_daytime_beach.sh
bash scripts/prompt_matched/sunset/last3_sample_k2_sunset_beach.sh
bash scripts/prompt_matched/sunset/last3_sample_k4_cat.sh
```

Run an unsplit seven-rate sweep by sampling distribution:

```bash
bash scripts/prompt_matched/sunset/sample_k0_unconditioned.sh
bash scripts/prompt_matched/sunset/sample_k1_daytime_beach.sh
bash scripts/prompt_matched/sunset/sample_k2_sunset_beach.sh
bash scripts/prompt_matched/sunset/sample_k4_cat.sh
```

Run all four unsplit distributions sequentially (not recommended):

```bash
bash scripts/prompt_matched/sunset/all.sh
```

### Prompt Mismatched In Range
For this experiment we recover over seven different sampling percentages. This can take a long time, thus we split up the optimization across the first four and last three sampling percentages to run in parallel. The first set of commands below aggregates optimization over all seven sampling percentages at once, whereas the ones below it, which we used, split up the jobs. 


Split by sampling distribution for the complete first4+last3 sweep:

```bash
bash scripts/prompt_mismatched/sunset/first4_last3_sample_k0_unconditioned.sh
bash scripts/prompt_mismatched/sunset/first4_last3_sample_k1_daytime_beach.sh
bash scripts/prompt_mismatched/sunset/first4_last3_sample_k2_sunset_beach.sh
bash scripts/prompt_mismatched/sunset/first4_last3_sample_k4_cat.sh
```

Run all four distributions for the complete first4+last3 sweep (not recommended):

```bash
bash scripts/prompt_mismatched/sunset/first4_last3_all.sh
```

Schedule the first four sampling rates separately:

```bash
bash scripts/prompt_mismatched/sunset/first4_sample_k0_unconditioned.sh
bash scripts/prompt_mismatched/sunset/first4_sample_k1_daytime_beach.sh
bash scripts/prompt_mismatched/sunset/first4_sample_k2_sunset_beach.sh
bash scripts/prompt_mismatched/sunset/first4_sample_k4_cat.sh
```

Schedule the last three sampling rates separately:

```bash
bash scripts/prompt_mismatched/sunset/last3_sample_k0_unconditioned.sh
bash scripts/prompt_mismatched/sunset/last3_sample_k1_daytime_beach.sh
bash scripts/prompt_mismatched/sunset/last3_sample_k2_sunset_beach.sh
bash scripts/prompt_mismatched/sunset/last3_sample_k4_cat.sh
```

This command runs all optimization for this experiment sequentially (not recommended):

```bash
bash scripts/prompt_mismatched/sunset/all.sh
```

### Out Of Range
For this experiment we recover over seven different sampling percentages. This can take a long time, thus we split up the optimization across the first four and last three sampling percentages to run in parallel. The first set of commands below aggregates optimization over all seven sampling percentages at once, whereas the ones below it, which we used, split up the jobs. 

Split by sampling distribution for the complete first4+last3 sweep:

```bash
bash scripts/out_of_range/sunset/first4_last3_sample_k0_unconditioned.sh
bash scripts/out_of_range/sunset/first4_last3_sample_k1_daytime_beach.sh
bash scripts/out_of_range/sunset/first4_last3_sample_k2_sunset_beach.sh
bash scripts/out_of_range/sunset/first4_last3_sample_k4_cat.sh
```

Run all four distributions for the complete first4+last3 sweep:

```bash
bash scripts/out_of_range/sunset/first4_last3_all.sh
```

Schedule the first four sampling rates separately:

```bash
bash scripts/out_of_range/sunset/first4_sample_k0_unconditioned.sh
bash scripts/out_of_range/sunset/first4_sample_k1_daytime_beach.sh
bash scripts/out_of_range/sunset/first4_sample_k2_sunset_beach.sh
bash scripts/out_of_range/sunset/first4_sample_k4_cat.sh
```

Schedule the last three sampling rates separately:

```bash
bash scripts/out_of_range/sunset/last3_sample_k0_unconditioned.sh
bash scripts/out_of_range/sunset/last3_sample_k1_daytime_beach.sh
bash scripts/out_of_range/sunset/last3_sample_k2_sunset_beach.sh
bash scripts/out_of_range/sunset/last3_sample_k4_cat.sh
```

This command runs all optimization for this experiment sequentially (not recommended):

```bash
bash scripts/out_of_range/sunset/all.sh
```

## CFG Ablation Commands

### Prompt Matched In Range CFG Ablation

The updated prompt-matched ablation uses the fixed seven-rate grid and copies
its unconditioned and CFG 7.5 references from the prompt-matched first4/last3
main-experiment outputs.

Split by sampling distribution:

```bash
bash scripts/ablation/prompt_matched/sunset/sample_k0_unconditioned.sh
bash scripts/ablation/prompt_matched/sunset/sample_k1_daytime_beach.sh
bash scripts/ablation/prompt_matched/sunset/sample_k2_sunset_beach.sh
bash scripts/ablation/prompt_matched/sunset/sample_k4_cat.sh
```

Run all four distributions sequentially (not recommended):

```bash
bash scripts/ablation/prompt_matched/sunset/all.sh
```

### Prompt Mismatched In Range CFG Ablation

Split by sampling distribution:

```bash
bash scripts/ablation/prompt_mismatched/sunset/sample_k0_unconditioned.sh
bash scripts/ablation/prompt_mismatched/sunset/sample_k1_daytime_beach.sh
bash scripts/ablation/prompt_mismatched/sunset/sample_k2_sunset_beach.sh
bash scripts/ablation/prompt_mismatched/sunset/sample_k4_cat.sh
```

Run all four distributions sequentially (not recommended):

```bash
bash scripts/ablation/prompt_mismatched/sunset/all.sh
```

### Out Of Range CFG Ablation

Split by sampling distribution:

```bash
bash scripts/ablation/out_of_range/sunset/sample_k0_unconditioned.sh
bash scripts/ablation/out_of_range/sunset/sample_k1_daytime_beach.sh
bash scripts/ablation/out_of_range/sunset/sample_k2_sunset_beach.sh
bash scripts/ablation/out_of_range/sunset/sample_k4_cat.sh
```

Run all four distributions sequentially (not recommended):

```bash
bash scripts/ablation/out_of_range/sunset/all.sh
```

## Output Layout

Runs write to:

```text
results/<experiment_family>/sunset/[first4_|last3_]sample_<prior>/diffusion_backprop/<case>/cs/item_000/samp_<rate>/rep_<id>/
results/ablation/<experiment_family>/sunset/sample_<prior>/diffusion_backprop/<case>/cs/item_000/samp_<rate>/rep_<id>/
```

Each leaf run stores `run_config.json`, `dataset_item.json`, `run_data.npz`, `run_summary.txt`, `recon_cs.png`, `zero_filled_ifft.png`, and `z_rec.pt`.

Each suite also writes `suite_manifest.json`, `suite_results.json`, and compact `results_cs.csv/.npz` tables. These raw and restartable experiment artifacts stay under `results/<experiment_family>/...` or `results/ablation/<experiment_family>/...`.

`results/analysis/...` is the separate derived-output layer. Analysis notebooks
write paper figures and compact summaries there so they do not mix with the
raw per-run artifacts. Analysis-only measurements, such as the compact k-tilde
convergence traces, also write directly into this layer.

The pre-fix prompt-matched main and CFG-ablation artifacts are archived under
the corresponding `prompt_matched_old` result and analysis directories.

K-tilde convergence runs write one compact `.convergence.npz` trace and one
matching `.convergence.meta.json` file per prompt under
`results/analysis/ktilde_convergence/`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{delise2026active,
  title={Active Learning for Conditional Generative Compressed Sensing},
  author={DeLise, Alexander and Dexter, Nick},
  journal={arXiv preprint arXiv:2605.05435},
  year={2026}
}
```
