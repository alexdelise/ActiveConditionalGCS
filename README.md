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
PDF/PNG figures.

## File Tree

```text
📦 ActiveConditionalGCS/
│
├── 📄 README.md                         ← This guide: setup, naming, output layout, and complete run commands
├── 📄 requirements.txt                  ← Direct Python dependencies for the paper runs
├── 📄 THIRD_PARTY_NOTICES.md            ← Third-party attribution notes
├── 📄 build_dataset.py                  ← Builds/validates dataset artifacts from `datasets/config.json`
├── 📄 build_ktilde.py                   ← Builds/validates Christoffel/K-tilde artifacts
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
│   ├── prompt_matched_in_range_base.json
│   ├── prompt_mismatched_in_range_base.json
│   ├── prompt_mismatched_in_range_first4_base.json
│   ├── prompt_mismatched_in_range_last3_base.json
│   ├── out_of_range_base.json
│   ├── out_of_range_first4_base.json
│   ├── out_of_range_last3_base.json
│   ├── *_cfg_ablation_base.json
│   ├── *_sample_*_suite.json            ← Per-prior suite manifests launched by scripts
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
│   ├── Ktilde_SD15__fft__*.npz
│   └── README.md
│
├── 📁 scripts/                          ← Parallel launch scripts and aggregate wrappers
│   ├── run_<experiment>_sample_<prior>.sh              ← Per-prior paper run
│   ├── run_<experiment>_all.sh                         ← All priors for one experiment
│   ├── run_<experiment>_cfg_ablation_sample_<prior>.sh ← Per-prior CFG ablation
│   ├── run_<experiment>_cfg_ablation_all.sh            ← All priors for one CFG ablation
│   ├── run_<experiment>_first4_* / run_<experiment>_last3_* ← Split long sweeps
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
└── 📁 results/                          ← Empty output folder for regenerated code-only submission runs
```

## Naming

| Name | Dataset | Meaning |
| --- | --- | --- |
| `prompt_matched_in_range` | `sunset_beach_signal_sd15_512x512` | Signal and recovery prompt match in range. |
| `prompt_mismatched_in_range` | `sunset_sandy_coast_signal_sd15_512x512` | Signal is in range but the recovery prompt is mismatched. |
| `out_of_range` | `out_of_range_512x512` | External out-of-range sunset image. |

| Prior | Meaning |
| --- | --- |
| `k0_unconditioned` | Unconditioned K-tilde prior. |
| `k1_daytime_beach` | Daytime beach prompt prior. |
| `k2_sunset_beach` | Sunset beach prompt prior. |
| `k4_cat` | Cat prompt prior. |


## Main Experiment Commands
We provide commands to perform runs split by sampling distribution for parallel efficiency. Each program launched by one of the command belows requires about 5GB VRAM, so all four runs, corresponding to each sampling distribution, can fit on a 24GB VRAM GPU like the NVIDIA RTX A5000 GPUs we use in the paper. 

### Prompt Matched In Range

Split by sampling distribution:

```bash
bash scripts/run_prompt_matched_in_range_sample_k0_unconditioned.sh
bash scripts/run_prompt_matched_in_range_sample_k1_daytime_beach.sh
bash scripts/run_prompt_matched_in_range_sample_k2_sunset_beach.sh
bash scripts/run_prompt_matched_in_range_sample_k4_cat.sh
```

Run all four distributions sequentially (not recommended):

```bash
bash scripts/run_prompt_matched_in_range_all.sh
```

### Prompt Mismatched In Range
For this experiment we recover over seven different sampling percentages. This can take a long time, thus we split up the optimization across the first four and last three sampling percentages to run in parallel. The first set of commands below aggregates optimization over all seven sampling percentages at once, whereas the ones below it, which we used, split up the jobs. 


Split by sampling distribution for the complete first4+last3 sweep:

```bash
bash scripts/run_prompt_mismatched_in_range_first4_last3_sample_k0_unconditioned.sh
bash scripts/run_prompt_mismatched_in_range_first4_last3_sample_k1_daytime_beach.sh
bash scripts/run_prompt_mismatched_in_range_first4_last3_sample_k2_sunset_beach.sh
bash scripts/run_prompt_mismatched_in_range_first4_last3_sample_k4_cat.sh
```

Run all four distributions for the complete first4+last3 sweep (not recommended):

```bash
bash scripts/run_prompt_mismatched_in_range_first4_last3_all.sh
```

Schedule the first four sampling rates separately:

```bash
bash scripts/run_prompt_mismatched_in_range_first4_sample_k0_unconditioned.sh
bash scripts/run_prompt_mismatched_in_range_first4_sample_k1_daytime_beach.sh
bash scripts/run_prompt_mismatched_in_range_first4_sample_k2_sunset_beach.sh
bash scripts/run_prompt_mismatched_in_range_first4_sample_k4_cat.sh
```

Schedule the last three sampling rates separately:

```bash
bash scripts/run_prompt_mismatched_in_range_last3_sample_k0_unconditioned.sh
bash scripts/run_prompt_mismatched_in_range_last3_sample_k1_daytime_beach.sh
bash scripts/run_prompt_mismatched_in_range_last3_sample_k2_sunset_beach.sh
bash scripts/run_prompt_mismatched_in_range_last3_sample_k4_cat.sh
```

This command runs all optimization for this experiment sequentially (not recommended):

```bash
bash scripts/run_prompt_mismatched_in_range_all.sh
```

### Out Of Range
For this experiment we recover over seven different sampling percentages. This can take a long time, thus we split up the optimization across the first four and last three sampling percentages to run in parallel. The first set of commands below aggregates optimization over all seven sampling percentages at once, whereas the ones below it, which we used, split up the jobs. 

Split by sampling distribution for the complete first4+last3 sweep:

```bash
bash scripts/run_out_of_range_first4_last3_sample_k0_unconditioned.sh
bash scripts/run_out_of_range_first4_last3_sample_k1_daytime_beach.sh
bash scripts/run_out_of_range_first4_last3_sample_k2_sunset_beach.sh
bash scripts/run_out_of_range_first4_last3_sample_k4_cat.sh
```

Run all four distributions for the complete first4+last3 sweep:

```bash
bash scripts/run_out_of_range_first4_last3_all.sh
```

Schedule the first four sampling rates separately:

```bash
bash scripts/run_out_of_range_first4_sample_k0_unconditioned.sh
bash scripts/run_out_of_range_first4_sample_k1_daytime_beach.sh
bash scripts/run_out_of_range_first4_sample_k2_sunset_beach.sh
bash scripts/run_out_of_range_first4_sample_k4_cat.sh
```

Schedule the last three sampling rates separately:

```bash
bash scripts/run_out_of_range_last3_sample_k0_unconditioned.sh
bash scripts/run_out_of_range_last3_sample_k1_daytime_beach.sh
bash scripts/run_out_of_range_last3_sample_k2_sunset_beach.sh
bash scripts/run_out_of_range_last3_sample_k4_cat.sh
```

This command runs all optimization for this experiment sequentially (not recommended):

```bash
bash scripts/run_out_of_range_all.sh
```

## CFG Ablation Commands

### Prompt Matched In Range CFG Ablation

Split by sampling distribution:

```bash
bash scripts/run_prompt_matched_in_range_cfg_ablation_sample_k0_unconditioned.sh
bash scripts/run_prompt_matched_in_range_cfg_ablation_sample_k1_daytime_beach.sh
bash scripts/run_prompt_matched_in_range_cfg_ablation_sample_k2_sunset_beach.sh
bash scripts/run_prompt_matched_in_range_cfg_ablation_sample_k4_cat.sh
```

Run all four distributions sequentially (not recommended):

```bash
bash scripts/run_prompt_matched_in_range_cfg_ablation_all.sh
```

### Prompt Mismatched In Range CFG Ablation

Split by sampling distribution:

```bash
bash scripts/run_prompt_mismatched_in_range_cfg_ablation_sample_k0_unconditioned.sh
bash scripts/run_prompt_mismatched_in_range_cfg_ablation_sample_k1_daytime_beach.sh
bash scripts/run_prompt_mismatched_in_range_cfg_ablation_sample_k2_sunset_beach.sh
bash scripts/run_prompt_mismatched_in_range_cfg_ablation_sample_k4_cat.sh
```

Run all four distributions sequentially (not recommended):

```bash
bash scripts/run_prompt_mismatched_in_range_cfg_ablation_all.sh
```

### Out Of Range CFG Ablation

Split by sampling distribution:

```bash
bash scripts/run_out_of_range_cfg_ablation_sample_k0_unconditioned.sh
bash scripts/run_out_of_range_cfg_ablation_sample_k1_daytime_beach.sh
bash scripts/run_out_of_range_cfg_ablation_sample_k2_sunset_beach.sh
bash scripts/run_out_of_range_cfg_ablation_sample_k4_cat.sh
```

Run all four distributions sequentially (not recommended):

```bash
bash scripts/run_out_of_range_cfg_ablation_all.sh
```

## Output Layout

Runs write to:

```text
results/<experiment>[_split][_cfg_ablation]_sample_<prior>/diffusion_backprop/<case>/cs/item_000/samp_<rate>/rep_<id>/
```

Each leaf run stores `run_config.json`, `dataset_item.json`, `run_data.npz`, `run_summary.txt`, `recon_cs.png`, `zero_filled_ifft.png`, and `z_rec.pt`.

Each suite also writes `suite_manifest.json`, `suite_results.json`, and compact `results_cs.csv/.npz` tables.

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
