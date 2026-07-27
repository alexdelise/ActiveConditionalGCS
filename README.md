# Active Learning for Conditional Generative Compressed Sensing

## Setup

Use Python 3.11 or 3.12 with a CUDA-capable PyTorch installation. Direct
dependencies are listed in [requirements.txt](requirements.txt).

```bash
cd ActiveConditionalGCS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHON_BIN="$(pwd)/.venv/bin/python"
```

All launchers use the active environment's `python` by default and honor
`PYTHON_BIN`. Stable Diffusion 1.5 is downloaded through Hugging Face
Diffusers or loaded from the local Hugging Face cache. Paper-style PDF figures
require a TeX installation.

## Repository Layout

- [configs/unweighted/](configs/unweighted/) contains backward-normalized,
  unweighted experiment configurations.
- [configs/weighted/](configs/weighted/) contains unitary, weighted
  least-squares experiment configurations.
- [scripts/unweighted/](scripts/unweighted/) and
  [scripts/weighted/](scripts/weighted/) contain the corresponding launchers.
- [results/unweighted/](results/unweighted/) and
  [results/weighted/](results/weighted/) contain raw runs and their respective
  [unweighted figures](results/unweighted/figures/) and
  [weighted figures](results/weighted/figures/).
- [analyze_results/unweighted/](analyze_results/unweighted/) and
  [analyze_results/weighted/](analyze_results/weighted/) contain analysis
  notebooks.
- [ktilde/unweighted/](ktilde/unweighted/) contains the S500 artifacts, while
  [ktilde/weighted/](ktilde/weighted/) contains the S10000 reference and
  convergence-study artifacts.

## Experiment Families

| Family | Dataset | Meaning |
| --- | --- | --- |
| `prompt_matched` | `sunset_beach_signal_sd15_512x512` | In-range signal and matched recovery prompt. |
| `prompt_mismatched` | `sunset_sandy_coast_signal_sd15_512x512` | In-range signal with mismatched recovery prompt. |
| `out_of_range` | `out_of_range_512x512` | External out-of-range sunset image. |

The four K-tilde sampling priors are `k0_unconditioned`,
`k1_daytime_beach`, `k2_sunset_beach`, and `k4_cat`.

## Unweighted Experiments

Use [scripts/unweighted/run_split.sh](scripts/unweighted/run_split.sh):

```bash
./scripts/unweighted/run_split.sh \
  <main|ablation> \
  <prompt_matched|prompt_mismatched|out_of_range> \
  <first4|last3> \
  <unprompted|daytime_beach|sunset_beach|cat>
```

`first4` runs sampling ratios
`0.00015625, 0.0003125, 0.000625, 0.00125`; `last3` runs
`0.0025, 0.005, 0.01`.

```bash
./scripts/unweighted/run_split.sh main prompt_matched first4 sunset_beach
./scripts/unweighted/run_split.sh main prompt_matched last3 sunset_beach
./scripts/unweighted/run_split.sh ablation prompt_matched first4 sunset_beach
./scripts/unweighted/run_split.sh ablation prompt_matched last3 sunset_beach
```

Use [scripts/unweighted/run_suite.sh](scripts/unweighted/run_suite.sh) to run
both splits sequentially:

```bash
./scripts/unweighted/run_suite.sh main prompt_matched sunset_beach
```

Validate every split manifest without loading SD1.5:

```bash
python scripts/unweighted/validate_suite.py
```

The S500 K-tilde builders are
[scripts/unweighted/ktilde/main/build_all.sh](scripts/unweighted/ktilde/main/build_all.sh),
[scripts/unweighted/ktilde/cfg_ablation/build_all.sh](scripts/unweighted/ktilde/cfg_ablation/build_all.sh),
and
[scripts/unweighted/ktilde/build_all_paper.sh](scripts/unweighted/ktilde/build_all_paper.sh).

## Weighted Experiments

Use [scripts/weighted/run_split.sh](scripts/weighted/run_split.sh):

```bash
./scripts/weighted/run_split.sh \
  <main|ablation> \
  <prompt_matched|prompt_mismatched|out_of_range> \
  <first3|last2> \
  <sampling-prior>
```

Uniform MCS, inverse-square, and VDHH launchers are under
[scripts/weighted/baselines/](scripts/weighted/baselines/).

### K-Tilde Convergence

Run one trial/prior pair with
[scripts/weighted/ktilde_convergence/run_trial.sh](scripts/weighted/ktilde_convergence/run_trial.sh):

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k0 1
```

Validate the complete job grid with
[scripts/weighted/ktilde_convergence/list_all.sh](scripts/weighted/ktilde_convergence/list_all.sh):

```bash
./scripts/weighted/ktilde_convergence/list_all.sh
```

Trial artifacts are stored under
[ktilde/weighted/convergence_trials/](ktilde/weighted/convergence_trials/),
scalar traces under
[results/weighted/ktilde_convergence/traces/](results/weighted/ktilde_convergence/traces/),
and convergence figures under
[results/weighted/figures/ktilde_convergence/](results/weighted/figures/ktilde_convergence/).

Reference builders and single-reference convergence measurements are under
[scripts/weighted/ktilde_convergence/reference/](scripts/weighted/ktilde_convergence/reference/).

## Output Layout

Raw unweighted runs are stored below
[results/unweighted/](results/unweighted/), and raw weighted runs are stored
below [results/weighted/](results/weighted/). Every leaf contains its resolved
configuration, measurements, reconstruction, scalar metrics, and recovered
latent.

Analysis outputs are stored beside their experiment family:

- [results/unweighted/figures/](results/unweighted/figures/)
- [results/weighted/figures/](results/weighted/figures/)

## Analysis

Shared Python analysis helpers are in
[analyze_results/](analyze_results/). Notebook groups are:

- [analyze_results/unweighted/main/](analyze_results/unweighted/main/)
- [analyze_results/unweighted/ablation/](analyze_results/unweighted/ablation/)
- [analyze_results/weighted/main/](analyze_results/weighted/main/)
- [analyze_results/weighted/ablation/](analyze_results/weighted/ablation/)

## Citation

```bibtex
@article{delise2026active,
  title={Active Learning for Conditional Generative Compressed Sensing},
  author={DeLise, Alexander and Dexter, Nick},
  journal={arXiv preprint arXiv:2605.05435},
  year={2026}
}
```
