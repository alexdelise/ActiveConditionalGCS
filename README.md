# Active Learning for Conditional Generative Compressed Sensing

This repository contains the code and experiment artifacts for *Active
Learning for Conditional Generative Compressed Sensing*. The paper studies
image recovery from subsampled Fourier measurements using a
prompt-conditioned generative model. It separates the prompt used to design
the sampling distribution, denoted by $c_s$, from the prompt that defines the
recovery class, denoted by $c_r$. When the target lies in a
prompt-conditioned model class, its signal prompt is denoted by $c_*$.

The implementation provides:

- empirical Christoffel function estimation for prompt-conditioned Stable
  Diffusion 1.5 model classes;
- prompt-conditioned Fourier sampling distributions;
- latent optimization for conditional generative compressed sensing;
- the reported unweighted reconstruction experiments;
- a theory-aligned replication using a unitary, inverse-probability-weighted
  subsampled Fourier operator;
- uniform Monte Carlo sampling (MCS), pure inverse-square, and variable-density
  half-half (VDHH) baselines;
- convergence diagnostics for the empirical Christoffel estimates and the
  associated compatibility statistic; and
- analysis notebooks for the main experiments, CFG ablations, reconstruction
  panels, sampling distributions, and prompt compatibility figures.

## Setup

Use Python 3.11 or 3.12 with a CUDA-capable PyTorch installation. The direct
Python dependencies are pinned in [requirements.txt](requirements.txt).

```bash
cd ActiveConditionalGCS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHON_BIN="$(pwd)/.venv/bin/python"
```

The launchers use the active environment's `python` by default and honor
`PYTHON_BIN` when it is set.

Stable Diffusion 1.5 weights are loaded through Hugging Face Diffusers. A
machine running dataset generation, empirical Christoffel estimation, or
reconstruction must either have access to
`stable-diffusion-v1-5/stable-diffusion-v1-5` or have the model in its local
Hugging Face cache. If authentication is required, use `huggingface-cli login`
and place `HF_HOME` or `HF_HUB_CACHE` on persistent storage. Licensing and
attribution information is provided in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The analysis notebooks use Matplotlib with TeX rendering for paper-style PDF
figures. A TeX installation is therefore required when exporting the figures.

## Repository Organization

<pre>
📦 ActiveConditionalGCS/
│
├── 📄 <a href="README.md">README.md</a>                         ← Setup, terminology, repository layout, and experiment guide
├── 📄 <a href="requirements.txt">requirements.txt</a>                  ← Python dependencies
├── 📄 <a href="THIRD_PARTY_NOTICES.md">THIRD_PARTY_NOTICES.md</a>           ← Third-party attribution and model notices
├── 📄 <a href="activeconditionalgcs.pdf">activeconditionalgcs.pdf</a>          ← Current paper PDF
├── 📄 <a href="build_dataset.py">build_dataset.py</a>                  ← Build or validate the fixed datasets
├── 📄 <a href="build_ktilde.py">build_ktilde.py</a>                   ← Estimate empirical Christoffel functions
├── 📄 <a href="run_ktilde_convergence.py">run_ktilde_convergence.py</a>         ← Run one convergence trial
├── 📄 <a href="run_conditioning_regression.py">run_conditioning_regression.py</a>      ← Run reconstruction suites
├── 📄 <a href="run_cs.py">run_cs.py</a>                          ← Run one Christoffel-sampling configuration
├── 📄 <a href="run_mcs.py">run_mcs.py</a>                         ← Run one uniform-MCS configuration
├── 📄 <a href="run_all.py">run_all.py</a>                         ← Run all samplers enabled by one configuration
├── 📄 <a href="run_suite.py">run_suite.py</a>                       ← Run a generic collection of reconstruction cases
│
├── 📁 <a href="src/">src/</a>                              ← Core reconstruction package
│   ├── 📄 <a href="src/config.py">config.py</a>                     ← Configuration models and validation
│   ├── 📄 <a href="src/datasets.py">datasets.py</a>                   ← Dataset construction and loading
│   ├── 📄 <a href="src/diffusion.py">diffusion.py</a>                  ← SD1.5, DDIM, prompt, and VAE helpers
│   ├── 📄 <a href="src/fft.py">fft.py</a>                        ← Partial Fourier operators and adjoints
│   ├── 📄 <a href="src/ktilde.py">ktilde.py</a>                     ← Empirical Christoffel artifacts and sampling laws
│   ├── 📄 <a href="src/metrics.py">metrics.py</a>                    ← Reconstruction metrics
│   ├── 📄 <a href="src/reconstruction.py">reconstruction.py</a>             ← Latent reconstruction loop
│   ├── 📄 <a href="src/runner.py">runner.py</a>                     ← Sweep execution and artifact writing
│   ├── 📄 <a href="src/sampling.py">sampling.py</a>                   ← Sampling masks and weighted operators
│   └── 📄 <a href="src/utils.py">utils.py</a>                      ← Reproducibility and artifact-path helpers
│
├── 📁 <a href="configs/">configs/</a>                          ← Reconstruction configurations and suite manifests
│   ├── 📁 <a href="configs/unweighted/">unweighted/</a>                   ← Reported experiments and CFG ablations
│   └── 📁 <a href="configs/weighted/">weighted/</a>                     ← Weighted replication, baselines, and ablations
│
├── 📁 <a href="datasets/">datasets/</a>                         ← Fixed image datasets and generation metadata
│   ├── 📁 <a href="datasets/sunset_beach_signal_sd15_512x512/">sunset_beach_signal_sd15_512x512/</a>
│   ├── 📁 <a href="datasets/sunset_sandy_coast_signal_sd15_512x512/">sunset_sandy_coast_signal_sd15_512x512/</a>
│   └── 📁 <a href="datasets/out_of_range_512x512/">out_of_range_512x512/</a>
│
├── 📁 <a href="ktilde/">ktilde/</a>                          ← Empirical Christoffel artifacts
│   ├── 📁 <a href="ktilde/unweighted/">unweighted/</a>                   ← S500 estimates used by the reported experiments
│   └── 📁 <a href="ktilde/weighted/">weighted/</a>                     ← S10000 references and convergence trials
│       ├── 📁 <a href="ktilde/weighted/reference/">reference/</a>
│       └── 📁 <a href="ktilde/weighted/convergence_trials/">convergence_trials/</a>
│
├── 📁 <a href="scripts/">scripts/</a>                          ← Experiment launchers and validation utilities
│   ├── 📁 <a href="scripts/unweighted/">unweighted/</a>                   ← Unweighted main experiments and ablations
│   │   └── 📁 <a href="scripts/unweighted/ktilde/">ktilde/</a>                   ← S500 empirical Christoffel builders
│   └── 📁 <a href="scripts/weighted/">weighted/</a>                     ← Weighted main experiments and ablations
│       ├── 📁 <a href="scripts/weighted/baselines/">baselines/</a>                ← MCS, inverse-square, and VDHH launchers
│       └── 📁 <a href="scripts/weighted/ktilde_convergence/">ktilde_convergence/</a>      ← Five-trial convergence launchers
│
├── 📁 <a href="analyze_results/">analyze_results/</a>                  ← Shared analysis helpers and paper notebooks
│   ├── 📁 <a href="analyze_results/unweighted/">unweighted/</a>
│   │   ├── 📁 <a href="analyze_results/unweighted/main/">main/</a>
│   │   └── 📁 <a href="analyze_results/unweighted/ablation/">ablation/</a>
│   └── 📁 <a href="analyze_results/weighted/">weighted/</a>
│       ├── 📁 <a href="analyze_results/weighted/main/">main/</a>
│       └── 📁 <a href="analyze_results/weighted/ablation/">ablation/</a>
│
└── 📁 <a href="results/">results/</a>                          ← Local reconstruction outputs and generated figures
    ├── 📁 <a href="results/unweighted/">unweighted/</a>
    │   └── 📁 <a href="results/unweighted/figures/">figures/</a>
    └── 📁 <a href="results/weighted/">weighted/</a>
        ├── 📁 <a href="results/weighted/figures/">figures/</a>
        ├── 📁 <a href="results/weighted/ktilde_convergence/">ktilde_convergence/</a>
        └── 📁 <a href="results/weighted/metrics/">metrics/</a>
</pre>

### Entry Points

| Location | Purpose |
| --- | --- |
| [build_dataset.py](build_dataset.py) | Build or validate the fixed image datasets described by [datasets/config.json](datasets/config.json). |
| [build_ktilde.py](build_ktilde.py) | Estimate an empirical Christoffel function $\widetilde{K}$ from generated secants and save its induced sampling law. |
| [run_ktilde_convergence.py](run_ktilde_convergence.py) | Run one independent S10000 convergence trial and record scalar diagnostics every ten iterations. |
| [run_conditioning_regression.py](run_conditioning_regression.py) | Run a suite that crosses one sampling law with the configured recovery prompts, sampling ratios, and repeats. |
| [run_cs.py](run_cs.py) | Run Christoffel sampling for a single reconstruction configuration. |
| [run_mcs.py](run_mcs.py) | Run uniform Monte Carlo sampling for a single reconstruction configuration. |
| [run_all.py](run_all.py) | Run every sampling method enabled in one configuration. |
| [run_suite.py](run_suite.py) | Run a generic collection of named reconstruction cases. |

### Core Package

The implementation shared by the entry points is in [src/](src/).

| Location | Purpose |
| --- | --- |
| [src/config.py](src/config.py) | Configuration dataclasses, validation, sampling-method names, and aliases. |
| [src/constants.py](src/constants.py) | Shared device and numerical constants. |
| [src/datasets.py](src/datasets.py) | Dataset construction, indexing, and artifact loading. |
| [src/diffusion.py](src/diffusion.py) | SD1.5 loading, prompt encoding, DDIM denoising, and VAE helpers. |
| [src/fft.py](src/fft.py) | Unnormalized and unitary partial Fourier transforms and their adjoints. |
| [src/ktilde.py](src/ktilde.py) | Empirical Christoffel estimation, probability construction, metadata validation, and artifact loading. |
| [src/metrics.py](src/metrics.py) | Reconstruction metrics and image-formatting helpers. |
| [src/reconstruction.py](src/reconstruction.py) | Latent optimization, measurement losses, early stopping, and optimization traces. |
| [src/runner.py](src/runner.py) | Sampling-ratio sweeps, deterministic repeats, resume behavior, and result serialization. |
| [src/sampling.py](src/sampling.py) | Christoffel, MCS, inverse-square, and VDHH mask construction and weighted measurement operators. |
| [src/utils.py](src/utils.py) | Reproducibility, environment metadata, JSON output, hashing, CUDA cleanup, and artifact-path resolution. |

Further implementation notes are available in [src/README.md](src/README.md).

### Configurations, Artifacts, Launchers, and Analysis

| Location | Contents |
| --- | --- |
| [configs/unweighted/](configs/unweighted/) | Configurations and suite manifests for the reported unweighted experiments and CFG ablations. |
| [configs/weighted/](configs/weighted/) | Configurations for the theory-aligned weighted experiments, weighted CFG ablations, and classical sampling baselines. |
| [datasets/](datasets/) | Dataset indices, fixed ground-truth images, generation metadata, and the dataset build configuration. |
| [ktilde/unweighted/](ktilde/unweighted/) | S500 empirical Christoffel artifacts used by the unweighted experiments, including the CFG-ablation estimates. |
| [ktilde/weighted/](ktilde/weighted/) | S10000 reference estimates, the convergence-trial manifest, and completed convergence artifacts. |
| [scripts/unweighted/](scripts/unweighted/) | Launchers for unweighted main experiments, CFG ablations, and S500 empirical Christoffel estimation. |
| [scripts/weighted/](scripts/weighted/) | Launchers for weighted main experiments, CFG ablations, sampling baselines, and the five-trial convergence study. |
| [analyze_results/](analyze_results/) | Shared loading, validation, metric, and plotting functions. |
| [analyze_results/unweighted/](analyze_results/unweighted/) | Main and CFG-ablation notebooks for the unweighted experiment suite. |
| [analyze_results/weighted/](analyze_results/weighted/) | Main, CFG-ablation, and convergence notebooks for the weighted experiment suite. |
| [results/unweighted/](results/unweighted/) | Unweighted reconstruction artifacts and [unweighted figures](results/unweighted/figures/). |
| [results/weighted/](results/weighted/) | Weighted reconstruction artifacts, convergence traces, derived metrics, and [weighted figures](results/weighted/figures/). |

The README files inside [configs/](configs/), [datasets/](datasets/),
[ktilde/](ktilde/), [scripts/](scripts/), and
[analyze_results/](analyze_results/) provide more focused descriptions of
their respective contents.

## Prompt Roles and Experiment Names

The prompt family used for sampling and recovery is:

| Code name | Paper notation | Conditioning |
| --- | --- | --- |
| `k0_unconditioned` | $c_{\mathrm{uc}}$ | Unconditioned generation or recovery. |
| `k1_daytime_beach` | $c_{\mathrm{db}}$ | `daytime beach` |
| `k2_sunset_beach` | $c_{\mathrm{sb}}$ | `sunset beach` |
| `k4_cat` | $c_{\mathrm{ca}}$ | `cat` |

In a reconstruction suite, the selected K-tilde artifact determines the
sampling prompt $c_s$ and hence the empirical sampling law. Each suite then
evaluates the four recovery prompts $c_r$ above. The three experiment families
describe the relationship between the target signal and these
prompt-conditioned model classes:

| Family | Dataset | Interpretation |
| --- | --- | --- |
| `prompt_matched` | `sunset_beach_signal_sd15_512x512` | In-range prompt-matched task. The target was generated with $c_* = c_{\mathrm{sb}}$, which is included in the sampling and recovery prompt family. |
| `prompt_mismatched` | `sunset_sandy_coast_signal_sd15_512x512` | In-range prompt-mismatched task. The target was generated with `sunset over a sandy coast`, which is not one of the four sampling and recovery prompts. |
| `out_of_range` | `out_of_range_512x512` | Out-of-range task using an external sunset image rather than an image generated by the fixed SD1.5 model. |

## Empirical Christoffel Sampling

For a sampling prompt $c_s$, [build_ktilde.py](build_ktilde.py) generates
independent pairs of images from the prompt-conditioned model class, forms
normalized secants, and records the largest observed Fourier-coordinate
energy. The resulting array $\widetilde{K}_{c_s}$ is an empirical
approximation to the generalized Christoffel function of the self-difference
class $\mathbb{F}_{c_s}-\mathbb{F}_{c_s}$. Its normalized empirical
Christoffel sampling law is

$$
\widetilde{\mu}_{c_s}(i)
=
\frac{
  \widetilde{K}_{c_s}(i)
}{
  \displaystyle\sum_{\ell=1}^{n}\widetilde{K}_{c_s}(\ell)
},
\qquad i\in D.
$$

The unweighted experiment suite uses the checked-in S500 estimates in
[ktilde/unweighted/](ktilde/unweighted/). The theory-aligned weighted suite
uses the S10000 references in
[ktilde/weighted/reference/](ktilde/weighted/reference/) and applies the
$\zeta=1/2$ uniform mixture

$$
\widetilde{\mu}_{c_s,1/2}(i)
=
\frac{1}{2}\widetilde{\mu}_{c_s}(i)
+
\frac{1}{2n}.
$$

Regularization is applied after normalizing $\widetilde{K}$, so the saved
S10000 estimates can be reused without regenerating secants.

## Reconstruction Operators

The unweighted suite reproduces the measurement and objective conventions
used for the originally reported reconstruction results. It uses the standard
unnormalized forward FFT (`backward` normalization in PyTorch) and an
unweighted least-squares residual.

The weighted suite uses the unitary two-dimensional discrete Fourier transform
$\mathbf{F}_{\mathrm{u}}$ independently in each color channel. For sampled frequencies
$\Omega=(I_1,\ldots,I_m)$, its block-RGB operator is

$$
\left(\mathbf{A}^{\mathrm{w}}_{\Omega,c_s}\mathbf{f}\right)_{j,q}
=
\frac{
  \left(\mathbf{F}_{\mathrm{u}}\mathbf{f}_q\right)_{I_j}
}{
  \sqrt{m\,\widetilde{\mu}_{c_s,1/2}(I_j)}
},
\qquad
j=1,\ldots,m,\quad q=1,2,3.
$$

The corresponding latent reconstruction minimizes the weighted measurement
residual over the recovery class $\mathbb{F}_{c_r}$. The implementation stores
both raw and weighted residuals and does not clip inverse-probability weights;
the $\zeta=1/2$ mixture supplies the probability floor $1/(2n)$.

Both suites force the DC coefficient into the mask and sample the remaining
frequencies without replacement. For the weighted Christoffel experiments,
the non-DC draw uses the renormalized DC-excluded proposal, while the recovery
weights use the original regularized law $\widetilde{\mu}_{c_s,1/2}$.

## Experiment Suites

### Unweighted Experiments

The unweighted main experiments use seven sampling ratios, split into
`first4` and `last3` for scheduling:

- `first4`: `0.00015625`, `0.0003125`, `0.000625`, `0.00125`;
- `last3`: `0.0025`, `0.005`, `0.01`.

The suite crosses the four empirical Christoffel sampling laws with four
recovery prompts in each of the three recovery scenarios. Separate CFG
ablation manifests evaluate how classifier-free guidance changes the effective
recovery class.

### Weighted Experiments

The theory-aligned weighted main experiments use five sampling ratios:

`0.00125`, `0.0025`, `0.005`, `0.01`, and `0.025`.

Each cell uses five independently seeded masks. The main Christoffel grid
contains four sampling prompts, four recovery prompts, three scenarios, five
sampling ratios, and five repeats, for 1,200 reconstructions.

The weighted baseline configurations provide:

- uniform Monte Carlo sampling, denoted MCS;
- pure inverse-square variable-density sampling; and
- VDHH, which deterministically includes a centered low-frequency disk and
  samples the remaining coefficients uniformly outside it.

All three baselines use the same unitary weighted operator, with weights
defined by the sampling law or inclusion probabilities appropriate to that
design.

### Empirical Christoffel Convergence Study

The convergence study keeps each checked-in S10000 estimate fixed as a
reference and generates five additional independent S10000 estimates per
sampling prompt. Scalar diagnostics are saved every ten iterations; only the
final S10000 K-tilde array is retained for each trial.

The primary statistic fixes the reference Christoffel estimate in the
numerator and evaluates the inverse-probability sensitivity of the current
regularized sampling law. This directly probes the low-probability tail that
controls the max-ratio prompt compatibility factor. The analysis reports the
arithmetic mean across the five trials and a 95% Student-$t$ confidence
interval on the original metric scale.

## Running Experiments

The public split launchers keep the command interface compact while resolving
the appropriate configuration, output tag, sampling ratios, and recovery
cases.

Run an unweighted split:

```bash
./scripts/unweighted/run_split.sh \
  <main|ablation> \
  <prompt_matched|prompt_mismatched|out_of_range> \
  <first4|last3> \
  <unprompted|daytime_beach|sunset_beach|cat>
```

Run a weighted Christoffel split:

```bash
./scripts/weighted/run_split.sh \
  <main|ablation> \
  <prompt_matched|prompt_mismatched|out_of_range> \
  <first3|last2> \
  <sample_k0_unconditioned|sample_k1_daytime_beach|sample_k2_sunset_beach|sample_k4_cat>
```

Run one weighted baseline split:

```bash
./scripts/weighted/baselines/run_mcs_split.sh prompt_matched first3 unprompted
./scripts/weighted/baselines/run_inverse_square_split.sh prompt_matched first3 unprompted
./scripts/weighted/baselines/run_vdhh_split.sh prompt_matched first3 unprompted
```

Run one convergence trial:

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k0 1
```

Validate the manifests or list the resolved job grids without loading SD1.5:

```bash
python scripts/unweighted/validate_suite.py
python scripts/weighted/validate_suite.py
./scripts/weighted/baselines/list_all.sh
./scripts/weighted/ktilde_convergence/list_all.sh
```

Detailed launcher documentation is available in
[scripts/unweighted/README.md](scripts/unweighted/README.md) and
[scripts/weighted/README.md](scripts/weighted/README.md).

## Output and Resume Behavior

Raw experiment outputs are stored under [results/unweighted/](results/unweighted/)
or [results/weighted/](results/weighted/) according to the operator and
objective convention. The result tag preserves the experiment family, rate
split, sampling law, and recovery case.

A completed reconstruction leaf contains:

- the resolved run configuration and dataset reference;
- the empirical Christoffel artifact reference when applicable;
- sampled Fourier indices, selected probabilities, and mask metadata;
- the zero-filled initialization and reconstructed image;
- the recovered latent variable;
- scalar image-quality metrics;
- optimization and residual traces; and
- the numerical arrays needed to recompute measurement diagnostics.

Weighted leaves additionally record the FFT convention, raw and regularized
probability summaries, $\zeta$, replicated RGB weights, weight extrema, and
raw and weighted measurement residuals.

The runners are resumable. Before optimizing a cell, they validate its saved
artifacts and skip a completed leaf. Split jobs can therefore be restarted
after interruption without recomputing valid reconstructions.

Raw reconstruction outputs and convergence traces are ignored by Git. Paper
figures are written to:

- [results/unweighted/figures/](results/unweighted/figures/)
- [results/weighted/figures/](results/weighted/figures/)

## Analysis

Shared analysis functions are located in [analyze_results/](analyze_results/).
The main notebook groups are:

- [analyze_results/unweighted/main/](analyze_results/unweighted/main/)
- [analyze_results/unweighted/ablation/](analyze_results/unweighted/ablation/)
- [analyze_results/weighted/main/](analyze_results/weighted/main/)
- [analyze_results/weighted/ablation/](analyze_results/weighted/ablation/)

The main recovery notebooks load available split tags, validate their
experiment metadata, and support partially completed grids. They generate
sampling-ratio sweeps for PSNR, SSIM, LPIPS, and per-pixel mean absolute error,
along with reconstruction panels and completion tables.

The K-tilde/Lambda notebook visualizes the empirical Christoffel sampling laws,
computes the empirical prompt compatibility diagnostic, audits the probability
distributions, and ingests the five-trial convergence traces.

## Reproducibility

Configurations, suite manifests, fixed datasets, empirical Christoffel
artifacts, and generated figures are separated by experiment convention.
Every run stores its resolved configuration, seed, environment information,
dataset identity, and relevant artifact metadata. Sampling seeds depend on the
scenario, sampling method, sampling ratio, and repeat, but not on the recovery
prompt, so masks remain paired across recovery prompts.

The five convergence trials use disjoint latent-seed blocks and reuse the same
trial block across the four sampling prompts. This pairing isolates
prompt-dependent differences while preventing overlap with the reference
estimate.

## Citation

```bibtex
@article{delise2026active,
  title={Active Learning for Conditional Generative Compressed Sensing},
  author={DeLise, Alexander and Dexter, Nick},
  journal={arXiv preprint arXiv:2605.05435},
  year={2026}
}
```
