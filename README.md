# Active Learning for Conditional Generative Compressed Sensing

This repository contains the code and experiment artifacts for *Active
Learning for Conditional Generative Compressed Sensing*. The paper studies
image recovery from subsampled Fourier measurements using a
prompt-conditioned generative model. It separates the prompt used to design
the sampling distribution, denoted by $c_s$, from the prompt that defines the
recovery class, denoted by $c_r$. When the target belongs to a
prompt-conditioned model class, its signal prompt is denoted by $c_*$.

The repository provides:

- empirical Christoffel function estimation for Stable Diffusion 1.5 model
  classes;
- prompt-conditioned Fourier sampling distributions;
- latent optimization for conditional generative compressed sensing;
- the reported unweighted reconstruction experiments and CFG ablations;
- uniform and inverse-square sampling baselines;
- convergence diagnostics for the empirical Christoffel estimates and prompt
  compatibility values; and
- analysis notebooks for reconstruction metrics, recovered images, sampling
  distributions, and convergence.

## Setup

Use Python 3.11 or 3.12 with a CUDA-capable PyTorch installation. The Python
dependencies are pinned in [requirements.txt](requirements.txt).

```bash
cd ActiveConditionalGCS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launchers use the active environment's `python` by default and honor
`PYTHON_BIN` when it is set.

Stable Diffusion 1.5 weights are loaded through Hugging Face Diffusers. A
machine that generates datasets, estimates empirical Christoffel functions,
or performs reconstruction must have access to
`stable-diffusion-v1-5/stable-diffusion-v1-5` or a local cached copy. Licensing
and attribution information is provided in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The analysis notebooks use Matplotlib with TeX rendering for paper-style PDF
figures. A TeX installation containing `amsmath` and `amssymb` is therefore
required when exporting figures.

## Repository Organization

<pre>
📦 ActiveConditionalGCS/
│
├── 📄 <a href="README.md">README.md</a>                         ← Project overview and experiment guide
├── 📄 <a href="requirements.txt">requirements.txt</a>                  ← Pinned Python dependencies
├── 📄 <a href="THIRD_PARTY_NOTICES.md">THIRD_PARTY_NOTICES.md</a>           ← Model and dependency notices
├── 📄 <a href="activeconditionalgcs.pdf">activeconditionalgcs.pdf</a>          ← Current paper PDF
├── 📄 <a href="build_dataset.py">build_dataset.py</a>                  ← Build or validate fixed datasets
├── 📄 <a href="build_ktilde.py">build_ktilde.py</a>                   ← Estimate empirical Christoffel functions
├── 📄 <a href="run_ktilde_convergence.py">run_ktilde_convergence.py</a>         ← Run one convergence trial
├── 📄 <a href="run_conditioning_regression.py">run_conditioning_regression.py</a>      ← Run reconstruction suites
├── 📄 <a href="run_cs.py">run_cs.py</a>                          ← Run one Christoffel-sampling configuration
├── 📄 <a href="run_mcs.py">run_mcs.py</a>                         ← Run one uniform-sampling configuration
├── 📄 <a href="run_all.py">run_all.py</a>                         ← Run enabled sampling methods
├── 📄 <a href="run_suite.py">run_suite.py</a>                       ← Run a collection of reconstruction cases
│
├── 📁 <a href="src/">src/</a>                              ← Core reconstruction package
├── 📁 <a href="configs/">configs/</a>                          ← Experiment configurations and manifests
│   ├── 📁 <a href="configs/unweighted/">unweighted/</a>                   ← Unweighted experiments and CFG ablations
│   └── 📁 <a href="configs/weighted/">weighted/</a>                     ← Weighted unitary experiments and diagnostics
│
├── 📁 <a href="datasets/">datasets/</a>                         ← Fixed images and generation metadata
├── 📁 <a href="ktilde/">ktilde/</a>                          ← Empirical Christoffel artifacts
│   ├── 📁 <a href="ktilde/unweighted/">unweighted/</a>                   ← S500 estimates used by reported experiments
│   └── 📁 <a href="ktilde/weighted/">weighted/</a>                     ← S10000 references and convergence trials
│
├── 📁 <a href="scripts/">scripts/</a>                          ← Experiment launchers and validation utilities
│   ├── 📁 <a href="scripts/unweighted/">unweighted/</a>                   ← Unweighted experiments and baselines
│   └── 📁 <a href="scripts/weighted/">weighted/</a>                     ← Weighted reconstruction and K-tilde launchers
│
├── 📁 <a href="analyze_results/">analyze_results/</a>                  ← Shared analysis helpers and notebooks
│   ├── 📁 <a href="analyze_results/unweighted/">unweighted/</a>                   ← Main and CFG-ablation analyses
│   └── 📁 <a href="analyze_results/weighted/">weighted/</a>                     ← Main, ablation, diagnostic, and K-tilde analyses
│
└── 📁 <a href="results/">results/</a>                          ← Local run artifacts and generated figures
    ├── 📁 <a href="results/unweighted/">unweighted/</a>                   ← Per-experiment artifacts and figures
    └── 📁 <a href="results/weighted/">weighted/</a>                     ← Per-experiment artifacts and figures
</pre>

Each major folder has a README describing its contents and local workflows:
[source package](src/README.md), [configurations](configs/README.md),
[datasets](datasets/README.md), [Christoffel artifacts](ktilde/README.md),
[launchers](scripts/README.md), and [analysis](analyze_results/README.md).

## Prompt Roles and Experiment Names

The sampling and recovery prompt family is:

| Code name | Paper notation | Conditioning |
| --- | --- | --- |
| `k0_unconditioned` | $c_{\mathrm{uc}}$ | Unconditioned generation or recovery |
| `k1_daytime_beach` | $c_{\mathrm{db}}$ | `daytime beach` |
| `k2_sunset_beach` | $c_{\mathrm{sb}}$ | `sunset beach` |
| `k4_cat` | $c_{\mathrm{ca}}$ | `cat` |

The empirical Christoffel artifact selected by a reconstruction suite
determines $c_s$. Each suite then evaluates the four corresponding recovery
conditions $c_r$.

| Experiment | Dataset | Interpretation |
| --- | --- | --- |
| `prompt_matched` | `sunset_beach_signal_sd15_512x512` | In-range target generated with $c_*=c_{\mathrm{sb}}$ |
| `prompt_mismatched` | `sunset_sandy_coast_signal_sd15_512x512` | In-range target generated with `sunset over a sandy coast`, outside the four-prompt family |
| `out_of_range` | `out_of_range_512x512` | External sunset image outside the fixed SD1.5 model range |

## Experiments

### Reported unweighted experiments

The reported reconstruction experiments use seven sampling ratios:

```text
0.00015625, 0.0003125, 0.000625, 0.00125, 0.0025, 0.005, 0.01
```

For scheduling, these are divided into `first4` and `last3`. The main study
crosses four empirical Christoffel sampling distributions with four recovery
prompts in each of the three scenarios. The CFG ablations vary the guidance
used during recovery. Uniform sampling and inverse-square variable-density
sampling provide additional reconstruction baselines.

### Weighted reconstruction experiments

The weighted experiments use the unitary Fourier operator, the probability
weights associated with each sampling design, sampling ratios from 1% through
5%, and resumable 2,000-iteration Adam recovery. The current main studies are
the [out-of-range](configs/weighted/out_of_range/) and
[prompt-matched](configs/weighted/prompt_matched/) experiments. Recovery-CFG
ablations are under [configs/weighted/ablation/](configs/weighted/ablation/).

### Empirical Christoffel convergence study

The convergence study treats each S10000 estimate as a fixed reference and
generates five additional independent S10000 estimates per sampling prompt.
Scalar diagnostics are recorded every ten iterations, and uncertainty is
reported across the five trials. See [ktilde/README.md](ktilde/README.md) for
the sampling-law and compatibility definitions.

## Running Experiments

Run one unweighted rate split:

```bash
./scripts/unweighted/run_split.sh \
  <main|ablation> \
  <prompt_matched|prompt_mismatched|out_of_range> \
  <first4|last3> \
  <unprompted|daytime_beach|sunset_beach|cat>
```

Run one S10000 convergence trial:

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k0 1
```

Inspect configurations without loading Stable Diffusion:

```bash
python scripts/unweighted/validate_suite.py
./scripts/weighted/ktilde_convergence/list_all.sh
```

More commands and scheduling details are documented in
[scripts/README.md](scripts/README.md).

## Output and Resume Behavior

Each reconstruction stores its resolved configuration, sampled indices,
metrics, optimization trace, reconstructed image, and latent variable beneath
`results/`. LPIPS, PSNR, SSIM, and per-pixel MAE are saved with new runs; the
analysis notebooks can compute missing LPIPS values for older artifacts. Runs
are resumable: if a command is interrupted, run the same command again and it
will reuse completed reconstructions before continuing with the remaining
ones.

## Analysis

The main notebooks produce sampling-ratio sweeps, uncertainty summaries,
completion tables, and recovered-image panels. Shared loading and plotting
functions are in [analyze_results/](analyze_results/), with notebook-specific
instructions in [analyze_results/README.md](analyze_results/README.md).

## Reproducibility

Every reconstruction records its resolved configuration, random seed, dataset
identity, sampling indices, environment information, and relevant empirical
Christoffel artifact metadata. Sampling seeds depend on the experiment,
sampling method, sampling ratio, and repeat but not on the recovery prompt, so
recovery conditions use paired masks. The fixed datasets and empirical
Christoffel artifacts are stored separately from generated results.

## Citation

```bibtex
@article{delise2026active,
  title={Active Learning for Conditional Generative Compressed Sensing},
  author={DeLise, Alexander and Dexter, Nick},
  journal={arXiv preprint arXiv:2605.05435},
  year={2026}
}
```
