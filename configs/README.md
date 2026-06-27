# Configs

This folder contains JSON inputs for the reproducibility scripts.

## Base Configs

- `prompt_matched/sunset/base.json`: prompt-matched in-range sunset experiment.
- `prompt_matched/sunset/first4_base.json`: first four sampling rates for parallel prompt-matched runs.
- `prompt_matched/sunset/last3_base.json`: last three sampling rates for parallel prompt-matched runs.
- `prompt_mismatched/sunset/base.json`: prompt-mismatched in-range sunset experiment.
- `prompt_mismatched/sunset/first4_base.json`: first four sampling rates for parallel prompt-mismatched runs.
- `prompt_mismatched/sunset/last3_base.json`: last three sampling rates for parallel prompt-mismatched runs.
- `out_of_range/sunset/base.json`: out-of-range sunset experiment.
- `out_of_range/sunset/first4_base.json`: first four sampling rates for parallel out-of-range runs.
- `out_of_range/sunset/last3_base.json`: last three sampling rates for parallel out-of-range runs.
- `ablation/<experiment>/sunset/[base|first4_base|last3_base].json`: corrected
  recovery-only CFG ablation bases copied from the corresponding main
  experiment so the sampling grid and optimizer settings remain identical.
- `ablation/<experiment>_old/sunset/`: archived original CFG-ablation configs.
- `example_run.json`: small single-config example.

All paper launch scripts use the single `reconstruction_solver` block and the
`cs` Christoffel/K-tilde sampler.

## Suite Manifests

Files named `[first4_|last3_]sample_<prior>_suite.json` define a case grid for
one experiment and one sampling prior. Suites live under
`configs/<experiment_family>/sunset/` or
`configs/ablation/<experiment_family>/sunset/`, which leaves room for future
datasets such as faces without mixing them into the sunset configs. Each suite
points at a base config and then overrides the K-tilde prior, recovery prompt,
guidance scale for CFG ablations, sampling percentage list, and repeat count
where needed. The aggregate shell scripts compose these per-prior manifests;
there is intentionally no separate combined prompt-matched suite manifest.

Corrected CFG-ablation suites contain recovery CFG 1, 1.5, 3, and 5 cases.
They inherit five repeats from their split base configs. Unconditioned and CFG
7.5 reference rows come from compatible main-experiment outputs rather than
being recomputed by the ablation suites.

Inspect any suite without running reconstructions:

```bash
python run_conditioning_regression.py \
  --suite-config configs/prompt_mismatched/sunset/first4_sample_k1_daytime_beach_suite.json \
  --sampling-methods cs \
  --list-cases
```
