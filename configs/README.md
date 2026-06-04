# Configs

This folder contains JSON inputs for the reproducibility scripts.

## Base Configs

- `prompt_matched/sunset/base.json`: prompt-matched in-range sunset experiment.
- `prompt_mismatched/sunset/base.json`: prompt-mismatched in-range sunset experiment.
- `prompt_mismatched/sunset/first4_base.json`: first four sampling rates for parallel prompt-mismatched runs.
- `prompt_mismatched/sunset/last3_base.json`: last three sampling rates for parallel prompt-mismatched runs.
- `out_of_range/sunset/base.json`: out-of-range sunset experiment.
- `out_of_range/sunset/first4_base.json`: first four sampling rates for parallel out-of-range runs.
- `out_of_range/sunset/last3_base.json`: last three sampling rates for parallel out-of-range runs.
- `ablation/<experiment>/sunset/base.json`: CFG-scale ablation bases where the ablation needs a separate base config.
- `diffusion_backprop_example.json`: small single-config example.

All paper launch scripts use `dc_methods.diffusion_backprop` and the `cs`
Christoffel/K-tilde sampler.

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

Inspect any suite without running reconstructions:

```bash
python run_conditioning_regression.py \
  --suite-config configs/prompt_mismatched/sunset/first4_sample_k1_daytime_beach_suite.json \
  --sampling-methods cs \
  --list-cases
```
