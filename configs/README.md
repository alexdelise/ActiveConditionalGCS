# Configs

This folder contains JSON inputs for the reproducibility scripts.

## Base Configs

- `prompt_matched_in_range_base.json`: prompt-matched in-range experiment.
- `prompt_mismatched_in_range_base.json`: prompt-mismatched in-range experiment.
- `prompt_mismatched_in_range_first4_base.json`: first four sampling rates for parallel runs.
- `prompt_mismatched_in_range_last3_base.json`: last three sampling rates for parallel runs.
- `prompt_mismatched_in_range_cfg_ablation_base.json`: CFG-scale ablation for the mismatched experiment.
- `out_of_range_base.json`: out-of-range experiment.
- `out_of_range_first4_base.json`: first four sampling rates for parallel runs.
- `out_of_range_last3_base.json`: last three sampling rates for parallel runs.
- `out_of_range_cfg_ablation_base.json`: CFG-scale ablation for the out-of-range experiment.
- `diffusion_backprop_example.json`: small single-config example.

All paper launch scripts use `dc_methods.diffusion_backprop` and the `cs`
Christoffel/K-tilde sampler.

## Suite Manifests

Files named `*_sample_<prior>_suite.json` define a case grid for one experiment
and one sampling prior. Each suite points at a base config and then overrides
the K-tilde prior, recovery prompt, guidance scale for CFG ablations, sampling
percentage list, and repeat count where needed. The aggregate shell scripts
compose these per-prior manifests; there is intentionally no separate combined
prompt-matched suite manifest.

Inspect any suite without running reconstructions:

```bash
python run_conditioning_regression.py \
  --suite-config configs/prompt_mismatched_in_range_first4_sample_k1_daytime_beach_suite.json \
  --sampling-methods cs \
  --list-cases
```
