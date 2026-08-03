# Experiment Configurations

[unweighted/](unweighted/) contains the public seven-rate main experiments,
CFG ablations, uniform baselines, and inverse-square baselines. Main
Christoffel experiments use `first4` and `last3` manifests so the seven rates
can be scheduled independently. Baseline manifests select one sampling method
and define the four recovery prompts explicitly.

[example_run.json](example_run.json) is a compact single-run example. A suite
manifest points to a base configuration and supplies the sampling condition,
recovery condition, sampling ratios, repeat count, and any case-specific
overrides.

Inspect a manifest without loading Stable Diffusion:

```bash
python run_conditioning_regression.py \
  --suite-config configs/unweighted/prompt_mismatched/sunset/first4_sample_k1_daytime_beach_suite.json \
  --sampling-methods cs \
  --list-cases
```
