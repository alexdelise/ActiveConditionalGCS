# Experiment Configurations

[unweighted/](unweighted/) contains the seven-rate unweighted main experiments,
recovery-CFG ablations, uniform baselines, and inverse-square baselines.

[weighted/](weighted/) contains the weighted unitary experiments:

- [out_of_range/](weighted/out_of_range/)
- [prompt_matched/](weighted/prompt_matched/)
- [ablation/](weighted/ablation/)
- [diagnostics/](weighted/diagnostics/)

A suite manifest points to a base configuration and supplies the sampling law,
recovery condition, sampling ratios, repeat count, and case-specific
overrides. [example_run.json](example_run.json) is a compact single-run
example.

Inspect a manifest without loading Stable Diffusion:

```bash
python run_conditioning_regression.py \
  --suite-config configs/weighted/out_of_range/k2_suite.json \
  --sampling-methods cs \
  --list-cases
```
