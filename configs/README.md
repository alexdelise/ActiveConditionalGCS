# Experiment Configurations

- [unweighted/](unweighted/) contains the seven-rate main and CFG-ablation
  configurations with `first4` and `last3` splits.
- [weighted/](weighted/) contains the five-rate main, CFG-ablation, and
  sampling-baseline configurations with `first3` and `last2` splits.
- [example_run.json](example_run.json) is a compact single-run example.

Each suite manifest points to its base configuration and defines the sampling
prior, recovery prompt, sampling grid, and repeat count.

Inspect a manifest without running a reconstruction:

```bash
python run_conditioning_regression.py \
  --suite-config configs/unweighted/prompt_mismatched/sunset/first4_sample_k1_daytime_beach_suite.json \
  --sampling-methods cs \
  --list-cases
```
