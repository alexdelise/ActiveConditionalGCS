# Experiment Configurations

[weighted/](weighted/) contains the three main reconstruction experiments and
the recovery-CFG ablation. Each suite selects a base configuration, sampling
law, recovery condition, and output tag. The base configuration specifies the
sampling ratios, trials, optimizer, and Fourier convention.

Inspect a main or ablation setting from the repository root:

```bash
./scripts/weighted/run_main.sh prompt_matched k2 sunset_beach --dry-run
./scripts/weighted/run_ablation.sh out_of_range k2 3 --dry-run
```

[example_run.json](example_run.json) illustrates a single-run configuration.
