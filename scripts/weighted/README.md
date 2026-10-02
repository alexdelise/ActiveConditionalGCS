# Weighted Experiment Commands

## Main Reconstructions

```bash
./scripts/weighted/run_main.sh <scenario> <sampling-law> <recovery-prompt>
```

Scenarios: `prompt_matched`, `prompt_mismatched`, `out_of_range`.
Sampling laws: `k0`, `k1`, `k2`, `k4`, `mcs`, `inverse_square`.
Recovery prompts: `unprompted`, `daytime_beach`, `sunset_beach`, `cat`.

One command runs five sampling ratios and five trials, giving 25
reconstructions. Different law/recovery settings can run independently.

## Recovery-CFG Ablation

```bash
./scripts/weighted/run_ablation.sh <scenario> <sampling-law> <cfg>
```

Use the same scenarios, the four Christoffel laws, and CFG `1`, `3`,
`5`, or `7.5`. The recovery prompt is `"sunset beach"`. One setting
contains 25 reconstructions. CFG 1 uses the main-study reference instead
of writing a duplicate ablation result.

Both commands support `--dry-run`, use the active environment's Python,
and honor `PYTHON_BIN`. Run an interrupted command again to resume.

## Christoffel Studies

Run one independent convergence trial:

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k0 1
```

Build a sampling-CFG estimate or a cross-class estimate:

```bash
./scripts/weighted/ktilde_cfg_ablation/run.sh k2 3
./scripts/weighted/ktilde_cross_class/run.sh ca
```

See [the artifact guide](../../ktilde/weighted/README.md) for the definitions.
The saved results are sufficient for visualization without rerunning these
expensive computations.
