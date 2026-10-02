# Weighted Experiment Commands

## Main Reconstructions

Names in square brackets are placeholders. Replace them with the desired
values and omit the brackets when running a command.

```text
./scripts/weighted/run_main.sh [scenario] [sampling-law] [recovery-prompt]
```

Example:

```bash
./scripts/weighted/run_main.sh prompt_matched k2 sunset_beach
```

Scenarios: `prompt_matched`, `prompt_mismatched`, `out_of_range`.
Sampling laws: `k0`, `k1`, `k2`, `k4`, `mcs`, `inverse_square`.
Recovery prompts: `unprompted`, `daytime_beach`, `sunset_beach`, `cat`.

One command runs five sampling ratios and five trials, giving 25
reconstructions. Different law/recovery settings can run independently.

## Recovery-CFG Ablation

```text
./scripts/weighted/run_ablation.sh [scenario] [sampling-law] [cfg]
```

Example:

```bash
./scripts/weighted/run_ablation.sh prompt_matched k2 3
```

Use the same scenarios, the four Christoffel laws, and CFG `1`, `3`,
`5`, or `7.5`. The recovery prompt is `"sunset beach"`. One setting
contains 25 reconstructions. CFG 1 uses the main-study reference instead
of writing a duplicate ablation result.

Both commands support `--dry-run`, use the active environment's Python,
and honor `PYTHON_BIN`. Run an interrupted command again to resume.

## Christoffel Studies

Run one independent convergence trial:

```text
./scripts/weighted/ktilde_convergence/run_trial.sh [sampling-law] [trial]
```

Example:

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k0 1
```

Build a sampling-CFG estimate:

```text
./scripts/weighted/ktilde_cfg_ablation/run.sh [sampling-law] [cfg]
```

Example:

```bash
./scripts/weighted/ktilde_cfg_ablation/run.sh k2 3
```

Build a cross-class estimate with the sunset-beach class fixed as the second class:

```text
./scripts/weighted/ktilde_cross_class/run.sh [first-class]
```

Example:

```bash
./scripts/weighted/ktilde_cross_class/run.sh ca
```

Convergence trials accept `k0`, `k1`, `k2`, or `k4` and trial numbers `1`–`5`.
Sampling-CFG estimates accept `k1`, `k2`, or `k4` and CFG `1`, `3`, or `5`.
Cross-class estimates accept `uc`, `db`, or `ca` as the first class.

See [the artifact guide](../../ktilde/weighted/README.md) for the definitions.
The saved results are sufficient for visualization without rerunning these
expensive computations.
