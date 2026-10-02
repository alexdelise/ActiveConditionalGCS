# Weighted K-Tilde Artifacts

This directory is the single artifact bank for weighted experiments. It holds
the four fixed 10000 secant self-difference estimates, sampling-CFG estimates, and
ordered cross-class estimates. Their build definitions are in the adjacent
`config_*.json` files.

Five independent convergence trials remain grouped by prompt and trial under
[convergence_trials/](convergence_trials/). Their scalar traces are stored in
[../../results/weighted/ktilde/traces/](../../results/weighted/ktilde/traces/).

Run one convergence trial with:

```text
./scripts/weighted/ktilde_convergence/run_trial.sh [sampling-law] [trial]
```

Example:

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k0 1
```

Build a sampling-CFG artifact with
[the CFG launcher](../../scripts/weighted/ktilde_cfg_ablation/run.sh), or an
ordered cross-class artifact with
[the cross-class launcher](../../scripts/weighted/ktilde_cross_class/run.sh).
Their command templates are:

```text
./scripts/weighted/ktilde_cfg_ablation/run.sh [sampling-law] [cfg]
./scripts/weighted/ktilde_cross_class/run.sh [first-class]
```

Examples:

```bash
./scripts/weighted/ktilde_cfg_ablation/run.sh k2 3
./scripts/weighted/ktilde_cross_class/run.sh ca
```

Convergence trials accept `k0`, `k1`, `k2`, or `k4` and trial numbers `1`–`5`.
Sampling-CFG estimates accept `k1`, `k2`, or `k4` and CFG `1`, `3`, or `5`.
Cross-class estimates accept `uc`, `db`, or `ca`, with `sb` fixed as the second
class. All final artifacts are written directly into this directory.

The single
[weighted K-tilde notebook](../../analyze_results/weighted/ktilde/ktilde_analysis.ipynb)
audits and visualizes the self-difference, convergence, sampling-CFG, and
cross-class results. Figures and summary tables are written to
[../../results/weighted/ktilde/figures/](../../results/weighted/ktilde/figures/).
