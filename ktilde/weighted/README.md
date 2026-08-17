# Weighted K-Tilde Artifacts

This directory is the single artifact bank for weighted experiments. It holds
the four fixed S10000 self-difference estimates, sampling-CFG estimates, and
ordered cross-class estimates. Their build definitions are in the adjacent
`config_*.json` files.

Five independent convergence trials remain grouped by prompt and trial under
[convergence_trials/](convergence_trials/). Their scalar traces are stored in
[../../results/weighted/ktilde/traces/](../../results/weighted/ktilde/traces/).

Run one convergence trial with:

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k0 1
```

Build a sampling-CFG artifact with
[the CFG launcher](../../scripts/weighted/ktilde_cfg_ablation/run.sh), or an
ordered cross-class artifact with
[the cross-class launcher](../../scripts/weighted/ktilde_cross_class/run.sh).
All final artifacts are written directly into this directory.

The single
[weighted K-tilde notebook](../../analyze_results/weighted/ktilde/ktilde_analysis.ipynb)
audits and visualizes the self-difference, convergence, sampling-CFG, and
cross-class results. Figures and summary tables are written to
[../../results/weighted/ktilde/figures/](../../results/weighted/ktilde/figures/).
