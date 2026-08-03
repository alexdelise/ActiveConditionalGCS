# Experiment Launchers

[unweighted/](unweighted/) contains launchers for the reported main
experiments, CFG ablations, uniform and inverse-square baselines, and S500
empirical Christoffel estimation. Scenario wrappers call the same public
split launcher, so commands have consistent arguments and resume behavior.

[weighted/ktilde_convergence/](weighted/ktilde_convergence/) contains the
S10000 reference and five-trial convergence launchers. Corrected weighted
reconstruction launchers will be added after the new study is complete.

The reconstruction launchers use a compact split interface:

```bash
./scripts/unweighted/run_split.sh main prompt_matched first4 sunset_beach
```

Run one independent convergence trial with:

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k2 1
```

All launchers resolve the project root from their own location, use the active
environment's Python, honor `PYTHON_BIN`, and forward additional arguments to
the corresponding runner. Repeating an interrupted reconstruction command
loads completed reconstructions and continues with the remaining work.
