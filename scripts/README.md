# Launch Scripts

- [unweighted/](unweighted/) contains reconstruction, CFG-ablation, and S500
  K-tilde launchers.
- [weighted/ktilde_convergence/](weighted/ktilde_convergence/) contains the
  S10000 convergence launchers.

The reconstruction launchers use a compact split interface:

```bash
./scripts/unweighted/run_split.sh main prompt_matched first4 sunset_beach
```

Run one independent convergence trial with:

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k2 1
```

All launchers resolve the project root from their physical path, use the
active environment's Python, honor `PYTHON_BIN`, and forward remaining
arguments to the corresponding runner.
