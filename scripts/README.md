# Launch Scripts

- [unweighted/](unweighted/) contains reconstruction, CFG-ablation, and S500
  K-tilde launchers.
- [weighted/](weighted/) contains weighted reconstruction, sampling-baseline,
  CFG-ablation, and S10000 convergence launchers.

Both experiment suites use the same split-launcher interface:

```bash
./scripts/unweighted/run_split.sh main prompt_matched first4 sunset_beach
./scripts/weighted/run_split.sh main prompt_matched first3 sample_k2_sunset_beach
```

All launchers resolve the project root from their physical path, use the
active environment's Python, honor `PYTHON_BIN`, and forward remaining
arguments to the corresponding runner.
