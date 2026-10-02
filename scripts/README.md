# Experiment Commands

[weighted/](weighted/) contains the reconstruction and Christoffel-study
launchers. [release/](release/) contains saved-data packaging and download
tools. Run commands from the repository root in the configured Python
environment.

Command templates:

```text
./scripts/weighted/run_main.sh [scenario] [sampling-law] [recovery-prompt]
./scripts/weighted/run_ablation.sh [scenario] [sampling-law] [cfg]
```

Examples:

```bash
./scripts/weighted/run_main.sh prompt_matched k2 sunset_beach
./scripts/weighted/run_ablation.sh prompt_matched k2 3
```

Download the saved data with:

```bash
python scripts/release/download_results.py
```

The saved-data release is public. Reconstruction commands use the active
Python environment, honor `PYTHON_BIN`, and resume when repeated after an
interruption. Append `--dry-run` to inspect a reconstruction setting without
starting optimization.
