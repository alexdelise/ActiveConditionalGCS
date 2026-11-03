# Unweighted Experiments

The unweighted experiments compare four Christoffel sampling laws with uniform
and inverse-square sampling, and include a recovery-CFG ablation. The code uses the
shared [source package](../src/), [datasets](../datasets/), and
[analysis helpers](../analyze_results/) at the repository root.

## Contents

- [configs/](configs/): reconstruction configurations and suite manifests
- [scripts/](scripts/): reconstruction and Christoffel estimation commands
- [analyze_results/](analyze_results/): main, ablation, and Christoffel notebooks
- [ktilde/](ktilde/): the stored 500-secant estimates and their definitions
- [results/](results/): local reconstruction artifacts and generated figures

Run these commands from the repository root in the configured Python environment:

```text
./unweighted/scripts/run_suite.sh [experiment-type] [scenario] [sampling-law]
```

Here, `[experiment-type]` is `main` or `ablation`, `[scenario]` is
`prompt_matched`, `prompt_mismatched`, or `out_of_range`, and `[sampling-law]`
is `k0`, `k1`, `k2`, or `k4`. Omit the brackets when running a command.

Examples:

```bash
./unweighted/scripts/run_suite.sh main prompt_matched k0
./unweighted/scripts/run_suite.sh ablation prompt_matched k0
python unweighted/scripts/validate_suite.py
```

The main experiments use seven sampling ratios, five trials, recovery CFG 7.5,
and 500 Adam iterations. The recovery-CFG ablation compares CFG 1, 3, 5, and 7.5
under the four Christoffel laws. Results are written into [results/](results/).
Run the same command again after an interruption to reuse completed reconstructions and resume
unfinished optimization. Download the saved results with
`python scripts/release/download_results.py`.

Open the [main notebooks](analyze_results/main/),
[CFG-ablation notebooks](analyze_results/ablation/), or
[Christoffel notebook](analyze_results/ktilde/ktilde_analysis.ipynb) to analyze
the saved artifacts. Each experiment writes PDFs into its own results folder.
