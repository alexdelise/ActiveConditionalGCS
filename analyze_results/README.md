# Analysis

This folder contains the analysis helpers and notebooks used to inspect reproduced results and recreate paper-style figures.

- `sd15_conditioning_experiment.py`: loaders and plotting utilities for prompt-conditioning experiments.
- `sd15_cfg_ablation_analysis.py`: CFG-ablation analysis utilities.
- `ktilde_lambda_comparison.ipynb`: k-tilde convergence, sampling-distribution, and prompt-compatibility analysis.
- `prompt_matched_in_range_results.ipynb`: updated seven-rate prompt-matched in-range results notebook.
- `prompt_mismatched_in_range_results.ipynb`: prompt-mismatched in-range results notebook.
- `out_of_range_results.ipynb`: out-of-range results notebook.
- `prompt_matched_in_range_cfg_ablation.ipynb`: prompt-matched CFG ablation.
- `prompt_mismatched_in_range_cfg_ablation.ipynb`: prompt-mismatched CFG ablation.
- `out_of_range_cfg_ablation.ipynb`: out-of-range CFG ablation.
- `old/`: archived pre-fix notebooks, using the same cleaned names inside the archive folder.

The notebooks expect completed outputs under the nested `../results/<experiment_family>/sunset/` layout and write PDF figures under `../results/analysis/...`.

Current CFG-ablation notebooks merge first4/last3 outputs, synchronize
compatible unconditioned and CFG 7.5 main-experiment references, and compare
recovery CFG 1, 1.5, 3, 5, and 7.5 over five trials. Archived notebooks read
only matching `_old` result paths.

The early convergence section in `ktilde_lambda_comparison.ipynb` expects
all four traces from `scripts/ktilde/convergence/measure_<prior>.sh`. It
displays one shared 2x2 panel per convergence metric and saves each panel plus
four standalone PDFs under `results/analysis/ktilde_convergence/figures/`. The
relative-`l2` panel is `sd15_ktilde_convergence_grid.pdf`; added metrics use
explicit stems such as `sd15_ktilde_convergence_relative_linf_error_grid.pdf`.

The top-level experiment-family directories under `results/` contain raw,
restartable run artifacts. `results/analysis/` contains derived figures,
compact summaries, and analysis-only traces generated from those runs.
This separation keeps notebook-generated outputs out of the raw run trees;
all of them remain under the Git-ignored `results/` root because they are
reproducible outputs rather than source files.
