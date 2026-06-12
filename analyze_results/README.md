# Analysis

This folder contains the analysis helpers and notebooks used to inspect reproduced results and recreate paper-style figures.

- `sd15_conditioning_experiment.py`: loaders and plotting utilities for prompt-conditioning experiments.
- `sd15_cfg_ablation_analysis.py`: CFG-ablation analysis utilities.
- `sd15_ktilde_lambda_comparison.ipynb`: k-tilde convergence, sampling-distribution, and prompt-compatibility analysis.
- `sd15_prompt_matched_in_range_results.ipynb`: updated seven-rate prompt-matched in-range results notebook.
- `sd15_prompt_matched_in_range_results_old.ipynb`: archived pre-fix prompt-matched results notebook.
- `sd15_prompt_mismatched_in_range_results.ipynb`: prompt-mismatched in-range results notebook.
- `sd15_out_of_range_results.ipynb`: out-of-range results notebook.
- `sd15_prompt_matched_in_range_cfg_ablation.ipynb`: prompt-matched CFG ablation.
- `sd15_prompt_mismatched_in_range_cfg_ablation.ipynb`: prompt-mismatched CFG ablation.
- `sd15_out_of_range_cfg_ablation.ipynb`: out-of-range CFG ablation.

The notebooks expect completed outputs under the nested `../results/<experiment_family>/sunset/` layout and write PDF figures under `../results/analysis/...`.

The early convergence section in `sd15_ktilde_lambda_comparison.ipynb` expects
all four traces from `scripts/ktilde/convergence/measure_<prior>.sh`. It
displays the shared 2x2 panel and saves that panel plus four standalone PDFs
under `results/analysis/ktilde_convergence/figures/`.
