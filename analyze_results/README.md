# Analysis

This folder contains the analysis helpers and notebooks used to inspect reproduced results and recreate paper-style figures.

- `sd15_conditioning_experiment.py`: loaders and plotting utilities for prompt-conditioning experiments.
- `sd15_cfg_ablation_analysis.py`: CFG-ablation analysis utilities.
- `sd15_prompt_matched_in_range_results.ipynb`: prompt-matched in-range results notebook.
- `sd15_prompt_mismatched_in_range_results copy.ipynb`: prompt-mismatched in-range results notebook.
- `sd15_out_of_range_results.ipynb`: out-of-range results notebook.
- `sd15_prompt_matched_in_range_cfg_ablation.ipynb`: prompt-matched CFG ablation.
- `sd15_prompt_mismatched_in_range_cfg_ablation.ipynb`: prompt-mismatched CFG ablation.
- `sd15_out_of_range_cfg_ablation.ipynb`: out-of-range CFG ablation.

The notebooks expect completed outputs under the nested `../results/<experiment_family>/sunset/` layout and write PDF figures under `../results/analysis/...`.
