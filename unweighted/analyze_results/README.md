# Unweighted Analysis

## Main Experiments

- [main/prompt_matched_results.ipynb](main/prompt_matched_results.ipynb)
- [main/prompt_mismatched_results.ipynb](main/prompt_mismatched_results.ipynb)
- [main/out_of_range_results.ipynb](main/out_of_range_results.ipynb)

Each notebook shows completion counts, metric sweeps, recovered-image panels,
optimization traces with trial uncertainty, and aggregate metrics. It reads
the corresponding directory under [results/](../results/) and writes figures
and numerical summaries into that experiment's figures folder.

[unweighted_main_forest.ipynb](unweighted_main_forest.ipynb) produces the
combined main-experiment forest plot.

## Recovery-CFG Ablation

The [ablation notebooks](ablation/) compare CFG 1, 3, 5, and 7.5,
with five trials for every setting. The CFG 7.5 rows are read directly from
the main results. [The combined forest notebook](ablation/unweighted_ablation_forest.ipynb)
summarizes all three scenarios.

## Christoffel Studies

[ktilde/ktilde_analysis.ipynb](ktilde/ktilde_analysis.ipynb) analyzes the
stored Christoffel functions and compatibility values.

The [saved-data guide](../../scripts/release/README.md) describes how to
download the results.
