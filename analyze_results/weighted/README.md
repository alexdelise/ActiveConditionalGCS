# Weighted Analysis

## Main Experiments

- [prompt_matched/prompt_matched_results.ipynb](prompt_matched/prompt_matched_results.ipynb)
- [prompt_mismatched/prompt_mismatched_results.ipynb](prompt_mismatched/prompt_mismatched_results.ipynb)
- [out_of_range/out_of_range_results.ipynb](out_of_range/out_of_range_results.ipynb)

Each notebook shows completion counts, metric sweeps, recovered-image panels,
optimization traces with trial uncertainty, and aggregate metrics. It reads
the corresponding directory under [results/weighted/](../../results/weighted/)
and writes PDFs and tables into that experiment's figures folder.

[weighted_main_forest.ipynb](weighted_main_forest.ipynb) produces the combined
main-experiment forest plot.

## Recovery-CFG Ablation

The [ablation notebooks](ablation/README.md) compare CFG 1, 3, 5, and 7.5,
with five trials for every setting. The CFG 1 rows are read directly from the
main results. [The combined forest notebook](ablation/weighted_ablation_forest.ipynb)
summarizes all three scenarios.

## Christoffel Studies

[ktilde/ktilde_analysis.ipynb](ktilde/ktilde_analysis.ipynb) analyzes
self-difference laws, convergence, sampling-CFG estimates, and cross-class
compatibility values. It writes into
[results/weighted/ktilde/figures/](../../results/weighted/ktilde/figures/).

The [saved-data guide](../../scripts/release/README.md) describes the data
needed to run every notebook without reconstructing images. LPIPS, PSNR, SSIM,
and the best weighted objective are read from the saved artifacts. SSIM uses
a 7 × 7 uniform RGB window with sample covariance.
