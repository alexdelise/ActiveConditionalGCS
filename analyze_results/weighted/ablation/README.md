# Weighted Recovery-CFG Ablation

Open the [prompt-matched](prompt_matched_cfg_ablation.ipynb),
[prompt-mismatched](prompt_mismatched_cfg_ablation.ipynb), or
[out-of-range](out_of_range_cfg_ablation.ipynb) notebook.
[weighted_ablation_forest.ipynb](weighted_ablation_forest.ipynb) combines the
three scenarios.

The recovery prompt is `"sunset beach"`, with CFG 1, 3, 5, and 7.5.
Sampling uses the four fixed Christoffel laws generated with CFG 7.5.
Every setting uses ratios 1% through 5%, five trials, the weighted unitary
operator, and the main experiment's 2,000-iteration learning-rate schedule.

Each scenario contains 400 analysis rows: 100 CFG 1 references read directly
from the main results and 300 new CFG 3, 5, and 7.5 reconstructions. The full
study contains 1,200 analysis rows, including 900 new reconstructions.

The notebooks show separate metric sweeps for LPIPS, PSNR, SSIM, and best
weighted loss, optimization traces, reconstruction panels, and aggregate
forest plots. Shaded sweep bands are 95% confidence intervals. Figures and
summary tables are written under
[results/weighted/ablation/](../../../results/weighted/ablation/).

```bash
./scripts/weighted/run_ablation.sh prompt_matched k2 3
./scripts/weighted/run_ablation.sh prompt_matched k2 3 --dry-run
```
