# Unweighted Analysis

- [main/](main/) contains the three main recovery notebooks
- [ablation/](ablation/) contains the three recovery-CFG ablation notebooks
- [ktilde/ktilde_analysis.ipynb](ktilde/ktilde_analysis.ipynb) contains the
  unweighted K-tilde and compatibility analysis

The notebooks read artifacts from the corresponding directory under
[../../results/unweighted/](../../results/unweighted/). Each experiment writes
its PDFs and summary tables into its own `figures/` folder. Main recovery
notebooks compare Christoffel, uniform, and inverse-square sampling using PSNR,
SSIM, LPIPS, and per-pixel MAE.
