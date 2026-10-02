# Unweighted Analysis

- [main/](main/) contains the three main recovery notebooks
- [ablation/](ablation/) contains the three recovery-CFG ablation notebooks
- [ktilde/ktilde_analysis.ipynb](ktilde/ktilde_analysis.ipynb) analyzes the
  stored Christoffel functions and compatibility values

The notebooks use [the shared analysis helpers](../../analyze_results/) and
read [the archived results](../results/). Each experiment writes its PDFs
and summary tables into its own figures folder. Main recovery notebooks
compare Christoffel, MCS, and inverse-square sampling using PSNR, SSIM,
LPIPS, and per-pixel MAE.
