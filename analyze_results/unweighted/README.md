# Unweighted Analysis

- [main/](main/) contains the three recovery notebooks and the S500
  K-tilde/Lambda notebook.
- [ablation/](ablation/) contains the three CFG-ablation notebooks.

The notebooks read from
[../../results/unweighted/](../../results/unweighted/), use artifacts from
[../../ktilde/unweighted/](../../ktilde/unweighted/), and write figures to
[../../results/unweighted/figures/](../../results/unweighted/figures/).

Main recovery notebooks compare Christoffel, uniform, and inverse-square
sampling with PSNR, SSIM, LPIPS, and per-pixel MAE sweeps and recovered-image
panels. New runs provide LPIPS in their saved result data; older runs continue
to use the shared analysis-time LPIPS cache.
