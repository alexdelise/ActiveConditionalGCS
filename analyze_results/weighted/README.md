# Weighted Analysis

- [out_of_range/](out_of_range/) contains the main weighted out-of-range
  recovery notebook and its analysis helper
- [prompt_matched/](prompt_matched/) contains the main weighted in-range,
  prompt-matched recovery notebook and its analysis helper
- [ablation/](ablation/) contains the three recovery-CFG ablation notebooks
- [ktilde/ktilde_analysis.ipynb](ktilde/ktilde_analysis.ipynb) contains the
  regular, convergence, sampling-CFG, and cross-class K-tilde analyses
- [diagnostics/](diagnostics/) retains focused optimization diagnostics that
  are not part of the main experiment grid

Each notebook reads from the matching directory under
[../../results/weighted/](../../results/weighted/) and writes PDFs and summary
tables into that experiment directory's `figures/` folder.
