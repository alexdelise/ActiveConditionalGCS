# Weighted Launchers

Reconstruction launchers are grouped by experiment:

- [out_of_range/](out_of_range/)
- [prompt_matched/](prompt_matched/)
- [ablation/](ablation/)
- [diagnostics/](diagnostics/)

K-tilde launchers are grouped under
[ktilde_convergence/](ktilde_convergence/),
[ktilde_cfg_ablation/](ktilde_cfg_ablation/), and
[ktilde_cross_class/](ktilde_cross_class/).

Every reconstruction launcher is safe to repeat. Completed rows are skipped,
and an incomplete reconstruction resumes from its latest optimizer checkpoint.
