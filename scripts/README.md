# Experiment Launchers

[unweighted/](unweighted/) contains launchers for the unweighted main
experiments, recovery-CFG ablations, uniform and inverse-square baselines, and
S500 empirical Christoffel estimation.

[weighted/](weighted/) contains launchers for the weighted experiments:

- [out_of_range/](weighted/out_of_range/)
- [prompt_matched/](weighted/prompt_matched/)
- [ablation/](weighted/ablation/)
- [diagnostics/](weighted/diagnostics/)
- [ktilde_convergence/](weighted/ktilde_convergence/)
- [ktilde_cfg_ablation/](weighted/ktilde_cfg_ablation/)
- [ktilde_cross_class/](weighted/ktilde_cross_class/)

All launchers resolve the project root from their physical location, use the
active environment's Python, and honor `PYTHON_BIN`. Reconstruction commands
are safe to repeat: complete rows are skipped and an incomplete row resumes
from its latest optimizer checkpoint.
