# Weighted Launchers

[run_split.sh](run_split.sh) launches one main or CFG-ablation rate split:

```bash
./scripts/weighted/run_split.sh \
  <main|ablation> \
  <prompt_matched|prompt_mismatched|out_of_range> \
  <first3|last2> \
  <sampling-prior>
```

Scenario wrappers are available under [prompt_matched/](prompt_matched/),
[prompt_mismatched/](prompt_mismatched/), [out_of_range/](out_of_range/), and
[ablation/](ablation/).

Uniform MCS, inverse-square, and VDHH launchers are under
[baselines/](baselines/). K-tilde convergence launchers are under
[ktilde_convergence/](ktilde_convergence/).

Validate the weighted CS manifests with [validate_suite.py](validate_suite.py):

```bash
python scripts/weighted/validate_suite.py
```
