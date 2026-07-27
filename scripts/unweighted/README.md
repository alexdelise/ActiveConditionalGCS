# Unweighted Launchers

[run_split.sh](run_split.sh) launches one rate split:

```bash
./scripts/unweighted/run_split.sh \
  <main|ablation> \
  <prompt_matched|prompt_mismatched|out_of_range> \
  <first4|last3> \
  <sampling-prior>
```

The split definitions are:

```text
first4 = 0.00015625, 0.0003125, 0.000625, 0.00125
last3  = 0.0025, 0.005, 0.01
```

[run_suite.sh](run_suite.sh) runs both splits sequentially. Scenario-specific
wrappers are available under [prompt_matched/](prompt_matched/),
[prompt_mismatched/](prompt_mismatched/), [out_of_range/](out_of_range/), and
[ablation/](ablation/).

S500 K-tilde builders are under [ktilde/](ktilde/). Validate every suite
manifest with [validate_suite.py](validate_suite.py):

```bash
python scripts/unweighted/validate_suite.py
```
