# Unweighted Experiment Commands

Run commands from the repository root in the configured Python environment:

```bash
./unweighted/scripts/run_suite.sh main prompt_matched k0
./unweighted/scripts/run_suite.sh ablation prompt_matched k0
python unweighted/scripts/validate_suite.py
```

The scenario can be `prompt_matched`, `prompt_mismatched`, or
`out_of_range`. Priors are `k0`, `k1`, `k2`, and `k4`. These
commands preserve the original seven sampling ratios and five trials and
write into [the archived results](../results/).

The original rate-split commands remain available for individual jobs:

```bash
./unweighted/scripts/run_split.sh main prompt_matched first4 k0 --list-cases
./unweighted/scripts/baselines/run_mcs_split.sh prompt_matched first4 sunset_beach --dry-run
./unweighted/scripts/baselines/run_inverse_square_split.sh prompt_matched last3 sunset_beach --dry-run
```

The [configurations](../configs/) specify the original optimization settings.
Repeat an interrupted reconstruction command to reuse completed artifacts and
resume unfinished optimization.
