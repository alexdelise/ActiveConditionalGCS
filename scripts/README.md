# Scripts

The launch scripts mirror the nested config and result layout. Experiment family
and dataset live in the directory path, while the filename names the run variant.

```text
scripts/<experiment_family>/sunset/sample_<prior>.sh
scripts/<experiment_family>/sunset/[first4_|last3_]sample_<prior>.sh
scripts/ablation/<experiment_family>/sunset/sample_<prior>.sh
```

All paper launch scripts run `cs`, meaning Christoffel/K-tilde sampling.

## Split Runs

The longer `prompt_mismatched_in_range` and `out_of_range` experiments keep the original split:

- `first4`: sampling ratios `0.00015625`, `0.0003125`, `0.000625`, `0.00125`
- `last3`: sampling ratios `0.0025`, `0.005`, `0.01`

The split lets different sampling distributions run independently in parallel.

## Aggregates

- `prompt_matched/sunset/all.sh`: all four priors for prompt-matched `cs`.
- `prompt_mismatched/sunset/first4_all.sh`: all four priors for the first four mismatched rates.
- `prompt_mismatched/sunset/last3_all.sh`: all four priors for the last three mismatched rates.
- `out_of_range/sunset/first4_all.sh`: all four priors for the first four out-of-range rates.
- `out_of_range/sunset/last3_all.sh`: all four priors for the last three out-of-range rates.

Convenience wrappers named `first4_last3_sample_<prior>.sh` run both split parts
for one sampling distribution. Aggregate convenience wrappers named
`first4_last3_all.sh` run both split parts across all four sampling
distributions.

## CFG Ablation

CFG-ablation scripts first copy existing reference rows from the corresponding standard split or standard run, then launch the CFG 1, 3, and 5 cases. This preserves the original paper workflow and avoids recomputing reference cases unnecessarily.

The K-tilde builders needed by the CFG-ablation notebooks live under
`ablation/ktilde/sunset/build_cfg*_conditioned_prompts.sh`.
