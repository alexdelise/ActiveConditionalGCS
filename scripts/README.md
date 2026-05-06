# Scripts

The scripts are copied from the working run structure and renamed to descriptive experiment names. They are designed for parallel cluster use.

## Naming Pattern

```text
run_<experiment>[_first4|_last3|_cfg_ablation]_sample_<prior>.sh
```

All paper launch scripts run `cs`, meaning Christoffel/K-tilde sampling.

## Split Runs

The longer `prompt_mismatched_in_range` and `out_of_range` experiments keep the original split:

- `first4`: sampling ratios `0.00015625`, `0.0003125`, `0.000625`, `0.00125`
- `last3`: sampling ratios `0.0025`, `0.005`, `0.01`

The split lets different sampling distributions run independently in parallel.

## Aggregates

- `run_prompt_matched_in_range_all.sh`: all four priors for prompt-matched `cs`.
- `run_prompt_mismatched_in_range_first4_all.sh`: all four priors for the first four mismatched rates.
- `run_prompt_mismatched_in_range_last3_all.sh`: all four priors for the last three mismatched rates.
- `run_out_of_range_first4_all.sh`: all four priors for the first four out-of-range rates.
- `run_out_of_range_last3_all.sh`: all four priors for the last three out-of-range rates.

Convenience wrappers named `run_<experiment>_first4_last3_sample_<prior>.sh` run both split parts for one sampling distribution.
Aggregate convenience wrappers named `run_<experiment>_first4_last3_all.sh` run both split parts across all four sampling distributions.

## CFG Ablation

CFG-ablation scripts first copy existing reference rows from the corresponding standard split or standard run, then launch the CFG 1, 3, and 5 cases. This preserves the original paper workflow and avoids recomputing reference cases unnecessarily.
