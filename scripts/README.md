# Scripts

The launch scripts mirror the nested config and result layout. Experiment family
and dataset live in the directory path, while the filename names the run variant.

```text
scripts/<experiment_family>/sunset/sample_<prior>.sh
scripts/<experiment_family>/sunset/[first4_|last3_]sample_<prior>.sh
scripts/ablation/<experiment_family>/sunset/sample_<prior>.sh
scripts/ktilde/main/build_<prior>.sh
scripts/ktilde/cfg_ablation/build_<prior>_cfg<scale>.sh
scripts/ktilde/convergence/[build_|measure_]<prior>.sh
```

All paper launch scripts run `cs`, meaning Christoffel/K-tilde sampling.

## Split Runs

All three main experiments use the same seven-rate grid and split:

- `first4`: sampling ratios `0.00015625`, `0.0003125`, `0.000625`, `0.00125`
- `last3`: sampling ratios `0.0025`, `0.005`, `0.01`

The split lets different sampling distributions run independently in parallel.

## Aggregates

- `prompt_matched/sunset/all.sh`: all four priors for prompt-matched `cs`.
- `prompt_matched/sunset/first4_all.sh`: all four priors for the first four prompt-matched rates.
- `prompt_matched/sunset/last3_all.sh`: all four priors for the last three prompt-matched rates.
- `prompt_mismatched/sunset/first4_all.sh`: all four priors for the first four mismatched rates.
- `prompt_mismatched/sunset/last3_all.sh`: all four priors for the last three mismatched rates.
- `out_of_range/sunset/first4_all.sh`: all four priors for the first four out-of-range rates.
- `out_of_range/sunset/last3_all.sh`: all four priors for the last three out-of-range rates.

Convenience wrappers named `first4_last3_sample_<prior>.sh` run both split parts
for one sampling distribution. Aggregate convenience wrappers named
`first4_last3_all.sh` run both split parts across all four sampling
distributions.

## K-Tilde Builders

Canonical k-tilde build launchers live together under `ktilde/`:

- `ktilde/main/`: one launcher for each of the four primary CFG 7.5, S500
  priors, plus `build_all.sh`.
- `ktilde/cfg_ablation/`: one launcher for each of the nine conditioned CFG 1,
  3, and 5 S500 priors, per-CFG aggregates, and `build_all.sh`.
- `ktilde/convergence/`: one S10000 reference builder and one convergence
  measurement launcher for each of the four paper prompts.

All build launchers delegate to `ktilde/build_named.sh`, honor `PYTHON_BIN`,
validate existing artifacts by default, and forward arguments such as
`--force`. The top-level `ktilde/build_all_paper.sh` aggregate processes all
13 checked-in paper priors sequentially.

## CFG Ablation

CFG-ablation scripts first copy existing reference rows from the corresponding
standard split or standard run, then launch the CFG 1, 3, and 5 cases. The
updated prompt-matched ablation follows its seven-rate first4/last3 main sweep;
the other ablations retain their existing five-rate grids.

The historical K-tilde commands under
`ablation/ktilde/sunset/build_cfg*_conditioned_prompts.sh` remain as
compatibility aliases. They delegate to the canonical
`ktilde/cfg_ablation/build_cfg*_all.sh` launchers.

## K-Tilde Convergence

The launchers under `ktilde/convergence/` are split into independent jobs for
the four paper prompts:

- `build_<prior>.sh` creates the final 10,000-iteration reference in `ktilde/`.
- `measure_<prior>.sh` reruns the estimator and writes only the relative L2
  convergence trace under `results/analysis/ktilde_convergence/`, while
  streaming the measured relative L2 error every 10 iterations.
- `build_all.sh` and `measure_all.sh` run their corresponding four jobs
  sequentially.

Run a prompt's build before its matching measurement. All launchers honor
`PYTHON_BIN` and forward additional arguments such as `--force`.
