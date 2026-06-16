# K-tilde Artifacts

This folder stores the /K-tilde probability maps used by `cs` sampling (Christoffel sampling).

The main priors are:

- `Ktilde_SD15__fft__k0_512x512_S500_ns20`: unconditioned prior.
- `Ktilde_SD15__fft__k1daytimebeach_512x512_S500_ns20`: daytime beach prior.
- `Ktilde_SD15__fft__k2sunsetbeach_512x512_S500_ns20`: sunset beach prior.
- `Ktilde_SD15__fft__k4cat_512x512_S500_ns20`: cat prior.

CFG-ablation variants are included for guidance scales 1, 3, and 5 where needed by the notebooks and suite manifests.

Every checked-in paper artifact has an independent canonical build launcher:

```bash
scripts/ktilde/main/build_k0_unconditioned.sh
scripts/ktilde/main/build_k1_daytime_beach.sh
scripts/ktilde/main/build_k2_sunset_beach.sh
scripts/ktilde/main/build_k4_cat.sh

scripts/ktilde/cfg_ablation/build_k1_daytime_beach_cfg1.sh
scripts/ktilde/cfg_ablation/build_k2_sunset_beach_cfg1.sh
scripts/ktilde/cfg_ablation/build_k4_cat_cfg1.sh
scripts/ktilde/cfg_ablation/build_k1_daytime_beach_cfg3.sh
scripts/ktilde/cfg_ablation/build_k2_sunset_beach_cfg3.sh
scripts/ktilde/cfg_ablation/build_k4_cat_cfg3.sh
scripts/ktilde/cfg_ablation/build_k1_daytime_beach_cfg5.sh
scripts/ktilde/cfg_ablation/build_k2_sunset_beach_cfg5.sh
scripts/ktilde/cfg_ablation/build_k4_cat_cfg5.sh
```

The matching `build_all.sh` scripts rebuild or validate each complete family.
The CFG directory also provides `build_cfg1_all.sh`, `build_cfg3_all.sh`, and
`build_cfg5_all.sh`. Run `scripts/ktilde/build_all_paper.sh` to process all 13
checked-in paper priors sequentially. All launchers accept `--force` to
regenerate an existing artifact and honor `PYTHON_BIN`.

`config_convergence.json` defines separate 10,000-iteration references for the
unconditioned, daytime-beach, sunset-beach, and cat prompts. These references
leave the main 500-iteration paper priors unchanged. Build one reference with:

```bash
bash scripts/ktilde/convergence/build_k2_sunset_beach.sh
```

After its final reference exists, `run_ktilde_convergence.py` deterministically
reruns the same estimator and saves scalar convergence metrics for every
iterate under `results/analysis/ktilde_convergence/`: relative `l2` error,
relative `linf` error, `max_i K_tilde_final_unitary(i) / mu_iteration(i)`, and
the max absolute log-ratio between final and current sampling distributions. It
also verifies that the rerun finishes at the saved reference and prints all
measured convergence metrics every 10 iterations.

The independent build and measurement launchers for all four references live
under `scripts/ktilde/convergence/`.

The convergence section in `analyze_results/sd15_ktilde_lambda_comparison.ipynb`
saves one 2x2 grid per metric and all standalone plots under
`results/analysis/ktilde_convergence/figures/`.

The shared `scripts/ktilde/build_named.sh` helper resolves the configured
Python interpreter and invokes `build_ktilde.py`; per-artifact scripts provide
the exact catalog and artifact name. The `.npz` stores the probability map.
The matching `.meta.json` stores prompt, seed, number of Monte Carlo samples,
guidance scale, and grid size.
