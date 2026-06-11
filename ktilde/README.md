# K-tilde Artifacts

This folder stores the /K-tilde probability maps used by `cs` sampling (Christoffel sampling).

The main priors are:

- `Ktilde_SD15__fft__k0_512x512_S500_ns20`: unconditioned prior.
- `Ktilde_SD15__fft__k1daytimebeach_512x512_S500_ns20`: daytime beach prior.
- `Ktilde_SD15__fft__k2sunsetbeach_512x512_S500_ns20`: sunset beach prior.
- `Ktilde_SD15__fft__k4cat_512x512_S500_ns20`: cat prior.

CFG-ablation variants are included for guidance scales 1, 3, and 5 where needed by the notebooks and suite manifests.

To rebuild a K-tilde artifact:

```bash
python build_ktilde.py --name Ktilde_SD15__fft__k2sunsetbeach_512x512_S500_ns20
```

`config_convergence.json` defines separate 10,000-iteration references for the
unconditioned, daytime-beach, sunset-beach, and cat prompts. These references
leave the main 500-iteration paper priors unchanged. Build one reference with:

```bash
python build_ktilde.py \
  --config ktilde/config_convergence.json \
  --name Ktilde_SD15__fft__k2sunsetbeach_512x512_S10000_ns20
```

After its final reference exists, `run_ktilde_convergence.py` deterministically
reruns the same estimator and saves only the relative L2 error of every iterate
under `results/analysis/ktilde_convergence/`. It also verifies that the rerun
finishes at the saved reference.

The independent build and measurement launchers for all four references live
under `scripts/ktilde/convergence/`.

The `.npz` stores the probability map. The matching `.meta.json` stores prompt, seed, number of Monte Carlo samples, guidance scale, and grid size.
