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

The `.npz` stores the probability map. The matching `.meta.json` stores prompt, seed, number of Monte Carlo samples, guidance scale, and grid size.
