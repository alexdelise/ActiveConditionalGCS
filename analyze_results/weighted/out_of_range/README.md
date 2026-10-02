# Weighted Out-of-Range Recovery

[out_of_range_results.ipynb](out_of_range_results.ipynb) analyzes the fixed
[target dataset](../../../datasets/out_of_range_512x512/).

The experiment compares four 10,000-secant empirical Christoffel laws, MCS,
and inverse-square sampling under four recovery prompts. It uses ratios 1%
through 5% and five trials, giving 600 reconstructions. Recovery uses weighted
least squares, a unitary Fourier transform, CFG 1, 20 DDIM steps, and
measurement-backprojection initialization. Adam runs for 2,000 iterations,
with learning rate 0.1 until iteration 400 and cosine decay to 0.001.
Christoffel sampling uses $\zeta=1/2$ regularization.

- [Configurations](../../../configs/weighted/out_of_range/)
- [Results and figures](../../../results/weighted/out_of_range/)
- [Saved-data downloads](../../../scripts/release/README.md)

Run one law and recovery prompt from the repository root:

```text
./scripts/weighted/run_main.sh out_of_range [sampling-law] [recovery-prompt]
```

Example:

```bash
./scripts/weighted/run_main.sh out_of_range k2 sunset_beach
./scripts/weighted/run_main.sh out_of_range k2 sunset_beach --dry-run
```

Repeat an interrupted command to reuse completed reconstructions and resume
unfinished optimization from its saved checkpoint. The notebook reads the
saved metrics, images, and traces and writes PDFs to the experiment's figures
folder.
