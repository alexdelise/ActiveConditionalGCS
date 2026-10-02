# Weighted In-Range Prompt-Matched Recovery

[prompt_matched_results.ipynb](prompt_matched_results.ipynb) analyzes the fixed
[target dataset](../../../datasets/sunset_beach_signal_sd15_512x512/).

The experiment compares four 10,000-secant empirical Christoffel laws, MCS,
and inverse-square sampling under four recovery prompts. It uses ratios 1%
through 5% and five trials, giving 600 reconstructions. Recovery uses weighted
least squares, a unitary Fourier transform, CFG 1, 20 DDIM steps, and
measurement-backprojection initialization. Adam runs for 2,000 iterations,
with learning rate 0.1 until iteration 400 and cosine decay to 0.001.
Christoffel sampling uses $\zeta=1/2$ regularization.

- [Configurations](../../../configs/weighted/prompt_matched/)
- [Results and figures](../../../results/weighted/prompt_matched/)
- [Saved-data downloads](../../../scripts/release/README.md)

Run one law and recovery prompt from the repository root:

```bash
./scripts/weighted/run_main.sh prompt_matched k2 sunset_beach
./scripts/weighted/run_main.sh prompt_matched k2 sunset_beach --dry-run
```

Repeat an interrupted command to reuse completed reconstructions and resume
unfinished optimization from its saved checkpoint. The notebook reads the
saved metrics, images, and traces and writes PDFs to the experiment's figures
folder.
