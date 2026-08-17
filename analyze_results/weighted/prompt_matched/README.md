# Weighted Prompt-Matched Recovery

This experiment uses the in-range target from
[the sunset-beach dataset](../../../datasets/sunset_beach_signal_sd15_512x512/).
It mirrors the weighted out-of-range experiment exactly except for the target
image. The `sunset_beach` recovery prompt is therefore the prompt-matched
condition.

The design contains four S10000 empirical Christoffel laws, uniform MCS, and
inverse-square sampling; four recovery prompts; sampling ratios from 1% to 5%;
and five trials per cell. Reconstruction uses weighted least squares, the
unitary Fourier operator, $\zeta=1/2$ for Christoffel laws, CFG 1, 20 DDIM
steps, measurement-backprojection initialization, and 2,000 Adam iterations.
The learning rate is $0.1$ through iteration 400 and then decays by a cosine
schedule to $0.001$.

- [Configurations](../../../configs/weighted/prompt_matched/)
- [Launchers](../../../scripts/weighted/prompt_matched/)
- [Results and figures](../../../results/weighted/prompt_matched/)
- [Analysis notebook](prompt_matched_results.ipynb)

Run or resume one shard from the repository root:

```bash
./scripts/weighted/prompt_matched/run_shard.sh \
  <k0|k1|k2|k4|mcs|inverse_square> \
  <unprompted|daytime_beach|sunset_beach|cat>
```

Validate the complete grid with:

```bash
./scripts/weighted/prompt_matched/list_all.sh
```

Rerunning a command skips complete reconstructions and resumes an incomplete
one from its saved latent, Adam moments, and optimization iteration.
