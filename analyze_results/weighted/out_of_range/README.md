# Weighted Out-of-Range Recovery

This is the main weighted out-of-range reconstruction experiment. It compares
four S10000 empirical Christoffel sampling laws with uniform and inverse-square
Fourier sampling. Every reconstruction uses weighted least squares, a unitary
Fourier operator, CFG 1, 20 DDIM steps, five trials, and sampling ratios
$m/n\in\{0.01,0.02,0.03,0.04,0.05\}$.

Adam runs for 2,000 iterations. The learning rate is $0.1$ through iteration
400 and then follows cosine decay to $0.001$ at iteration 2,000. Early stopping
and gradient clipping are disabled, optimizer checkpoints are written during
the run, and the best latent is saved independently of the final latent.

The complete design contains 600 reconstructions: six sampling laws, four
recovery prompts, five sampling ratios, and five trials. All 600 are complete.

- [Configurations](../../../configs/weighted/out_of_range/)
- [Launchers](../../../scripts/weighted/out_of_range/)
- [Results and figures](../../../results/weighted/out_of_range/)
- [Analysis notebook](out_of_range_results.ipynb)

Run or resume one sampling-law/recovery-prompt shard from the repository root:

```bash
./scripts/weighted/out_of_range/run_shard.sh \
  <k0|k1|k2|k4|mcs|inverse_square> \
  <unprompted|daytime_beach|sunset_beach|cat>
```

Validate all configured shards without launching reconstructions:

```bash
./scripts/weighted/out_of_range/list_all.sh
```

The MCS configurations use an exact backward-only loss divisor to avoid FP16
overflow. The latent gradient is rescaled before Adam, so this safeguard does
not change the recorded objective, gradient, or optimizer update.
