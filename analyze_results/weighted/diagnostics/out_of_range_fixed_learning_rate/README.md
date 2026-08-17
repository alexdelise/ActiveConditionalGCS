# Fixed-Learning-Rate Out-of-Range Diagnostic

This diagnostic preserves the earlier weighted out-of-range optimization study
with a fixed learning rate of $0.1$. It is retained for direct comparison with
the scheduled main experiment and is not used as the canonical reconstruction
grid.

- [Configurations](../../../../configs/weighted/diagnostics/out_of_range_fixed_learning_rate/)
- [Launchers](../../../../scripts/weighted/diagnostics/out_of_range_fixed_learning_rate/)
- [Results and figures](../../../../results/weighted/diagnostics/out_of_range_fixed_learning_rate/)
- [Analysis notebook](out_of_range_fixed_learning_rate.ipynb)

Run or resume one shard from the repository root:

```bash
./scripts/weighted/diagnostics/out_of_range_fixed_learning_rate/run_shard.sh \
  <k0|k1|k2|k4|mcs|inverse_square> \
  <unprompted|daytime_beach|sunset_beach|cat>
```
