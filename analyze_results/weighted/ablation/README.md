# Weighted Recovery-CFG Ablation

This two-trial ablation recreates the paper's recovery-conditioning study for
the weighted unitary reconstruction problem. It covers prompt-matched,
prompt-mismatched, and out-of-range targets. The sampling laws are the four
S10000 Christoffel estimates generated with sampling CFG 7.5; the ablated
quantity is the recovery condition.

Each scenario uses recovery lines `unconditioned`, `cfg1`, `cfg1p5`,
`cfg3`, `cfg5`, and `cfg7p5`, sampling ratios from 1% through 5%, and two
trials. CFG 1 is text-conditioned without classifier-free amplification and is
distinct from the empty-prompt unconditioned control. Reconstruction uses
weighted least squares, the unitary Fourier operator, $\zeta=1/2$, 20 DDIM
steps, 2,000 Adam iterations, and the same learning-rate schedule as the main
weighted experiments.

- [Configurations](../../../configs/weighted/ablation/)
- [Launchers](../../../scripts/weighted/ablation/)
- [Results and per-scenario figures](../../../results/weighted/ablation/)
- [Prompt-matched notebook](prompt_matched_cfg_ablation.ipynb)
- [Prompt-mismatched notebook](prompt_mismatched_cfg_ablation.ipynb)
- [Out-of-range notebook](out_of_range_cfg_ablation.ipynb)

Validate all 144 commands without launching work:

```bash
./scripts/weighted/ablation/list_all.sh
```

Run or resume one shard:

```bash
./scripts/weighted/ablation/run_shard.sh \
  <prompt_matched|prompt_mismatched|out_of_range> \
  <k0|k1|k2|k4> \
  <unconditioned|cfg1|cfg1p5|cfg3|cfg5|cfg7p5> \
  <first3|last2|all>
```

The complete design contains 720 reconstructions, or 240 per scenario.
