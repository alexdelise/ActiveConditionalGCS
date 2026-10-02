# Weighted Experiment Configurations

The main studies are [prompt_matched/](prompt_matched/),
[prompt_mismatched/](prompt_mismatched/), and [out_of_range/](out_of_range/).
Each contains six sampling-law suites. [ablation/](ablation/) contains the
four Christoffel-law suites for each scenario.

All main experiments use five sampling ratios from 1% to 5%, five trials,
a unitary Fourier operator, weighted least squares, and 2,000 Adam iterations.
The learning rate starts at 0.1 and follows cosine decay after iteration 400,
ending at 0.001. Christoffel laws use probability regularization with zeta 0.5.
MCS and inverse-square use their original sampling laws without this mixture.

Main recovery uses CFG 1. The ablation fixes the recovery prompt to
`"sunset beach"` and compares CFG 1, 3, 5, and 7.5. CFG 1 is read from
the compatible main results. Sampling laws remain the estimates generated
with sampling CFG 7.5.

Use the [public launchers](../../scripts/weighted/README.md) to run or inspect
a setting without editing these configurations.
