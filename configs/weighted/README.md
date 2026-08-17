# Weighted Configurations

- [out_of_range/](out_of_range/) defines the main weighted out-of-range study
- [prompt_matched/](prompt_matched/) defines the main weighted in-range study
- [ablation/](ablation/) defines the two-trial recovery-CFG study
- [diagnostics/](diagnostics/) contains focused optimization diagnostics

Main experiments use a unitary Fourier transform, weighted least squares,
sampling ratios from 1% through 5%, and the S10000 Christoffel artifact bank.
Uniform and inverse-square suites use their own exact sampling probabilities.
