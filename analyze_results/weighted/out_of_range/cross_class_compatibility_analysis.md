# Christoffel-Only Cross/Self Compatibility Notes

## Scope

The primary cross/self analysis contains only:

- the three proper S10000 cross-class numerators against sunset beach
- the sunset/sunset self-difference special case
- the four S10000 self-class Christoffel sampling laws
- the absolute unitary-scaled compatibility quantity

$$
\widetilde\Lambda(c_r,c_s)
=
\max_i
\frac{
\widetilde K(\mathbb F_{c_r}-\mathbb F_{c_{\mathrm{sb}}})(i)
}{
\widetilde\mu_{c_s}(i)
}.
$$

All sampling laws use the selected regularization value, with $\zeta=1/2$
matching the weighted reconstruction experiments. Uniform MCS,
inverse-square, and their reconstruction rows are not part of the cross/self
notebook.

## Compatibility ordering

Within each numerator row, lower $\widetilde\Lambda$ is favorable. The four
Christoffel laws are ordered as follows:

| Cross/self numerator | Sampling-law ordering |
|---|---|
| $\mathbb F_{c_{\mathrm{uc}}}-\mathbb F_{c_{\mathrm{sb}}}$ | `k0`, `k1`, `k2`, `k4` |
| $\mathbb F_{c_{\mathrm{db}}}-\mathbb F_{c_{\mathrm{sb}}}$ | `k2`, `k0`, `k1`, `k4` |
| $\mathbb F_{c_{\mathrm{sb}}}-\mathbb F_{c_{\mathrm{sb}}}$ | `k2`, `k1`, `k0`, `k4` |
| $\mathbb F_{c_{\mathrm{ca}}}-\mathbb F_{c_{\mathrm{sb}}}$ | `k0`, `k2`, `k1`, `k4` |

The corresponding absolute values are:

| Numerator | `k0` | `k1` | `k2` | `k4` |
|---|---:|---:|---:|---:|
| Unconditioned/sunset | 106.26 | 197.44 | 228.57 | 273.63 |
| Daytime/sunset | 84.62 | 99.74 | 71.33 | 148.97 |
| Sunset/sunset | 110.25 | 93.41 | 23.56 | 191.63 |
| Cat/sunset | 84.04 | 124.57 | 113.52 | 190.08 |

The ordering is a worst-frequency coverage statement, not a semantic-distance
ordering. A sampling law ranks well when it assigns sufficient probability to
every frequency where the fixed numerator is large. Consequently, the
self-class law associated with the recovery prompt need not be best for a
cross-class numerator. For example, within-cat variation in `k4` differs from
cat-versus-sunset variation, whose worst Fourier coordinates are covered more
effectively by `k0`.

## Secondary reconstruction diagnostic

The notebook retains performance correlations as a secondary diagnostic using
only Christoffel sampling. With the 400 completed Christoffel-only out-of-range
reconstructions, the agreement-oriented pooled
Spearman correlations are $0.568$ for weighted loss, $0.276$ for PSNR,
$0.250$ for SSIM, $0.129$ for LPIPS, and $0.185$ for pixel MAE. These weak and
inconsistent image-metric associations show that
$\widetilde\Lambda$ should not be described as dictating finite-optimization
reconstruction quality.

For reference, the strongest completed out-of-range Christoffel combinations
are:

| Criterion | Best sampling/recovery pair | Mean | 95% CI |
|---|---|---:|---:|
| Weighted objective | `k0` / daytime beach | 49.53 | [37.05, 62.00] |
| PSNR | `k0` / unconditioned | 19.27 dB | [18.56, 19.98] |
| SSIM | `k0` / unconditioned | 0.9331 | [0.9215, 0.9446] |
| LPIPS | `k0` / sunset beach | 0.5557 | [0.5320, 0.5793] |
| Pixel MAE | `k0` / unconditioned | 0.0778 | [0.0712, 0.0844] |

## Preliminary prompt-matched checkpoint

As of August 17 at 9:14 AM EDT, 76/400 prompt-matched Christoffel
reconstructions are finalized. All `k0`, `k1`, and `k2` combinations have five
completed 1% trials; each `k4` combination has four. A balanced provisional
comparison can therefore use the first four trials for all 16 combinations,
giving 64 rows at $m/n=0.01$.

At this checkpoint, the agreement-oriented Spearman correlations are $0.276$
for weighted loss, $0.494$ for PSNR, $0.297$ for SSIM, $0.541$ for LPIPS, and
$0.274$ for pixel MAE. Only LPIPS reaches the conventional 5% significance
threshold ($p=0.030$), and the study is not sufficiently complete for a final
claim.

The current balanced-checkpoint leaders are `k1`/daytime beach for loss,
`k0`/daytime beach for PSNR, SSIM, and pixel MAE, and `k0`/sunset beach for
LPIPS.

## Reproducible primary analysis

The lambda-only computation is in
[the shared K-tilde analysis notebook](../ktilde/ktilde_analysis.ipynb).
It audits the four S10000 artifacts, displays the four regularized probability
maps, prints the absolute $4\times4$ $\widetilde\Lambda$ table, and exports the
matching heatmap.
