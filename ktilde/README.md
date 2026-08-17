# Empirical Christoffel Artifacts and Fourier Operators

[unweighted/](unweighted/) contains the S500 empirical Christoffel estimates
used by the reported reconstruction experiments. [weighted/](weighted/)
contains the four S10000 reference estimates and the independent trials used
for the convergence study.

Build or validate the S500 artifacts with
[../scripts/unweighted/ktilde/main/build_all.sh](../scripts/unweighted/ktilde/main/build_all.sh)
and
[../scripts/unweighted/ktilde/cfg_ablation/build_all.sh](../scripts/unweighted/ktilde/cfg_ablation/build_all.sh).
S10000 reference and convergence launchers are under
[../scripts/weighted/ktilde_convergence/](../scripts/weighted/ktilde_convergence/).

## Empirical Christoffel sampling

For a sampling prompt $c$, the estimator generates independent pairs of
images from the prompt-conditioned model class, normalizes their secants, and
records the largest observed Fourier-coordinate energy. The resulting array
$\widetilde K_c$ approximates the generalized Christoffel function of the
self-difference class $\mathbb F_c-\mathbb F_c$. Its normalized sampling law
is

$$
\widetilde\mu_c(i)
=
\frac{\widetilde K_c(i)}
{\sum_{\ell=1}^{n}\widetilde K_c(\ell)},
\qquad i\in D.
$$

When probability smoothing is enabled, the implementation forms the uniform
mixture

$$
\widetilde\mu_{c,\zeta}(i)
=
(1-\zeta)\widetilde\mu_c(i)+\frac{\zeta}{n}.
$$

The S10000 convergence study uses $\zeta=1/2$. Regularization is applied after
normalizing $\widetilde K_c$, so a stored empirical Christoffel estimate can
be evaluated with different values of $\zeta$ without regenerating secants.

For an ordered cross-class estimate, the optional `pair_prompt` configuration
field fixes the prompt used to generate the second image of every secant. Thus
`prompt = "cat"` and `pair_prompt = "sunset beach"` estimate

$$
\widetilde K(\mathbb F_{c_{\mathrm{ca}}}-\mathbb F_{c_{\mathrm{sb}}}),
$$

with the cat image always generated first and the sunset-beach image second.
The three S10000 cross-class definitions are in
[weighted/config_cross_class_s10000.json](weighted/config_cross_class_s10000.json).

## Reconstruction operators

Let $F_{\mathrm u}$ denote the unitary two-dimensional Fourier transform and
let $I_1,\ldots,I_m$ be sampled frequencies. The theory-aligned weighted
operator is

$$
A_\Omega x
=
\left[
\frac{(F_{\mathrm u}x)(I_j)}
{\sqrt{m\,\mu(I_j)}}
\right]_{j=1}^{m}.
$$

The same spatial mask and weight are applied independently to each color
channel. Christoffel sampling forces the DC coefficient and draws the
remaining frequencies without replacement from the DC-excluded,
renormalized proposal. Reconstruction weights still use the original law
$\mu$, not the proposal used for the conditional draw.

The unweighted experiments use an unweighted Fourier residual. The  weighted reconstruction study uses the unitary operator above with
the probability law associated with each sampling design. Uniform sampling
uses $\mu(i)=1/n$, while inverse-square sampling uses

$$
\mu_{\mathrm{IS}}(i)
\propto
\frac{1}{1+u_i^2+v_i^2}.
$$

## Convergence artifacts

The fixed S10000 estimates serve as references for five independent trials
per prompt. Trial traces record relative $\ell^2$ error and the reference
compatibility statistic every ten iterations. Only each final S10000
empirical Christoffel array is retained. See
[weighted/README.md](weighted/README.md) for commands and paths.
