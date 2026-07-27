# Weighted K-Tilde Artifacts

[reference/](reference/) contains the four fixed S10000 artifacts.
[config_convergence.json](config_convergence.json) defines their build
parameters, and
[config_convergence_trials.json](config_convergence_trials.json) defines the
five independent latent-seed blocks.

Run one convergence job with
[../../scripts/weighted/ktilde_convergence/run_trial.sh](../../scripts/weighted/ktilde_convergence/run_trial.sh):

```bash
./scripts/weighted/ktilde_convergence/run_trial.sh k0 1
```

Final trial artifacts are stored below
[convergence_trials/](convergence_trials/), and scalar traces are stored under
[../../results/weighted/ktilde_convergence/traces/](../../results/weighted/ktilde_convergence/traces/).
