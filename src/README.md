# Source Package

The source package contains the SD1.5 diffusion-backprop reconstruction workflow used by the submission.

- [config.py](config.py): dataclasses and JSON loaders for the
  diffusion-backprop configurations.
- [diffusion.py](diffusion.py): SD1.5 pipeline loading, prompt encoding,
  latent/image conversion, and differentiable denoising.
- [datasets.py](datasets.py): dataset artifact build and load helpers.
- [ktilde.py](ktilde.py): offline Christoffel/K-tilde estimation,
  per-iteration observation, and validation.
- [sampling.py](sampling.py): Fourier sampling patterns and measurement
  operators.
- [reconstruction.py](reconstruction.py): diffusion-backprop reconstruction.
- [runner.py](runner.py): tagged suite execution, result-table writing, and
  resume behavior.
- [utils.py](utils.py): shared path, JSON, reproducibility, environment, and
  CUDA helpers.
