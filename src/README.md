# Source Package

The source package contains the SD1.5 diffusion-backprop reconstruction workflow used by the submission.

- `config.py`: dataclasses and JSON loaders for the simplified diffusion-backprop configs.
- `diffusion.py`: SD1.5 pipeline loading, prompt encoding, latent/image conversion, and differentiable denoising.
- `datasets.py`: dataset artifact build and load helpers.
- `ktilde.py`: offline Christoffel/K-tilde estimation, optional per-iteration observation, and validation.
- `sampling.py`: Christoffel/K-tilde FFT sampling patterns plus the measurement operator.
- `reconstruction.py`: diffusion-backprop reconstruction for one item, sampling rate, and repeat.
- `runner.py`: tagged suite execution, result-table writing, and resume behavior.
