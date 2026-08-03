# Source Package

The `src` package implements dataset handling, empirical Christoffel
estimation, Fourier sampling, Stable Diffusion generation, latent
reconstruction, metric evaluation, and result serialization.

## Modules

| Module | Purpose |
| --- | --- |
| [config.py](config.py) | Configuration dataclasses, sampling-method names, defaults, and JSON normalization |
| [constants.py](constants.py) | Shared device and numerical constants |
| [datasets.py](datasets.py) | Dataset construction, metadata validation, and fixed-image loading |
| [diffusion.py](diffusion.py) | Pipeline loading, prompt encoding, latent/image conversion, and differentiable DDIM denoising |
| [fft.py](fft.py) | Partial two-dimensional Fourier transforms and adjoints |
| [ktilde.py](ktilde.py) | Empirical Christoffel estimation, probability construction, metadata checks, and artifact loading |
| [metrics.py](metrics.py) | PSNR, SSIM, LPIPS, and image-shape conversion |
| [reconstruction.py](reconstruction.py) | Measurement construction, latent optimization, initialization, metrics, and per-reconstruction artifacts |
| [runner.py](runner.py) | Sampling sweeps, deterministic repeats, resume behavior, and aggregate result tables |
| [sampling.py](sampling.py) | Christoffel, uniform, and inverse-square masks plus weighted and unweighted measurement operators |
| [utils.py](utils.py) | Artifact paths, hashing, reproducibility, environment metadata, JSON output, and CUDA cleanup |

## Reconstruction flow

`runner.py` loads a resolved configuration and fixed dataset, prepares the
requested sampling distribution, and reuses prompt embeddings across the
sampling sweep. For each sampling ratio and repeat, `reconstruction.py`
constructs the mask and measurement operator, optimizes the initial latent
through the complete denoising chain, selects the best recorded latent, and
saves the reconstruction and diagnostics. Completed reconstructions are
loaded on subsequent invocations, allowing an interrupted command to resume.

`metrics.py` computes PSNR, SSIM, LPIPS, and per-pixel MAE for new
reconstructions and their zero-filled inverses. LPIPS is evaluated on CPU
after latent optimization so metric evaluation does not compete with the
diffusion model for GPU memory. Analysis-time LPIPS backfilling remains
available for artifacts created before LPIPS became a standard saved metric.

The sampling laws and reconstruction operators are described in
[../ktilde/README.md](../ktilde/README.md).
