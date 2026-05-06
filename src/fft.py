r"""Fourier-domain helpers used by sampling, measurements, and k-tilde estimation.

The partial Fourier helper adapts the CS4ML MRI-Generative-Models
``partialFourier3D`` convention to this repo's 2D PyTorch image setting. CS4ML
is copyright Juan Manuel Cardenas Cardenas and licensed under MIT; see
``THIRD_PARTY_NOTICES.md``.

The measurement code uses the standard unnormalized DFT convention so the sampling
operator can be written as ``A = (1 / sqrt(m)) P_\Omega F``. Concretely:

- the forward transform uses the usual unnormalized FFT, and
- the adjoint uses the unnormalized inverse transform.
"""

from __future__ import annotations

import torch


def fft2_dft(x: torch.Tensor) -> torch.Tensor:
    """Apply the standard unnormalized 2D DFT along the last two dimensions."""

    return torch.fft.fftn(x, dim=(-2, -1), norm="backward")


def ifft2_dft_adjoint(x: torch.Tensor) -> torch.Tensor:
    """Apply the adjoint of :func:`fft2_dft` along the last two dimensions."""

    return torch.fft.ifftn(x, dim=(-2, -1), norm="forward")


def partial_fourier_2d(inds: torch.Tensor, height: int, width: int, x: torch.Tensor, mode: int) -> torch.Tensor:
    """Apply a partial centered 2D Fourier transform or its adjoint.

    Args:
        inds: Flattened Fourier coefficient indices to sample or place back.
        height: Image height in pixels.
        width: Image width in pixels.
        x: Input tensor shaped as `(channels, height * width)` for forward mode or
            `(channels, num_coefficients)` for adjoint mode.
        mode: `1` for forward sampling and `2` for the adjoint operation.

    Returns:
        The sampled Fourier coefficients in forward mode or the reconstructed spatial-domain
        vectors in adjoint mode.
    """

    num_pixels = int(height) * int(width)
    if mode == 1:
        x = x.view(-1, height, width)
        transformed = fft2_dft(x)
        # Sampling indices are defined on the centered spectrum, matching the
        # k-tilde artifacts and mask visualizations.
        transformed = torch.fft.fftshift(transformed, dim=(-2, -1))
        transformed = transformed.view(-1, num_pixels)
        return transformed[..., inds]

    if mode == 2:
        num_coefficients = int(inds.numel())
        dtype = x.dtype if torch.is_complex(x) else torch.complex64
        transformed = torch.zeros(x.shape[:-1] + (num_pixels,), dtype=dtype, device=x.device)
        # Place measured coefficients back into an otherwise-zero Fourier grid;
        # unsampled frequencies remain zero for the adjoint operation.
        transformed[..., inds] = x.view(-1, num_coefficients).to(dtype)
        transformed = transformed.view(-1, height, width)
        transformed = torch.fft.ifftshift(transformed, dim=(-2, -1))
        spatial = ifft2_dft_adjoint(transformed)
        return spatial.view(-1, num_pixels)

    raise ValueError(f"mode must be 1 or 2, got {mode}.")
