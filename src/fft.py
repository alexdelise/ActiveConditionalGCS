r"""Fourier-domain helpers used by sampling, measurements, and k-tilde estimation.

The partial Fourier helper adapts the CS4ML MRI-Generative-Models
``partialFourier3D`` convention to this repo's 2D PyTorch image setting. CS4ML
is copyright Juan Manuel Cardenas Cardenas and licensed under MIT; see
``THIRD_PARTY_NOTICES.md``.

Legacy measurements use the standard unnormalized DFT convention. Weighted
experiments may instead request the unitary convention used in the theory:

- ``backward`` uses the usual unnormalized FFT and its true adjoint, and
- ``ortho`` uses the unitary FFT, whose inverse is also its adjoint.
"""

from __future__ import annotations

import torch


FFT_NORMALIZATIONS = {"backward", "ortho"}


def validate_fft_normalization(normalization: str) -> str:
    """Return a supported PyTorch FFT normalization token."""

    token = str(normalization).strip().lower()
    if token not in FFT_NORMALIZATIONS:
        raise ValueError(f"FFT normalization must be one of {sorted(FFT_NORMALIZATIONS)}, got {normalization!r}.")
    return token


def fft2_dft(x: torch.Tensor, normalization: str = "backward") -> torch.Tensor:
    """Apply a 2D DFT along the last two dimensions."""

    return torch.fft.fftn(x, dim=(-2, -1), norm=validate_fft_normalization(normalization))


def ifft2_dft_adjoint(x: torch.Tensor, normalization: str = "backward") -> torch.Tensor:
    """Apply the adjoint of :func:`fft2_dft` along the last two dimensions."""

    token = validate_fft_normalization(normalization)
    inverse_norm = "forward" if token == "backward" else "ortho"
    return torch.fft.ifftn(x, dim=(-2, -1), norm=inverse_norm)


def partial_fourier_2d(
    inds: torch.Tensor,
    height: int,
    width: int,
    x: torch.Tensor,
    mode: int,
    *,
    normalization: str = "backward",
) -> torch.Tensor:
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
        transformed = fft2_dft(x, normalization=normalization)
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
        spatial = ifft2_dft_adjoint(transformed, normalization=normalization)
        return spatial.view(-1, num_pixels)

    raise ValueError(f"mode must be 1 or 2, got {mode}.")
