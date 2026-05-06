"""CS and MCS FFT sampling-pattern builders plus the measurement operator."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .config import SAMPLING_METHODS
from .constants import DEVICE
from .fft import partial_fourier_2d


def dc_index(height: int, width: int) -> int:
    """Return the flattened centered-FFT index of the DC coefficient."""

    # FFT masks are built in centered coordinates after fftshift, so the DC
    # coefficient lives at the center pixel of the flattened grid.
    return (int(height) // 2) * int(width) + (int(width) // 2)


def build_sampling_pattern(
    N: int,
    m: int,
    samp_method: int,
    prob: Optional[np.ndarray],
    *,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a CS or MCS sampling pattern for a single sweep point.

    ``samp_method=1`` is Christoffel/K-tilde sampling and requires the
    precomputed probability map. ``samp_method=2`` is the uniform MCS baseline.
    Both always include the DC Fourier coefficient.
    """

    num_items = int(N)
    count = max(1, min(int(m), num_items))
    dc_idx = dc_index(H, W)

    if int(samp_method) == 1:
        if prob is None:
            raise ValueError("samp_method=1 (cs) requires a k-tilde probability map.")
        prob_used = np.asarray(prob, dtype=np.float64)
        if prob_used.shape != (num_items,):
            raise ValueError(f"k-tilde probability map has shape {prob_used.shape}, expected {(num_items,)}.")

        if count == 1:
            indices = np.array([dc_idx], dtype=np.int64)
        else:
            # Always include DC explicitly, then sample the remaining
            # coefficients without replacement from the normalized k-tilde law.
            prob_rest = prob_used.copy()
            prob_rest[dc_idx] = 0.0
            total = float(prob_rest.sum())
            if total > 0.0:
                prob_rest /= total
            else:
                prob_rest = np.ones(num_items, dtype=np.float64)
                prob_rest[dc_idx] = 0.0
                prob_rest /= prob_rest.sum()
            rest = np.random.choice(num_items, size=count - 1, replace=False, p=prob_rest).astype(np.int64)
            indices = np.concatenate(([dc_idx], rest)).astype(np.int64)
        mask = np.zeros(num_items, dtype=np.int8)
        mask[indices] = 1
        return indices, mask, prob_used

    if int(samp_method) == 2:
        # Uniform fallback branch retained for internal comparison/debugging.
        # The public paper scripts launch the Christoffel/K-tilde branch above.
        mask = np.zeros(num_items, dtype=np.int8)
        if count == 1:
            sampled = np.array([dc_idx], dtype=np.int64)
        else:
            pool = np.arange(num_items, dtype=np.int64)
            pool = np.delete(pool, dc_idx)
            remaining = np.random.permutation(pool)[: count - 1]
            sampled = np.concatenate(([dc_idx], remaining)).astype(np.int64)
        mask[sampled] = 1
        indices = np.nonzero(mask)[0].astype(np.int64)
        prob_used = np.ones(num_items, dtype=np.float64) / float(num_items)
        return indices, mask, prob_used

    raise ValueError(f"samp_method must be one of {sorted(SAMPLING_METHODS)}, got {samp_method}.")


class MeasurementOperator:
    """Linear FFT measurement operator for a fixed sampling mask."""

    def __init__(self, inds_np: np.ndarray, N: int, C: int, H: int, W: int):
        """Store the mask geometry and derived scaling constants."""

        self.N = int(N)
        self.C = int(C)
        self.H = int(H)
        self.W = int(W)
        self.inds_t = torch.from_numpy(inds_np).long().to(DEVICE)
        self.m = int(len(inds_np))
        # The forward operator is A = (1 / sqrt(m)) P_Omega F. The adjoint uses
        # the same scalar so zero-filled backprojections are on the same scale.
        self.scale_fwd = 1.0 / math.sqrt(self.m)
        self.scale_adj = self.scale_fwd

    def A(self, image_chw: torch.Tensor) -> torch.Tensor:
        """Apply the forward FFT measurement operator to an image."""

        image_chw = image_chw.to(device=DEVICE, dtype=torch.float32)
        channels, height, width = image_chw.shape
        if (channels, height, width) != (self.C, self.H, self.W):
            raise ValueError(
                f"MeasurementOperator.A expected image shape {(self.C, self.H, self.W)} "
                f"but received {(channels, height, width)}."
            )
        # Flatten only the spatial dimensions; color channels are measured with
        # the same Fourier mask and concatenated afterward.
        image_vec = image_chw.view(self.C, -1)
        sampled = partial_fourier_2d(self.inds_t, self.H, self.W, image_vec, mode=1)
        sampled = self.scale_fwd * sampled
        return sampled.reshape(-1)

    def At(self, measurements_flat: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint FFT measurement operator to a flattened measurement vector."""

        measurements_flat = measurements_flat.to(device=DEVICE)
        measurements_chm = measurements_flat.view(self.C, self.m)
        measurements_chm = self.scale_adj * measurements_chm
        # The adjoint places sampled coefficients back into a centered Fourier
        # grid before applying the inverse transform.
        image_vec = partial_fourier_2d(self.inds_t, self.H, self.W, measurements_chm, mode=2).real
        return image_vec.view(self.C, self.H, self.W)
