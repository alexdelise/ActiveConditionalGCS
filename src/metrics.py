"""Image-quality metrics and display-shape helpers for reconstruction outputs."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np


def calculate_psnr(image_a: np.ndarray, image_b: np.ndarray, max_value: float = 255.0) -> float:
    """Compute the peak signal-to-noise ratio between two images."""

    # Float conversion prevents unsigned integer wraparound in the residual
    mse = np.mean((np.asarray(image_a, dtype=np.float32) - np.asarray(image_b, dtype=np.float32)) ** 2)
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(max_value / math.sqrt(mse))


def calculate_ssim(
    image_a: np.ndarray,
    image_b: np.ndarray,
    max_value: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute a global SSIM score between two images.

    Args:
        image_a: First image to compare.
        image_b: Second image to compare.
        max_value: Maximum representable pixel value for the metric.
        k1: Standard SSIM stabilizer constant for the luminance term.
        k2: Standard SSIM stabilizer constant for the contrast term.

    Returns:
        The scalar SSIM score.
    """

    # This global form matches the scalar metric used by the original experiments
    x = np.asarray(image_a, dtype=np.float64).ravel()
    y = np.asarray(image_b, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")

    mu_x = float(x.mean())
    mu_y = float(y.mean())
    sigma_x2 = float(((x - mu_x) ** 2).mean())
    sigma_y2 = float(((y - mu_y) ** 2).mean())
    sigma_xy = float(((x - mu_x) * (y - mu_y)).mean())

    # Stabilizers keep constant or nearly constant images numerically defined
    c1 = (k1 * max_value) ** 2
    c2 = (k2 * max_value) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return float(numerator / denominator)


def _lpips_hwc_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image to a finite RGB array in ``[0, 1]``."""

    value = np.asarray(image, dtype=np.float32)
    if value.ndim == 2:
        value = np.repeat(value[:, :, None], 3, axis=2)
    elif value.ndim == 3 and value.shape[0] in {1, 3} and value.shape[-1] not in {1, 3}:
        value = np.transpose(value, (1, 2, 0))
    if value.ndim != 3 or value.shape[-1] not in {1, 3}:
        raise ValueError(f"LPIPS expects an HxW, HxWxC, or CxHxW image, got {value.shape}")
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=2)
    return np.clip(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


@lru_cache(maxsize=2)
def _lpips_model(device_name: str) -> tuple[Any, Any]:
    """Load and cache the standard LPIPS network on one device."""

    import lpips
    import torch

    device = torch.device(device_name)
    # The package's calibrated backbone provides the conventional LPIPS implementation
    model = lpips.LPIPS(net="alex").to(device).eval()
    model.requires_grad_(False)
    return model, device


def calculate_lpips(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    device: str = "cpu",
) -> float:
    """Compute LPIPS between two images whose pixel values lie in ``[0, 1]``."""

    import torch

    x = _lpips_hwc_rgb(image_a)
    y = _lpips_hwc_rgb(image_b)
    if x.shape != y.shape:
        raise ValueError(f"LPIPS image shapes must match, got {x.shape} and {y.shape}")

    model, torch_device = _lpips_model(str(device))
    # LPIPS accepts NCHW tensors normalized to [-1, 1]
    x_tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)
    y_tensor = torch.from_numpy(np.transpose(y, (2, 0, 1))).unsqueeze(0)
    x_tensor = (2.0 * x_tensor - 1.0).to(torch_device)
    y_tensor = (2.0 * y_tensor - 1.0).to(torch_device)
    with torch.inference_mode():
        score = model(x_tensor, y_tensor)
    return float(score.reshape(-1)[0].detach().cpu().item())


def chw_to_hwc_for_display(image_chw: np.ndarray) -> np.ndarray:
    """Convert a channel-first image array into a channel-last display array clipped to `[0, 1]`."""

    # Metric and plotting code share this conversion to avoid layout mismatches
    image_chw = image_chw.astype(np.float32)
    if image_chw.ndim == 2:
        return np.clip(image_chw, 0.0, 1.0)
    if image_chw.ndim == 3 and image_chw.shape[0] == 3:
        return np.clip(np.transpose(image_chw, (1, 2, 0)), 0.0, 1.0)
    if image_chw.ndim == 3:
        return np.clip(image_chw[0], 0.0, 1.0)
    return np.clip(image_chw, 0.0, 1.0)
