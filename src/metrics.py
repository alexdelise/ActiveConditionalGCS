"""Image-quality metrics and display-shape helpers for reconstruction outputs."""

from __future__ import annotations

import math

import numpy as np


def calculate_psnr(image_a: np.ndarray, image_b: np.ndarray, max_value: float = 255.0) -> float:
    """Compute the peak signal-to-noise ratio between two images."""

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

    x = np.asarray(image_a, dtype=np.float64).ravel()
    y = np.asarray(image_b, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")

    mu_x = float(x.mean())
    mu_y = float(y.mean())
    sigma_x2 = float(((x - mu_x) ** 2).mean())
    sigma_y2 = float(((y - mu_y) ** 2).mean())
    sigma_xy = float(((x - mu_x) * (y - mu_y)).mean())

    c1 = (k1 * max_value) ** 2
    c2 = (k2 * max_value) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return float(numerator / denominator)


def chw_to_hwc_for_display(image_chw: np.ndarray) -> np.ndarray:
    """Convert a channel-first image array into a channel-last display array clipped to `[0, 1]`."""

    image_chw = image_chw.astype(np.float32)
    if image_chw.ndim == 2:
        return np.clip(image_chw, 0.0, 1.0)
    if image_chw.ndim == 3 and image_chw.shape[0] == 3:
        return np.clip(np.transpose(image_chw, (1, 2, 0)), 0.0, 1.0)
    if image_chw.ndim == 3:
        return np.clip(image_chw[0], 0.0, 1.0)
    return np.clip(image_chw, 0.0, 1.0)
