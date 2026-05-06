"""Shared constants for the Stable Diffusion 1.5 FFT reconstruction package."""

from __future__ import annotations

import os

import torch

from .config import MODEL_ID


# Disable noisy progress bars before any pipeline modules are imported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DIFFUSERS_DISABLE_PROGRESS_BAR", "1")

# Use CUDA when it is available because all heavy inference paths are GPU-oriented.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
