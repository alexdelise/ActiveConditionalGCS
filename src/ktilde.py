"""K-tilde loading and creation helpers for the repo layout.

The Christoffel/K-tilde estimation loop is adapted from the CS4ML
MRI-Generative-Models example by Juan Manuel Cardenas Cardenas, used under the
MIT License. See ``THIRD_PARTY_NOTICES.md`` for the upstream notice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import GenerationConfig, KtildeBuildConfig, MODEL_ID, RuntimeConfig
from .constants import DEVICE
from .fft import partial_fourier_2d
from .utils import json_dump, resolve_ktilde_npz_path, set_seed_all


def ktilde_npz_path(ktilde_dir: str | Path, ktilde_name: str) -> Path:
    """Return the canonical `.npz` path for a named k-tilde artifact."""

    return Path(ktilde_dir) / f"{ktilde_name}.npz"


def ktilde_meta_path(ktilde_dir: str | Path, ktilde_name: str) -> Path:
    """Return the canonical metadata path for a named k-tilde artifact."""

    return Path(ktilde_dir) / f"{ktilde_name}.meta.json"


def expand_prompt_bank(definition: KtildeBuildConfig) -> List[str]:
    """Expand the optional prompt bank used when estimating a conditioned k-tilde."""

    raw_bank = list(definition.prompt_bank or [])
    if not raw_bank:
        return []

    template = str(definition.prompt_template or "").strip()
    prompts: List[str] = []
    for raw_token in raw_bank:
        token = str(raw_token).strip()
        if not token:
            continue
        if template:
            # Support both named and positional templates because older prompt
            # banks used both conventions.
            if "{type}" in template:
                prompt_text = template.format(type=token)
            elif "{}" in template:
                prompt_text = template.format(token)
            else:
                prompt_text = f"{template} {token}"
        else:
            prompt_text = token
        prompt_text = str(prompt_text).strip()
        if prompt_text:
            prompts.append(prompt_text)
    return prompts


def build_prompt_schedule(definition: KtildeBuildConfig) -> List[str]:
    """Build the prompt schedule used over the complete k-tilde Monte Carlo run."""

    prompt_bank = expand_prompt_bank(definition)
    if not prompt_bank:
        return [str(definition.prompt or "") for _ in range(int(definition.max_samples))]

    # Cycle through the prompt bank deterministically so each prompt contributes
    # evenly to the Monte Carlo estimate.
    repetitions = int(definition.max_samples) // len(prompt_bank)
    remainder = int(definition.max_samples) % len(prompt_bank)
    schedule: List[str] = []
    for _ in range(repetitions):
        schedule.extend(prompt_bank)
    if remainder > 0:
        schedule.extend(prompt_bank[:remainder])
    return schedule


def prompt_histogram(prompts: List[str]) -> Dict[str, int]:
    """Count how often each prompt appears in a k-tilde prompt schedule."""

    histogram: Dict[str, int] = {}
    for prompt_text in prompts:
        histogram[str(prompt_text)] = int(histogram.get(str(prompt_text), 0) + 1)
    return histogram


def save_ktilde_npz(path: str | Path, k_tilde: np.ndarray, probabilities: np.ndarray, metadata: Dict[str, Any]) -> None:
    """Write a k-tilde artifact to disk as a compressed NumPy archive."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(file_path),
        K_tilde=k_tilde.astype(np.float64),
        prob=probabilities.astype(np.float64),
        meta=json.dumps(metadata),
    )


def load_ktilde_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a compressed k-tilde archive and return the matrix, probabilities, and metadata."""

    data = np.load(str(path), allow_pickle=False)
    k_tilde = data["K_tilde"].astype(np.float64)
    probabilities = data["prob"].astype(np.float64)
    metadata = json.loads(str(data["meta"]))
    return k_tilde, probabilities, metadata


def ktilde_metadata(definition: KtildeBuildConfig, prompt_schedule: List[str]) -> Dict[str, Any]:
    """Build the canonical metadata payload for a named k-tilde artifact."""

    prompt_bank = expand_prompt_bank(definition)
    return {
        "model_id": MODEL_ID,
        "height": int(definition.height),
        "width": int(definition.width),
        "max_samples": int(definition.max_samples),
        "seed": int(definition.seed),
        "num_steps": int(definition.num_steps),
        "guidance_scale": float(definition.guidance_scale),
        "prompt_mode": "bank" if prompt_bank else "single",
        "prompt": str(definition.prompt or ""),
        "prompt_bank": prompt_bank,
        "prompt_template": str(definition.prompt_template or ""),
        "pair_same_prompt": bool(definition.pair_same_prompt),
        "prompt_schedule_counts": prompt_histogram(prompt_schedule),
    }


@torch.inference_mode()
def estimate_ktilde_christoffel(
    definition: KtildeBuildConfig,
    pipe,
    prompt_schedule: List[str],
    *,
    iteration_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    print_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the Christoffel function used by Christoffel sampling.

    Args:
        definition: K-tilde catalog entry to estimate.
        pipe: Loaded SD1.5 diffusion pipeline.
        prompt_schedule: Prompt sequence used over all Monte Carlo samples.
        iteration_callback: Optional observer called with the one-based iteration
            number and current k-tilde after every trial.
        print_progress: Whether to print the raw k-tilde update delta.

    Returns:
        The estimated k-tilde vector and the normalized sampling probabilities.
    """

    num_pixels = int(definition.height) * int(definition.width)
    k_tilde = np.zeros(num_pixels, dtype=np.float64)
    all_indices = torch.arange(num_pixels, device=DEVICE, dtype=torch.long)

    if len(prompt_schedule) != int(definition.max_samples):
        raise ValueError("Prompt schedule length must equal max_samples.")

    from .diffusion import encode_prompt, generate_from_latents, make_latents

    generation = GenerationConfig(
        num_steps=int(definition.num_steps),
        guidance_scale=float(definition.guidance_scale),
    )
    prompt_cache: Dict[str, Any] = {}

    def sample_image(prompt_text: str, *, seed: int) -> torch.Tensor:
        set_seed_all(int(seed))
        if prompt_text not in prompt_cache:
            # Reusing prompt embeddings avoids repeated text-encoder calls while
            # preserving the same image-generation seeds.
            prompt_cache[prompt_text] = encode_prompt(
                pipe,
                prompt_text,
                guidance_scale=float(definition.guidance_scale),
            )
        latents_init = make_latents(
            pipe,
            int(definition.height),
            int(definition.width),
            batch_size=1,
            dtype=torch.float32,
        )
        return generate_from_latents(pipe, latents_init, prompt_cache[prompt_text], generation).to(dtype=torch.float32)

    for sample_id in range(int(definition.max_samples)):
        previous = k_tilde.copy()
        prompt_i = str(prompt_schedule[sample_id])
        prompt_j = prompt_i if definition.pair_same_prompt else str(prompt_schedule[(sample_id + 1) % int(definition.max_samples)])
        image_i = sample_image(prompt_i, seed=int(definition.seed) + 2 * sample_id)
        image_j = sample_image(prompt_j, seed=int(definition.seed) + 2 * sample_id + 1)

        if image_i.ndim == 2:
            image_i = image_i.unsqueeze(0)
        if image_j.ndim == 2:
            image_j = image_j.unsqueeze(0)

        channels, height, width = image_i.shape
        if height * width != num_pixels:
            raise ValueError("K-tilde image size does not match the configured resolution.")

        diff = (image_i - image_j).reshape(channels, num_pixels).to(dtype=torch.float32, device=DEVICE)
        diff_norm = diff.norm(p=2).item()
        if diff_norm == 0.0:
            if iteration_callback is not None:
                iteration_callback(sample_id + 1, k_tilde)
            continue

        # The Christoffel estimate tracks the largest normalized Fourier energy
        # observed at each coefficient across generated image differences.
        fourier_values = partial_fourier_2d(all_indices, int(definition.height), int(definition.width), diff, mode=1)
        contribution = (fourier_values.abs() ** 2).sum(dim=0).detach().cpu().numpy().astype(np.float64)
        contribution /= diff_norm**2
        k_tilde = np.maximum(previous, contribution)

        if print_progress and (sample_id == 0 or sample_id % 10 == 0 or sample_id == int(definition.max_samples) - 1):
            delta = np.linalg.norm(k_tilde - previous, 2)
            print(sample_id, delta)
        if iteration_callback is not None:
            iteration_callback(sample_id + 1, k_tilde)

    total = np.sum(k_tilde)
    # Degenerate all-zero estimates fall back to a valid uniform distribution
    # instead of producing NaNs.
    probabilities = (np.ones(num_pixels, dtype=np.float64) / num_pixels) if total == 0.0 else (k_tilde / total)
    return k_tilde, probabilities


def validate_ktilde_metadata(metadata: Dict[str, Any], definition: KtildeBuildConfig) -> None:
    """Validate that an existing k-tilde artifact matches its catalog definition."""

    expected = ktilde_metadata(definition, build_prompt_schedule(definition))
    keys = [
        "model_id",
        "height",
        "width",
        "max_samples",
        "seed",
        "num_steps",
        "guidance_scale",
        "prompt_mode",
        "prompt",
        "prompt_bank",
        "prompt_template",
        "pair_same_prompt",
        "prompt_schedule_counts",
    ]
    for key in keys:
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"K-tilde '{definition.name}' does not match the catalog entry for field '{key}'.")


def load_ktilde_probabilities(ktilde_dir: str | Path, ktilde_name: str) -> Tuple[np.ndarray, Dict[str, Any], Path]:
    """Load the exact named k-tilde artifact referenced by a run config."""

    file_path = resolve_ktilde_npz_path(ktilde_dir, ktilde_name)
    _, probabilities, metadata = load_ktilde_npz(file_path)
    return probabilities, metadata, file_path


def regularize_sampling_probabilities(probabilities: np.ndarray, zeta: float) -> np.ndarray:
    """Mix a saved empirical sampling law with the uniform law."""

    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Sampling probability arrays must be nonempty.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Sampling probabilities must be finite and nonnegative.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Sampling probabilities must have positive total mass.")
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Sampling probabilities must sum to one before regularization, got {total:.16g}.")
    mixture = float(zeta)
    if not 0.0 <= mixture < 1.0:
        raise ValueError("zeta must lie in [0, 1).")
    if mixture == 0.0:
        return values.copy()
    return (1.0 - mixture) * values + mixture / float(values.size)


def build_ktilde(
    ktilde_dir: str | Path,
    definition: KtildeBuildConfig,
    runtime: RuntimeConfig,
    *,
    force: bool = False,
) -> Tuple[Path, np.ndarray]:
    """Create or reuse a named k-tilde artifact from the k-tilde catalog.

    Args:
        ktilde_dir: Root k-tilde directory inside the repo.
        definition: Catalog entry describing the k-tilde to build.
        runtime: Runtime pipeline settings used while generating Monte Carlo samples.
        force: When true, rebuild the artifact even if it already exists.

    Returns:
        The path to the k-tilde archive and the normalized sampling probabilities.
    """

    # Keep the diffusion dependency on the build path only so loading copied
    # k-tilde artifacts does not require diffusers to be installed.
    from .diffusion import load_sd15_pipeline

    file_path = ktilde_npz_path(ktilde_dir, definition.name)
    meta_path = ktilde_meta_path(ktilde_dir, definition.name)
    prompt_schedule = build_prompt_schedule(definition)

    if file_path.exists() and not force:
        # Existing artifacts are reused only after metadata validation, which
        # catches stale files from a different prompt/seed/resolution.
        _, probabilities, metadata = load_ktilde_npz(file_path)
        validate_ktilde_metadata(metadata, definition)
        return file_path, probabilities

    pipe = load_sd15_pipeline(runtime)
    k_tilde, probabilities = estimate_ktilde_christoffel(definition, pipe, prompt_schedule)
    metadata = ktilde_metadata(definition, prompt_schedule)
    save_ktilde_npz(file_path, k_tilde, probabilities, metadata)
    json_dump(meta_path, metadata)
    return file_path, probabilities
