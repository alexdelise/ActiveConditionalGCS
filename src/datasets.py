"""Dataset loading and dataset-building helpers for the repo layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .config import DatasetBuildConfig, GenerationConfig, RuntimeConfig
from .utils import json_dump, set_seed_all, sha256_text


def dataset_directory(datasets_dir: str | Path, dataset_name: str) -> Path:
    """Return the on-disk folder used by a named dataset artifact."""

    directory = Path(datasets_dir) / str(dataset_name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def dataset_index_path(datasets_dir: str | Path, dataset_name: str) -> Path:
    """Return the canonical `dataset_index.json` path for a named dataset artifact."""

    return dataset_directory(datasets_dir, dataset_name) / "dataset_index.json"


def save_preview_png(path: Path, image_chw: torch.Tensor) -> None:
    """Save a channel-first image tensor to disk without changing its spatial resolution."""

    # Import plotting only when an image is actually being written so metadata-only
    # operations do not depend on matplotlib being installed
    import matplotlib.pyplot as plt

    image_np = image_chw.detach().to(dtype=torch.float32, device="cpu").numpy()
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        display = np.clip(np.transpose(image_np, (1, 2, 0)), 0.0, 1.0)
    else:
        display = np.clip(image_np.squeeze(), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), display if display.ndim == 3 else display, cmap=None if display.ndim == 3 else "gray")


def _canonicalize_item_paths(dataset_dir_path: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite stored file paths so copied datasets remain valid inside the current layout."""

    item_id = int(item["item_id"])
    normalized = dict(item)
    # Dataset indexes may have been copied from another absolute path. Rebuild
    # local paths from the dataset folder so the submission is relocatable
    normalized["gt_png_path"] = str(dataset_dir_path / f"gt_{item_id:03d}.png")
    meta_path = dataset_dir_path / f"meta_{item_id:03d}.json"
    if meta_path.exists():
        normalized["meta_json_path"] = str(meta_path)
    return normalized


def load_dataset_index(datasets_dir: str | Path, dataset_name: str) -> Dict[str, Any]:
    """Load a named dataset artifact and normalize its embedded paths.

    Args:
        datasets_dir: Root datasets directory inside the repo.
        dataset_name: Exact dataset folder name to load.

    Returns:
        The normalized dataset index payload.
    """

    index_path = dataset_index_path(datasets_dir, dataset_name)
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' was not found at {index_path}.")

    import json

    with index_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    dataset_dir_path = dataset_directory(datasets_dir, dataset_name)
    dataset["dataset_key"] = str(dataset_name)
    dataset["dataset_path"] = str(dataset_dir_path)
    dataset["items"] = [_canonicalize_item_paths(dataset_dir_path, item) for item in dataset.get("items", [])]
    return dataset


def validate_dataset(dataset: Dict[str, Any], definition: DatasetBuildConfig) -> None:
    """Validate that an existing dataset artifact still matches its catalog definition."""

    items = list(dataset.get("items", []))
    if len(items) != len(definition.prompts):
        raise ValueError(
            f"Dataset '{definition.name}' contains {len(items)} items but the catalog defines "
            f"{len(definition.prompts)} prompts."
        )
    for index, item in enumerate(items):
        if int(item["height"]) != int(definition.height) or int(item["width"]) != int(definition.width):
            raise ValueError(f"Dataset '{definition.name}' contains item {index} with mismatched dimensions.")
        if int(item["truth_num_steps"]) != int(definition.num_steps):
            raise ValueError(f"Dataset '{definition.name}' contains item {index} with mismatched generation steps.")
        if str(item["prompt_text"]) != str(definition.prompts[index]):
            raise ValueError(f"Dataset '{definition.name}' contains item {index} with mismatched prompt text.")
        if float(item.get("guidance_scale", definition.guidance_scale)) != float(definition.guidance_scale):
            raise ValueError(f"Dataset '{definition.name}' contains item {index} with mismatched guidance scale.")


def build_dataset(
    datasets_dir: str | Path,
    definition: DatasetBuildConfig,
    runtime: RuntimeConfig,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Create or reuse a named dataset artifact from the dataset catalog.

    Args:
        datasets_dir: Root datasets directory inside the repo.
        definition: Catalog entry describing the dataset to build.
        runtime: Runtime pipeline settings used while generating dataset images.
        force: When true, rebuild the dataset even if it already exists.

    Returns:
        The normalized dataset index payload for the created or reused dataset.
    """

    # Keep the diffusion dependency on the build path only so loading copied
    # dataset metadata does not require diffusers to be installed
    from .diffusion import encode_prompt, generate_from_latents, load_sd15_pipeline, make_latents, offload_text_encoder

    index_path = dataset_index_path(datasets_dir, definition.name)
    if index_path.exists() and not force:
        # Reuse checked-in dataset artifacts when they match the catalog entry;
        # this keeps reproduction runs from regenerating ground-truth images
        dataset = load_dataset_index(datasets_dir, definition.name)
        validate_dataset(dataset, definition)
        return dataset

    pipe = load_sd15_pipeline(runtime)
    dataset_dir_path = dataset_directory(datasets_dir, definition.name)
    items: List[Dict[str, Any]] = []
    generation = GenerationConfig(
        num_steps=int(definition.num_steps),
        guidance_scale=float(definition.guidance_scale),
    )

    for item_id, prompt_text in enumerate(definition.prompts):
        print(f"Building dataset item {item_id} for '{definition.name}'")
        seed = int(definition.seed + definition.per_item_seed_offset + item_id)
        set_seed_all(seed)
        # Each dataset item is generated from a deterministic seed and prompt so
        # later reconstruction runs can treat the saved image as fixed truth
        prompt_embeddings = encode_prompt(
            pipe,
            str(prompt_text),
            guidance_scale=float(definition.guidance_scale),
        )
        latents_init = make_latents(
            pipe,
            int(definition.height),
            int(definition.width),
            batch_size=1,
            dtype=torch.float32,
        )
        image_chw = generate_from_latents(pipe, latents_init, prompt_embeddings, generation).to(dtype=torch.float32)

        item_record: Dict[str, Any] = {
            "item_id": int(item_id),
            "seed": int(seed),
            "prompt_text": str(prompt_text),
            "height": int(definition.height),
            "width": int(definition.width),
            "truth_num_steps": int(definition.num_steps),
            "guidance_scale": float(definition.guidance_scale),
            "prompt_sha256": sha256_text(prompt_text),
        }

        if definition.save_images:
            # Ground-truth PNGs are the reconstruction target consumed by the
            # Fourier measurement code
            gt_path = dataset_dir_path / f"gt_{item_id:03d}.png"
            save_preview_png(gt_path, image_chw)
            item_record["gt_png_path"] = str(gt_path)

        if definition.save_meta_json:
            meta_path = dataset_dir_path / f"meta_{item_id:03d}.json"
            json_dump(meta_path, item_record)
            item_record["meta_json_path"] = str(meta_path)

        items.append(item_record)

    dataset = {
        "dataset_key": str(definition.name),
        "dataset_path": str(dataset_dir_path),
        "items": items,
    }
    json_dump(index_path, dataset)
    offload_text_encoder(pipe)
    return load_dataset_index(datasets_dir, definition.name)
