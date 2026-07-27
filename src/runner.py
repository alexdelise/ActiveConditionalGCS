"""Experiment runners for tagged sampling sweeps."""

from __future__ import annotations

import gc
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import (
    RunConfig,
    enabled_sampling_method_ids,
    run_config_to_dict,
    sampling_method_folder,
)
from .datasets import load_dataset_index
from .ktilde import load_ktilde_probabilities, regularize_sampling_probabilities
from .utils import collect_env_info, json_dump, safe_empty_cuda_cache, set_reproducibility


OPTIONAL_RESULT_COLUMNS = [
    "pixel_mae",
    "zero_filled_pixel_mae",
]


def csv_safe_value(value: Any) -> Any:
    """Convert a run-record value into a CSV-friendly scalar or JSON string."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of run-record dictionaries to a CSV file without pandas."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    # Include optional columns even when all rows came from older artifacts that
    # predate those metrics, keeping CSV schemas stable across resumed runs.
    for key in OPTIONAL_RESULT_COLUMNS:
        if key not in fieldnames:
            fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe_value(row.get(key)) for key in fieldnames})


def column_array(rows: List[Dict[str, Any]], key: str, dtype) -> np.ndarray:
    """Collect one run-record field into a typed NumPy array."""

    return np.asarray([row[key] for row in rows], dtype=dtype)


def optional_column_array(rows: List[Dict[str, Any]], key: str, dtype, default: Any = np.nan) -> np.ndarray:
    """Collect an optional run-record field, filling legacy missing values."""

    return np.asarray([row.get(key, default) for row in rows], dtype=dtype)


def metric_log_fragment(row: Dict[str, Any], key: str, label: str, precision: int = 5) -> str:
    """Format an optional scalar metric for progress logs."""

    if key not in row:
        return ""
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f" | {label} {value:.{int(precision)}f}"


def sample_run_dir(parent_run_dir: str | Path, *, item_id: int, samp_perc: float, repeat_id: int) -> Path:
    """Return the artifact directory for one sweep leaf."""

    sample_tag = f"samp_{float(samp_perc):.5f}".replace(".", "p")
    repeat_tag = f"rep_{int(repeat_id):02d}"
    return Path(parent_run_dir) / f"item_{int(item_id):03d}" / sample_tag / repeat_tag


def npz_value_to_python(value: np.ndarray) -> Any:
    """Convert a loaded npz field into a Python scalar when possible."""

    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.copy()


def load_display_image(path: str | Path) -> np.ndarray:
    """Load a saved display PNG as a float image in ``[0, 1]``."""

    from PIL import Image

    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    if image.ndim == 2:
        image = image[:, :, None]
    return np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def replace_or_append_summary_value(summary_path: Path, key: str, value: float) -> None:
    """Add or refresh a numeric value in an existing per-run summary."""

    line = f"{key}: {float(value):.6f}\n"
    if not summary_path.is_file():
        return
    lines = summary_path.read_text(encoding="utf-8").splitlines(keepends=True)
    prefix = f"{key}:"
    for idx, existing in enumerate(lines):
        if existing.startswith(prefix):
            lines[idx] = line
            break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = f"{lines[-1]}\n"
        lines.append(line)
    summary_path.write_text("".join(lines), encoding="utf-8")


def backfill_pixel_mae_for_completed_run(
    run_dir: Path,
    row: Dict[str, Any],
    *,
    dataset_item: Dict[str, Any],
    image_height: int,
    image_width: int,
    method_folder: str,
) -> Dict[str, Any]:
    """Compute missing pixel-MAE fields for a skipped legacy run from saved PNGs."""

    missing_pixel_mae = "pixel_mae" not in row
    missing_zf_pixel_mae = "zero_filled_pixel_mae" not in row
    if not missing_pixel_mae and not missing_zf_pixel_mae:
        return row
    if "gt_png_path" not in dataset_item:
        return row

    gt_display = load_display_image(dataset_item["gt_png_path"])
    if gt_display.shape[:2] != (int(image_height), int(image_width)):
        return row
    updates: Dict[str, float] = {}

    if missing_pixel_mae:
        recon_path = run_dir / f"recon_{method_folder}.png"
        if recon_path.is_file():
            recon_display = load_display_image(recon_path)
            if recon_display.shape == gt_display.shape:
                updates["pixel_mae"] = float(np.mean(np.abs(gt_display - recon_display)))

    if missing_zf_pixel_mae:
        zero_filled_path = run_dir / "zero_filled_ifft.png"
        if zero_filled_path.is_file():
            zero_filled_display = load_display_image(zero_filled_path)
            if zero_filled_display.shape == gt_display.shape:
                updates["zero_filled_pixel_mae"] = float(np.mean(np.abs(gt_display - zero_filled_display)))

    if not updates:
        return row

    # When a skipped run predates pixel-MAE logging, refresh the compact npz and
    # text summary in place so downstream analysis sees a complete schema.
    row.update(updates)
    np.savez_compressed(
        str(run_dir / "run_data.npz"),
        **{key: value for key, value in row.items() if isinstance(value, (np.ndarray, np.number, float, int, str))},
    )
    summary_path = run_dir / "run_summary.txt"
    for key, value in updates.items():
        replace_or_append_summary_value(summary_path, key, value)
    return row


def load_completed_run_row(
    run_dir: Path,
    *,
    dataset_item: Optional[Dict[str, Any]] = None,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    method_folder: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load an already-completed per-run row, returning None if incomplete."""

    run_data_path = run_dir / "run_data.npz"
    run_summary_path = run_dir / "run_summary.txt"
    if not run_data_path.is_file() or not run_summary_path.is_file():
        return None
    # A run is considered resumable only when both the compact data and summary
    # marker are present.
    with np.load(str(run_data_path), allow_pickle=False) as payload:
        row = {str(key): npz_value_to_python(payload[key]) for key in payload.files}
    if dataset_item is not None and image_height is not None and image_width is not None and method_folder is not None:
        row = backfill_pixel_mae_for_completed_run(
            run_dir,
            row,
            dataset_item=dataset_item,
            image_height=int(image_height),
            image_width=int(image_width),
            method_folder=str(method_folder),
        )
    return row


def results_root(project_root: str | Path, tag: str) -> Path:
    """Return the results folder for a tagged run."""

    if not str(tag).strip():
        raise ValueError("Runners require a non-empty --tag value.")
    root = Path(project_root) / "results" / str(tag).strip()
    # Create the tag root early so metadata can be written before the heavy model
    # is loaded.
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_dataset_for_run(project_root: str | Path, cfg: RunConfig) -> Dict[str, Any]:
    """Load the exact dataset artifact referenced by a run config and validate its shape."""

    dataset = load_dataset_index(Path(project_root) / "datasets", cfg.dataset.name)
    for item in dataset.get("items", []):
        # Shape checks catch accidental config/dataset mismatches before any GPU
        # work starts.
        if int(item["height"]) != int(cfg.image.height) or int(item["width"]) != int(cfg.image.width):
            raise ValueError(
                f"Dataset '{cfg.dataset.name}' item {int(item['item_id'])} has shape "
                f"{(int(item['height']), int(item['width']))}, expected {(cfg.image.height, cfg.image.width)}."
            )
    if cfg.reconstruction.prompts is not None and len(cfg.reconstruction.prompts) != len(dataset.get("items", [])):
        raise ValueError("reconstruction.prompts must be null or match the dataset size exactly.")
    return dataset


def resolve_ktilde_for_run(project_root: str | Path, cfg: RunConfig) -> Dict[str, Any]:
    """Load the exact k-tilde artifact referenced by a run config and validate its shape."""

    raw_probabilities, metadata, file_path = load_ktilde_probabilities(Path(project_root) / "ktilde", cfg.ktilde.name)
    if int(metadata["height"]) != int(cfg.image.height) or int(metadata["width"]) != int(cfg.image.width):
        # K-tilde maps are resolution-specific because they index flattened FFT
        # coordinates.
        raise ValueError(
            f"K-tilde '{cfg.ktilde.name}' has shape {(metadata['height'], metadata['width'])}, "
            f"expected {(cfg.image.height, cfg.image.width)}."
        )
    probabilities = regularize_sampling_probabilities(
        raw_probabilities,
        float(cfg.sampling.probability_regularization_zeta),
    )

    def probability_summary(values: np.ndarray) -> Dict[str, Any]:
        array = np.asarray(values, dtype="<f8").reshape(-1)
        return {
            "count": int(array.size),
            "sum": float(array.sum()),
            "min": float(array.min()),
            "max": float(array.max()),
            "sha256_float64_le": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    artifact_hasher = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            artifact_hasher.update(chunk)

    return {
        "probabilities": probabilities,
        "raw_probabilities": raw_probabilities,
        "raw_probability_summary": probability_summary(raw_probabilities),
        "effective_probability_summary": probability_summary(probabilities),
        "probability_regularization_zeta": float(cfg.sampling.probability_regularization_zeta),
        "metadata": metadata,
        "path": str(file_path),
        "artifact_sha256": artifact_hasher.hexdigest(),
    }


def reconstruction_prompt(cfg: RunConfig, item_id: int) -> str:
    """Return the prompt used for reconstructing one dataset item."""

    if cfg.reconstruction.prompts is not None:
        return str(cfg.reconstruction.prompts[int(item_id)] or "")
    return str(cfg.reconstruction.prompt or "")


def save_run_metadata(run_root: Path, cfg: RunConfig, dataset: Dict[str, Any], ktilde_info: Optional[Dict[str, Any]]) -> None:
    """Write environment, dataset, k-tilde, and config metadata into the tagged results folder."""

    json_dump(run_root / "env_info.json", collect_env_info())
    json_dump(run_root / "repro_info.json", set_reproducibility(cfg.repro))
    json_dump(run_root / "run_config.json", run_config_to_dict(cfg))
    json_dump(run_root / "dataset_ref.json", dataset)
    if ktilde_info is not None:
        # Save the exact artifact metadata so a result directory is self
        # describing even after files are moved.
        json_dump(
            run_root / "ktilde_ref.json",
            {
                "name": cfg.ktilde.name,
                "path": ktilde_info["path"],
                "artifact_sha256": ktilde_info["artifact_sha256"],
                "metadata": ktilde_info["metadata"],
                "raw_probability_summary": ktilde_info["raw_probability_summary"],
                "effective_probability_summary": ktilde_info["effective_probability_summary"],
                "probability_regularization_zeta": ktilde_info["probability_regularization_zeta"],
                "fft_normalization": cfg.sampling.fft_normalization,
                "weighted_ls": bool(cfg.sampling.weighted_ls),
            },
        )


def run_method(
    project_root: str | Path,
    cfg: RunConfig,
    *,
    tag: str,
    samp_method: int,
    sampling_percentages: Optional[List[float]] = None,
    repeats_per_setting: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one sampling method over the configured sweep grid.

    Args:
        project_root: Refactored repo root that owns the datasets, ktilde, and results folders.
        cfg: Active run config.
        tag: Tagged results folder name passed by the CLI.
        samp_method: Sampling-method id to execute.

    Returns:
        A summary dictionary describing the produced results table.
    """

    # Keep pipeline and reconstruction imports on the execution path so config
    # validation helpers can be imported without pulling in the complete model stack.
    from .diffusion import encode_prompt, load_sd15_pipeline
    from .reconstruction import run_single_reconstruction

    run_root = results_root(project_root, tag)
    dataset = resolve_dataset_for_run(project_root, cfg)
    ktilde_info = resolve_ktilde_for_run(project_root, cfg) if int(samp_method) == 1 else None
    save_run_metadata(run_root, cfg, dataset, ktilde_info)

    method_folder = sampling_method_folder(samp_method)
    # Per-sampler folders let multiple sampling strategies share one suite tag
    # without overwriting leaf artifacts.
    method_root = run_root / method_folder
    method_root.mkdir(parents=True, exist_ok=True)

    execution_sampling_percentages = (
        [float(value) for value in sampling_percentages]
        if sampling_percentages is not None
        else [float(value) for value in cfg.sweep.sampling_perc_list]
    )
    missing_rates = [
        requested
        for requested in execution_sampling_percentages
        if not any(
            abs(requested - configured) <= 1e-12
            for configured in cfg.sweep.sampling_perc_list
        )
    ]
    if missing_rates:
        raise ValueError(
            "Execution sampling percentages must be present in the canonical "
            f"sweep; missing {missing_rates}."
        )
    execution_repeats_per_setting = (
        int(repeats_per_setting)
        if repeats_per_setting is not None
        else int(cfg.sweep.repeats_per_setting)
    )
    if (
        execution_repeats_per_setting <= 0
        or execution_repeats_per_setting > int(cfg.sweep.repeats_per_setting)
    ):
        raise ValueError(
            "Execution repeats must be positive and no greater than the canonical "
            f"repeat count ({cfg.sweep.repeats_per_setting})."
        )

    pipe = load_sd15_pipeline(cfg.runtime)
    all_rows: List[Dict[str, Any]] = []
    prompt_cache: Dict[str, Any] = {}

    try:
        for item in dataset["items"]:
            item_id = int(item["item_id"])
            prompt_text = reconstruction_prompt(cfg, item_id)
            if prompt_text not in prompt_cache:
                # Cache prompt embeddings once per unique recovery prompt; the
                # same embeddings are reused for all sampling rates/repeats.
                prompt_cache[prompt_text] = encode_prompt(
                    pipe,
                    prompt_text,
                    guidance_scale=float(cfg.gen_recon.guidance_scale),
                )
            prompt_embeddings = prompt_cache[prompt_text]

            for samp_perc in execution_sampling_percentages:
                for repeat_id in range(execution_repeats_per_setting):
                    run_dir = sample_run_dir(
                        method_root,
                        item_id=item_id,
                        samp_perc=float(samp_perc),
                        repeat_id=int(repeat_id),
                    )
                    existing_row = load_completed_run_row(
                        run_dir,
                        dataset_item=item,
                        image_height=int(cfg.image.height),
                        image_width=int(cfg.image.width),
                        method_folder=method_folder,
                    )
                    if existing_row is not None:
                        # Resume behavior is idempotent: completed leaves are
                        # loaded into the aggregate tables instead of rerun.
                        all_rows.append(existing_row)
                        print(
                            f"[recon skip ] sampler={method_folder:<6} item={item_id:03d} rep={repeat_id:02d} "
                            f"samp={float(samp_perc):.5f}"
                            f"{metric_log_fragment(existing_row, 'pixel_mae', 'MAE')}"
                            f" | existing result"
                        )
                        continue

                    row = run_single_reconstruction(
                        cfg,
                        pipe,
                        dataset_item=item,
                        samp_method=int(samp_method),
                        samp_perc=float(samp_perc),
                        repeat_id=int(repeat_id),
                        parent_run_dir=str(method_root),
                        prompt_text=prompt_text,
                        prompt_embeddings=prompt_embeddings,
                        probabilities=None if ktilde_info is None else ktilde_info["probabilities"],
                        ktilde_metadata=None if ktilde_info is None else ktilde_info["metadata"],
                    )
                    all_rows.append(row)
                    print(
                        f"[recon done ] sampler={method_folder:<6} item={item_id:03d} rep={repeat_id:02d} "
                        f"samp={float(samp_perc):.5f} | "
                        f"PSNR {float(row['psnr_db']):6.2f} dB | "
                        f"SSIM {float(row['ssim']):.4f}"
                        f"{metric_log_fragment(row, 'pixel_mae', 'MAE')}"
                    )

        # Rebuild the aggregate from all completed canonical leaves. Disjoint
        # rate shards therefore compose into one table without changing the
        # canonical five-rate run configuration.
        all_rows = []
        for item in dataset["items"]:
            for samp_perc in cfg.sweep.sampling_perc_list:
                for repeat_id in range(int(cfg.sweep.repeats_per_setting)):
                    run_dir = sample_run_dir(
                        method_root,
                        item_id=int(item["item_id"]),
                        samp_perc=float(samp_perc),
                        repeat_id=int(repeat_id),
                    )
                    existing_row = load_completed_run_row(
                        run_dir,
                        dataset_item=item,
                        image_height=int(cfg.image.height),
                        image_width=int(cfg.image.width),
                        method_folder=method_folder,
                    )
                    if existing_row is not None:
                        all_rows.append(existing_row)

        results_csv = run_root / f"results_{method_folder}.csv"
        write_results_csv(results_csv, all_rows)
        # Compact npz tables mirror the CSV for notebooks that prefer NumPy
        # loading over CSV parsing.
        compact_payload = {
            "samp_perc": column_array(all_rows, "samp_perc", np.float64),
            "psnr_db": column_array(all_rows, "psnr_db", np.float64),
            "item_id": column_array(all_rows, "item_id", np.int64),
            "repeat_id": column_array(all_rows, "repeat_id", np.int64),
            "samp_method": column_array(all_rows, "samp_method", np.int64),
            "ssim": column_array(all_rows, "ssim", np.float64),
            "pixel_mae": optional_column_array(all_rows, "pixel_mae", np.float64),
            "zero_filled_psnr_db": column_array(all_rows, "zero_filled_psnr_db", np.float64),
            "zero_filled_ssim": column_array(all_rows, "zero_filled_ssim", np.float64),
            "zero_filled_pixel_mae": optional_column_array(all_rows, "zero_filled_pixel_mae", np.float64),
            "final_raw_resid_l2": optional_column_array(all_rows, "final_raw_resid_l2", np.float64),
            "final_weighted_resid_l2": optional_column_array(all_rows, "final_weighted_resid_l2", np.float64),
        }
        np.savez_compressed(str(run_root / f"results_{method_folder}.npz"), **compact_payload)
        return {
            "method_folder": method_folder,
            "run_root": str(run_root),
            "results_csv": str(results_csv),
        }
    finally:
        prompt_cache.clear()
        del prompt_cache, pipe
        gc.collect()
        safe_empty_cuda_cache()


def run_methods(
    project_root: str | Path,
    cfg: RunConfig,
    *,
    tag: str,
    method_ids: List[int],
    sampling_percentages: Optional[List[float]] = None,
    repeats_per_setting: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run a list of sampling methods under a shared tagged results root."""

    seen: set[int] = set()
    outputs: List[Dict[str, Any]] = []
    for method_id in method_ids:
        # De-duplicate method ids so repeated CLI aliases do not rerun a sampler.
        if int(method_id) in seen:
            continue
        seen.add(int(method_id))
        outputs.append(
            run_method(
                project_root,
                cfg,
                tag=tag,
                samp_method=int(method_id),
                sampling_percentages=sampling_percentages,
                repeats_per_setting=repeats_per_setting,
            )
        )
    return outputs


def run_enabled_methods(
    project_root: str | Path,
    cfg: RunConfig,
    *,
    tag: str,
    families: Optional[List[str]] = None,
    sampling_percentages: Optional[List[float]] = None,
    repeats_per_setting: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run the sampling methods enabled in the config, optionally filtered by family."""

    method_ids = enabled_sampling_method_ids(cfg.sampling.methods_enabled, families=families)
    if not method_ids:
        family_text = ",".join(families or [])
        raise ValueError(f"No sampling methods are enabled for families={family_text!r}.")
    return run_methods(
        project_root,
        cfg,
        tag=tag,
        method_ids=method_ids,
        sampling_percentages=sampling_percentages,
        repeats_per_setting=repeats_per_setting,
    )
