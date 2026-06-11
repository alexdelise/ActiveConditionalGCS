"""Measure convergence of a k-tilde estimate against a saved final reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def convergence_output_paths(output_dir: Path, ktilde_name: str) -> tuple[Path, Path]:
    """Return the trace and metadata paths for one convergence experiment."""

    stem = f"{ktilde_name}.convergence"
    return output_dir / f"{stem}.npz", output_dir / f"{stem}.meta.json"


def save_convergence_trace(
    trace_path: Path,
    metadata_path: Path,
    iterations: List[int],
    relative_l2_error: List[float],
    metadata: Dict[str, Any],
) -> None:
    """Save one compact convergence trace and its human-readable metadata."""

    from src.utils import json_dump

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(trace_path),
        iteration=np.asarray(iterations, dtype=np.int64),
        relative_l2_error=np.asarray(relative_l2_error, dtype=np.float64),
        meta=json.dumps(metadata),
    )
    json_dump(metadata_path, metadata)


def main() -> None:
    """Rerun Algorithm 1 and compare every iterate with a saved final k-tilde."""

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Measure k-tilde convergence against a saved final reference.")
    parser.add_argument(
        "--config",
        type=str,
        default="ktilde/config_convergence.json",
        help="Path to the catalog containing the final reference k-tilde.",
    )
    parser.add_argument("--name", type=str, required=True, help="Exact final reference k-tilde name to rerun.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis/ktilde_convergence",
        help="Directory for compact convergence traces.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing convergence trace.")
    parser.add_argument("--rtol", type=float, default=1.0e-5, help="Relative tolerance for final-reference validation.")
    parser.add_argument("--atol", type=float, default=1.0e-8, help="Absolute tolerance for final-reference validation.")
    args = parser.parse_args()

    from src.config import RuntimeConfig, load_ktilde_catalog
    from src.diffusion import load_sd15_pipeline
    from src.ktilde import (
        build_prompt_schedule,
        estimate_ktilde_christoffel,
        ktilde_npz_path,
        load_ktilde_npz,
        validate_ktilde_metadata,
    )
    from src.utils import collect_env_info

    catalog_path = root / args.config
    catalog = load_ktilde_catalog(catalog_path)
    if args.name not in catalog:
        raise KeyError(f"K-tilde '{args.name}' is not defined in {args.config}.")
    definition = catalog[args.name]

    reference_path = ktilde_npz_path(root / "ktilde", definition.name)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Final reference k-tilde not found: {reference_path}. "
            f"Build it first with: python build_ktilde.py --config {args.config} --name {args.name}"
        )
    reference_ktilde, _, reference_metadata = load_ktilde_npz(reference_path)
    validate_ktilde_metadata(reference_metadata, definition)
    reference_norm = float(np.linalg.norm(reference_ktilde, 2))
    if reference_norm == 0.0:
        raise ValueError(f"Final reference k-tilde is all zero: {reference_path}")

    output_dir = root / args.output_dir
    trace_path, metadata_path = convergence_output_paths(output_dir, definition.name)
    if trace_path.exists() and not args.force:
        raise FileExistsError(f"Convergence trace already exists: {trace_path}. Pass --force to overwrite it.")

    iterations: List[int] = []
    relative_l2_error: List[float] = []

    def record_relative_error(iteration: int, current_ktilde: np.ndarray) -> None:
        iterations.append(int(iteration))
        relative_l2_error.append(float(np.linalg.norm(current_ktilde - reference_ktilde, 2) / reference_norm))

    pipe = load_sd15_pipeline(RuntimeConfig())
    rerun_ktilde, _ = estimate_ktilde_christoffel(
        definition,
        pipe,
        build_prompt_schedule(definition),
        iteration_callback=record_relative_error,
    )

    final_difference = rerun_ktilde - reference_ktilde
    reference_matches = bool(np.allclose(rerun_ktilde, reference_ktilde, rtol=float(args.rtol), atol=float(args.atol)))
    metadata: Dict[str, Any] = {
        "artifact_type": "ktilde_convergence_trace",
        "formula": "||K_tilde_iteration - K_tilde_final||_2 / ||K_tilde_final||_2",
        "ktilde_name": definition.name,
        "catalog_path": str(catalog_path.relative_to(root)),
        "reference_path": str(reference_path.relative_to(root)),
        "trace_path": str(trace_path.relative_to(root)),
        "iterations": int(len(iterations)),
        "reference_l2_norm": reference_norm,
        "final_relative_l2_error": float(relative_l2_error[-1]),
        "final_difference_l2_norm": float(np.linalg.norm(final_difference, 2)),
        "final_difference_max_abs": float(np.max(np.abs(final_difference), initial=0.0)),
        "reference_matches_rerun": reference_matches,
        "validation_rtol": float(args.rtol),
        "validation_atol": float(args.atol),
        "reference_metadata": reference_metadata,
        "environment": collect_env_info(),
    }
    save_convergence_trace(trace_path, metadata_path, iterations, relative_l2_error, metadata)
    print("Convergence trace ready:", trace_path)
    print("Final reference matches rerun:", reference_matches)

    if not reference_matches:
        raise RuntimeError(
            "The convergence rerun did not match the saved final k-tilde within "
            f"rtol={args.rtol} and atol={args.atol}. See {metadata_path}."
        )


if __name__ == "__main__":
    main()
