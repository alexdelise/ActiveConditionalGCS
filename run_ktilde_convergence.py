"""Measure convergence of K-tilde estimates against saved S10000 references."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


LEGACY_TRACE_SCHEMA_VERSION = 2
TRIAL_TRACE_SCHEMA_VERSION = 3
TRIAL_MANIFEST_SCHEMA_VERSION = 1
TRACE_METRIC_NAMES = (
    "relative_l2_error",
    "relative_linf_error",
    "lambda_ref_over_mu_m",
    "max_abs_log_mu_ratio",
)


def convergence_output_paths(output_dir: Path, ktilde_name: str) -> tuple[Path, Path]:
    """Return the legacy trace and metadata paths for one convergence experiment."""

    stem = f"{ktilde_name}.convergence"
    return output_dir / f"{stem}.npz", output_dir / f"{stem}.meta.json"


def display_path(path: Path, project_root: Path) -> str:
    """Return a project-relative path when possible, otherwise an absolute path."""

    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one human-readable JSON metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def save_convergence_trace(
    trace_path: Path,
    metadata_path: Path,
    iterations: Sequence[int],
    metric_traces: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Any],
) -> None:
    """Atomically save one compact scalar convergence trace and its metadata."""

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = trace_path.with_name(f".{trace_path.stem}.tmp.npz")
    payload: Dict[str, Any] = {
        "iteration": np.asarray(iterations, dtype=np.int64),
        "meta": json.dumps(dict(metadata), sort_keys=True, default=str),
    }
    for metric_name in TRACE_METRIC_NAMES:
        payload[metric_name] = np.asarray(metric_traces[metric_name], dtype=np.float64)
    np.savez_compressed(str(temporary), **payload)
    temporary.replace(trace_path)
    atomic_json_dump(metadata_path, metadata)


def ktilde_unitary_energy_scale(metadata: Mapping[str, Any], probabilities: np.ndarray) -> float:
    """Return the H*W factor converting stored FFT energies to the unitary convention."""

    height = int(metadata.get("height", 0) or 0)
    width = int(metadata.get("width", 0) or 0)
    if height > 0 and width > 0:
        return float(height * width)
    return float(np.asarray(probabilities, dtype=np.float64).size)


def distribution_from_ktilde(ktilde: np.ndarray) -> np.ndarray:
    """Normalize one K-tilde iterate into its sampling distribution."""

    values = np.asarray(ktilde, dtype=np.float64).reshape(-1)
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(values, dtype=np.float64) / float(values.size)
    return values / total


def max_abs_log_ratio(reference_mu: np.ndarray, current_mu: np.ndarray) -> float:
    """Return max_i |log(reference_mu_i/current_mu_i)|."""

    reference = np.asarray(reference_mu, dtype=np.float64).reshape(-1)
    current = np.asarray(current_mu, dtype=np.float64).reshape(-1)
    positive = (reference > 0.0) & (current > 0.0)
    both_zero = (reference == 0.0) & (current == 0.0)
    if not np.all(positive | both_zero):
        return float("inf")
    if not np.any(positive):
        return 0.0
    return float(np.max(np.abs(np.log(reference[positive] / current[positive])), initial=0.0))


def lambda_ref_over_mu(reference_ktilde_unitary: np.ndarray, current_mu: np.ndarray) -> float:
    """Return max_i K_ref(i)/mu_M(i), using the unitary-scaled reference numerator."""

    reference = np.asarray(reference_ktilde_unitary, dtype=np.float64).reshape(-1)
    current = np.asarray(current_mu, dtype=np.float64).reshape(-1)
    positive = current > 0.0
    if np.any((reference > 0.0) & ~positive):
        return float("inf")
    ratio = np.zeros_like(reference, dtype=np.float64)
    ratio[positive] = reference[positive] / current[positive]
    return float(np.max(ratio, initial=0.0))


def compute_convergence_metrics(
    current_ktilde: np.ndarray,
    *,
    reference_ktilde: np.ndarray,
    reference_ktilde_unitary: np.ndarray,
    reference_probabilities: np.ndarray,
    reference_l2_norm: float,
    reference_linf_norm: float,
    probability_regularization_zeta: float,
) -> Dict[str, float]:
    """Compute all scalar diagnostics saved by the convergence notebooks."""

    from src.ktilde import regularize_sampling_probabilities

    current = np.asarray(current_ktilde, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_ktilde, dtype=np.float64).reshape(-1)
    difference = current - reference
    current_mu_raw = distribution_from_ktilde(current)
    current_mu = regularize_sampling_probabilities(current_mu_raw, probability_regularization_zeta)
    reference_mu = regularize_sampling_probabilities(reference_probabilities, probability_regularization_zeta)
    return {
        "relative_l2_error": float(np.linalg.norm(difference, 2) / reference_l2_norm),
        "relative_linf_error": float(
            np.max(np.abs(difference), initial=0.0) / reference_linf_norm
        ),
        "lambda_ref_over_mu_m": lambda_ref_over_mu(reference_ktilde_unitary, current_mu),
        "max_abs_log_mu_ratio": max_abs_log_ratio(reference_mu, current_mu),
    }


def load_trial_manifest(path: Path) -> Dict[str, Any]:
    """Load and validate the checked-in five-trial experiment manifest."""

    with path.open("r", encoding="utf-8") as handle:
        manifest = dict(json.load(handle))
    if int(manifest.get("schema_version", -1)) != TRIAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported convergence-trial manifest schema: {path}")

    max_samples = int(manifest.get("max_samples", 0))
    metric_every = int(manifest.get("metric_every", 0))
    expected_trials = int(manifest.get("expected_trials", 0))
    zeta = float(manifest.get("probability_regularization_zeta", -1.0))
    if max_samples <= 0:
        raise ValueError("Trial max_samples must be positive.")
    if metric_every <= 0 or max_samples % metric_every != 0:
        raise ValueError("metric_every must be positive and divide max_samples.")
    if expected_trials <= 1:
        raise ValueError("expected_trials must be greater than one.")
    if not 0.0 <= zeta < 1.0:
        raise ValueError("probability_regularization_zeta must lie in [0, 1).")

    priors = dict(manifest.get("priors", {}))
    if not priors:
        raise ValueError("The convergence-trial manifest has no priors.")
    for alias, raw_prior in priors.items():
        prior = dict(raw_prior)
        if not str(alias) or not str(prior.get("role", "")) or not str(prior.get("reference_name", "")):
            raise ValueError(f"Malformed prior entry '{alias}' in {path}.")

    trials = [dict(item) for item in list(manifest.get("trials", []))]
    if len(trials) != expected_trials:
        raise ValueError(f"Expected {expected_trials} trial definitions, found {len(trials)}.")
    trial_ids = [int(item.get("trial", -1)) for item in trials]
    if sorted(trial_ids) != list(range(1, expected_trials + 1)):
        raise ValueError("Trial identifiers must be consecutive integers starting at one.")

    reference_seed = int(manifest.get("reference_seed", -1))
    block_width = 2 * max_samples
    intervals = [(reference_seed, reference_seed + block_width - 1, "reference")]
    seen_seeds: set[int] = set()
    for item in trials:
        trial_id = int(item["trial"])
        seed = int(item.get("seed", -1))
        if seed < 0 or seed in seen_seeds:
            raise ValueError(f"Invalid or duplicate base seed for trial {trial_id}.")
        seen_seeds.add(seed)
        intervals.append((seed, seed + block_width - 1, f"trial {trial_id}"))
    for left_index, (left_start, left_end, left_label) in enumerate(intervals):
        for right_start, right_end, right_label in intervals[left_index + 1 :]:
            if max(left_start, right_start) <= min(left_end, right_end):
                raise ValueError(
                    f"Latent-seed ranges overlap: {left_label} [{left_start}, {left_end}] and "
                    f"{right_label} [{right_start}, {right_end}]."
                )
    return manifest


def resolve_trial(
    manifest: Mapping[str, Any],
    *,
    reference_name: str,
    trial_id: int,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Resolve a canonical reference name and trial number through the manifest."""

    matching_priors = [
        (str(alias), dict(raw_prior))
        for alias, raw_prior in dict(manifest["priors"]).items()
        if str(dict(raw_prior).get("reference_name")) == str(reference_name)
    ]
    if len(matching_priors) != 1:
        raise KeyError(f"Reference '{reference_name}' is not uniquely defined in the trial manifest.")
    alias, prior = matching_priors[0]
    matching_trials = [
        dict(item) for item in list(manifest["trials"]) if int(dict(item)["trial"]) == int(trial_id)
    ]
    if len(matching_trials) != 1:
        raise KeyError(f"Trial {trial_id} is not defined in the trial manifest.")
    return alias, prior, matching_trials[0]


def trial_output_paths(
    project_root: Path,
    manifest: Mapping[str, Any],
    prior: Mapping[str, Any],
    trial: Mapping[str, Any],
    *,
    reference_name: str,
) -> Dict[str, Path | str]:
    """Resolve all output paths for one prompt/trial pair."""

    role = str(prior["role"])
    trial_id = int(trial["trial"])
    seed = int(trial["seed"])
    trial_label = f"trial_{trial_id:02d}"
    artifact_name = f"{reference_name}__trial{trial_id:02d}_seed{seed}"
    artifact_dir = project_root / str(manifest["artifact_root"]) / role / trial_label
    trace_dir = project_root / str(manifest["trace_root"]) / role
    return {
        "artifact_name": artifact_name,
        "artifact": artifact_dir / f"{artifact_name}.npz",
        "artifact_metadata": artifact_dir / f"{artifact_name}.meta.json",
        "trace": trace_dir / f"{trial_label}.convergence.npz",
        "trace_metadata": trace_dir / f"{trial_label}.convergence.meta.json",
    }


def _load_trace_metadata(trace_path: Path) -> Dict[str, Any]:
    """Load embedded trace metadata without trusting a possibly stale sidecar."""

    with np.load(str(trace_path), allow_pickle=False) as payload:
        return dict(json.loads(str(payload["meta"])))


def _validate_existing_trial(
    *,
    paths: Mapping[str, Path | str],
    trial_id: int,
    reference_sha256: str,
    max_samples: int,
    metric_every: int,
) -> bool:
    """Return true only when an existing trial artifact and trace are complete and compatible."""

    artifact_path = Path(paths["artifact"])
    trace_path = Path(paths["trace"])
    if not artifact_path.is_file() or not trace_path.is_file():
        return False
    with np.load(str(trace_path), allow_pickle=False) as payload:
        iterations = np.asarray(payload["iteration"], dtype=np.int64)
        metadata = dict(json.loads(str(payload["meta"])))
        for metric in TRACE_METRIC_NAMES:
            if metric not in payload.files or np.asarray(payload[metric]).shape != iterations.shape:
                return False
    expected_iterations = np.arange(metric_every, max_samples + 1, metric_every, dtype=np.int64)
    return bool(
        metadata.get("complete") is True
        and int(metadata.get("trial", -1)) == int(trial_id)
        and str(metadata.get("reference_sha256", "")) == str(reference_sha256)
        and np.array_equal(iterations, expected_iterations)
    )


def _trial_trace_metadata(
    base_metadata: Mapping[str, Any],
    *,
    iterations: Sequence[int],
    metric_traces: Mapping[str, Sequence[float]],
    complete: bool,
) -> Dict[str, Any]:
    """Build current scalar-trace metadata for an in-progress or complete trial."""

    metadata = dict(base_metadata)
    metadata["complete"] = bool(complete)
    metadata["recorded_points"] = int(len(iterations))
    metadata["completed_samples"] = int(iterations[-1]) if iterations else 0
    if iterations:
        metadata["latest_metrics"] = {
            metric: float(metric_traces[metric][-1]) for metric in TRACE_METRIC_NAMES
        }
    return metadata


def _run_trial_mode(
    *,
    root: Path,
    args: argparse.Namespace,
    catalog_path: Path,
    catalog: Mapping[str, Any],
) -> None:
    """Run one independent S10000 trial against a fixed saved reference."""

    manifest_path = root / args.trial_manifest
    manifest = load_trial_manifest(manifest_path)
    alias, prior, trial = resolve_trial(manifest, reference_name=args.name, trial_id=args.trial)
    definition = catalog[args.name]
    max_samples = int(manifest["max_samples"])
    metric_every = int(manifest["metric_every"])
    zeta = float(manifest["probability_regularization_zeta"])
    if int(definition.max_samples) != max_samples:
        raise ValueError("Reference catalog max_samples does not match the trial manifest.")
    if int(definition.seed) != int(manifest["reference_seed"]):
        raise ValueError("Reference catalog seed does not match the trial manifest.")

    reference_path = root / "ktilde" / "weighted" / "reference" / f"{definition.name}.npz"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Final S10000 reference not found: {reference_path}")
    reference_sha256 = sha256_file(reference_path)
    paths = trial_output_paths(
        root,
        manifest,
        prior,
        trial,
        reference_name=definition.name,
    )

    base_seed = int(trial["seed"])
    final_seed = base_seed + 2 * max_samples - 1
    resolved = {
        "prior_alias": alias,
        "role": str(prior["role"]),
        "trial": int(trial["trial"]),
        "computer": str(trial.get("computer", "")),
        "base_seed": base_seed,
        "final_latent_seed": final_seed,
        "max_samples": max_samples,
        "metric_every": metric_every,
        "probability_regularization_zeta": zeta,
        "reference": display_path(reference_path, root),
        "reference_sha256": reference_sha256,
        "artifact": display_path(Path(paths["artifact"]), root),
        "trace": display_path(Path(paths["trace"]), root),
    }
    if args.dry_run:
        print(json.dumps(resolved, sort_keys=True))
        return

    from src.config import RuntimeConfig
    from src.diffusion import load_sd15_pipeline
    from src.ktilde import (
        build_prompt_schedule,
        estimate_ktilde_christoffel,
        ktilde_metadata,
        load_ktilde_npz,
        save_ktilde_npz,
        validate_ktilde_metadata,
    )
    from src.utils import collect_env_info

    reference_ktilde, reference_probabilities, reference_metadata = load_ktilde_npz(reference_path)
    validate_ktilde_metadata(reference_metadata, definition)

    if _validate_existing_trial(
        paths=paths,
        trial_id=int(trial["trial"]),
        reference_sha256=reference_sha256,
        max_samples=max_samples,
        metric_every=metric_every,
    ) and not args.force:
        print("Completed convergence trial already validates:", paths["trace"])
        return

    trace_path = Path(paths["trace"])
    artifact_path = Path(paths["artifact"])
    existing_trace_complete = (
        trace_path.is_file() and bool(_load_trace_metadata(trace_path).get("complete"))
    )
    if not args.force and (
        existing_trace_complete
        or (artifact_path.is_file() and not trace_path.is_file())
    ):
        raise FileExistsError(
            "Existing trial outputs are inconsistent or incompatible. "
            f"Inspect {artifact_path.parent} and pass --force only to replace this exact trial."
        )
    if trace_path.is_file() and not existing_trace_complete:
        print("Incomplete scalar trace found; restarting this trial from iteration 1:", trace_path)

    reference_ktilde = np.asarray(reference_ktilde, dtype=np.float64).reshape(-1)
    reference_probabilities = np.asarray(reference_probabilities, dtype=np.float64).reshape(-1)
    reference_l2_norm = float(np.linalg.norm(reference_ktilde, 2))
    reference_linf_norm = float(np.max(np.abs(reference_ktilde), initial=0.0))
    if reference_l2_norm <= 0.0 or reference_linf_norm <= 0.0:
        raise ValueError(f"Final reference K-tilde is all zero: {reference_path}")
    unitary_energy_scale = ktilde_unitary_energy_scale(reference_metadata, reference_probabilities)
    reference_ktilde_unitary = reference_ktilde / unitary_energy_scale
    trial_definition = replace(
        definition,
        name=str(paths["artifact_name"]),
        seed=base_seed,
    )
    prompt_schedule = build_prompt_schedule(trial_definition)
    environment = collect_env_info()
    iterations: List[int] = []
    metric_traces: Dict[str, List[float]] = {metric: [] for metric in TRACE_METRIC_NAMES}
    trace_metadata_base: Dict[str, Any] = {
        "artifact_type": "ktilde_convergence_trial_trace",
        "schema_version": TRIAL_TRACE_SCHEMA_VERSION,
        "trial_manifest_schema_version": int(manifest["schema_version"]),
        "trial_manifest_path": display_path(manifest_path, root),
        "catalog_path": display_path(catalog_path, root),
        "ktilde_name": definition.name,
        "trial_artifact_name": str(paths["artifact_name"]),
        "role": str(prior["role"]),
        "prior_alias": alias,
        "trial": int(trial["trial"]),
        "computer": str(trial.get("computer", "")),
        "seed": base_seed,
        "latent_seed_start": base_seed,
        "latent_seed_end": final_seed,
        "max_samples": max_samples,
        "metric_every": metric_every,
        "probability_regularization_zeta": zeta,
        "fft_energy_convention": "unitary metrics from stored backward-normalized FFT energies",
        "unitary_fft_energy_scale": unitary_energy_scale,
        "reference_path": display_path(reference_path, root),
        "reference_sha256": reference_sha256,
        "trial_artifact_path": display_path(artifact_path, root),
        "trace_path": display_path(trace_path, root),
        "formulas": {
            "relative_l2_error": "||K_tilde_M - K_tilde_ref||_2 / ||K_tilde_ref||_2",
            "relative_linf_error": "||K_tilde_M - K_tilde_ref||_inf / ||K_tilde_ref||_inf",
            "lambda_ref_over_mu_m": "max_i K_tilde_ref_unitary(i) / mu_M_zeta(i)",
            "max_abs_log_mu_ratio": "max_i |log(mu_ref_zeta(i) / mu_M_zeta(i))|",
            "mu_M_zeta": "(1-zeta)*mu_M + zeta/n",
        },
        "reference_metadata": reference_metadata,
        "environment": environment,
    }

    def record_metrics(iteration: int, current_ktilde: np.ndarray) -> None:
        if int(iteration) % metric_every != 0:
            return
        metrics = compute_convergence_metrics(
            current_ktilde,
            reference_ktilde=reference_ktilde,
            reference_ktilde_unitary=reference_ktilde_unitary,
            reference_probabilities=reference_probabilities,
            reference_l2_norm=reference_l2_norm,
            reference_linf_norm=reference_linf_norm,
            probability_regularization_zeta=zeta,
        )
        iterations.append(int(iteration))
        for metric, value in metrics.items():
            metric_traces[metric].append(float(value))
        current_metadata = _trial_trace_metadata(
            trace_metadata_base,
            iterations=iterations,
            metric_traces=metric_traces,
            complete=False,
        )
        save_convergence_trace(
            trace_path,
            Path(paths["trace_metadata"]),
            iterations,
            metric_traces,
            current_metadata,
        )
        print(
            f"[ktilde convergence trial] iteration {iteration}/{max_samples} "
            + " ".join(f"{metric}={metrics[metric]:.8e}" for metric in TRACE_METRIC_NAMES),
            flush=True,
        )

    pipe = load_sd15_pipeline(RuntimeConfig())
    final_ktilde, final_probabilities = estimate_ktilde_christoffel(
        trial_definition,
        pipe,
        prompt_schedule,
        iteration_callback=record_metrics,
        print_progress=False,
    )
    expected_iterations = list(range(metric_every, max_samples + 1, metric_every))
    if iterations != expected_iterations:
        raise RuntimeError(
            f"Trial trace has {len(iterations)} points; expected {len(expected_iterations)}."
        )

    artifact_metadata = ktilde_metadata(trial_definition, prompt_schedule)
    artifact_metadata.update(
        {
            "artifact_type": "ktilde_convergence_trial_final",
            "schema_version": TRIAL_TRACE_SCHEMA_VERSION,
            "role": str(prior["role"]),
            "prior_alias": alias,
            "trial": int(trial["trial"]),
            "computer": str(trial.get("computer", "")),
            "latent_seed_start": base_seed,
            "latent_seed_end": final_seed,
            "reference_name": definition.name,
            "reference_path": display_path(reference_path, root),
            "reference_sha256": reference_sha256,
            "probability_regularization_zeta_for_metrics": zeta,
            "fft_energy_convention": "stored backward-normalized FFT energies",
            "environment": environment,
        }
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = artifact_path.with_name(f".{artifact_path.stem}.tmp.npz")
    save_ktilde_npz(
        temporary_artifact,
        np.asarray(final_ktilde, dtype=np.float64),
        np.asarray(final_probabilities, dtype=np.float64),
        artifact_metadata,
    )
    temporary_artifact.replace(artifact_path)
    atomic_json_dump(Path(paths["artifact_metadata"]), artifact_metadata)

    complete_metadata = _trial_trace_metadata(
        trace_metadata_base,
        iterations=iterations,
        metric_traces=metric_traces,
        complete=True,
    )
    complete_metadata["final_artifact_sha256"] = sha256_file(artifact_path)
    save_convergence_trace(
        trace_path,
        Path(paths["trace_metadata"]),
        iterations,
        metric_traces,
        complete_metadata,
    )
    print("Convergence trial ready:", trace_path)
    print("Final trial K-tilde ready:", artifact_path)


def _run_legacy_mode(
    *,
    root: Path,
    args: argparse.Namespace,
    catalog_path: Path,
    catalog: Mapping[str, Any],
) -> None:
    """Preserve the original deterministic one-to-one convergence rerun."""

    from src.config import RuntimeConfig
    from src.diffusion import load_sd15_pipeline
    from src.ktilde import (
        build_prompt_schedule,
        estimate_ktilde_christoffel,
        ktilde_npz_path,
        load_ktilde_npz,
        validate_ktilde_metadata,
    )
    from src.utils import collect_env_info

    definition = catalog[args.name]
    reference_path = ktilde_npz_path(root / "ktilde" / "weighted" / "reference", definition.name)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Final reference K-tilde not found: {reference_path}. "
            f"Build it first with: python build_ktilde.py --config {args.config} --name {args.name}"
        )
    reference_ktilde, reference_probabilities, reference_metadata = load_ktilde_npz(reference_path)
    validate_ktilde_metadata(reference_metadata, definition)
    reference_ktilde = np.asarray(reference_ktilde, dtype=np.float64).reshape(-1)
    reference_probabilities = np.asarray(reference_probabilities, dtype=np.float64).reshape(-1)
    reference_norm = float(np.linalg.norm(reference_ktilde, 2))
    reference_linf_norm = float(np.max(np.abs(reference_ktilde), initial=0.0))
    if reference_norm <= 0.0 or reference_linf_norm <= 0.0:
        raise ValueError(f"Final reference K-tilde is all zero: {reference_path}")
    unitary_energy_scale = ktilde_unitary_energy_scale(reference_metadata, reference_probabilities)
    reference_ktilde_unitary = reference_ktilde / unitary_energy_scale

    output_dir = root / args.output_dir
    trace_path, metadata_path = convergence_output_paths(output_dir, definition.name)
    if trace_path.exists() and not args.force:
        raise FileExistsError(f"Convergence trace already exists: {trace_path}. Pass --force to overwrite it.")

    iterations: List[int] = []
    metric_traces: Dict[str, List[float]] = {metric: [] for metric in TRACE_METRIC_NAMES}

    def record_metrics(iteration: int, current_ktilde: np.ndarray) -> None:
        metrics = compute_convergence_metrics(
            current_ktilde,
            reference_ktilde=reference_ktilde,
            reference_ktilde_unitary=reference_ktilde_unitary,
            reference_probabilities=reference_probabilities,
            reference_l2_norm=reference_norm,
            reference_linf_norm=reference_linf_norm,
            probability_regularization_zeta=0.0,
        )
        iterations.append(int(iteration))
        for metric, value in metrics.items():
            metric_traces[metric].append(float(value))
        if iteration % 10 == 0 or iteration == int(definition.max_samples):
            print(
                f"[ktilde convergence] iteration {iteration}/{definition.max_samples} "
                + " ".join(f"{metric}={metrics[metric]:.8e}" for metric in TRACE_METRIC_NAMES),
                flush=True,
            )

    pipe = load_sd15_pipeline(RuntimeConfig())
    rerun_ktilde, _ = estimate_ktilde_christoffel(
        definition,
        pipe,
        build_prompt_schedule(definition),
        iteration_callback=record_metrics,
        print_progress=False,
    )
    final_difference = np.asarray(rerun_ktilde, dtype=np.float64).reshape(-1) - reference_ktilde
    reference_matches = bool(
        np.allclose(rerun_ktilde, reference_ktilde, rtol=float(args.rtol), atol=float(args.atol))
    )
    metadata: Dict[str, Any] = {
        "artifact_type": "ktilde_convergence_trace",
        "schema_version": LEGACY_TRACE_SCHEMA_VERSION,
        "formula": "||K_tilde_iteration - K_tilde_final||_2 / ||K_tilde_final||_2",
        "formulas": {
            "relative_l2_error": "||K_tilde_iteration - K_tilde_final||_2 / ||K_tilde_final||_2",
            "relative_linf_error": "||K_tilde_iteration - K_tilde_final||_inf / ||K_tilde_final||_inf",
            "lambda_ref_over_mu_m": "max_i K_tilde_final_unitary(i) / mu_iteration(i)",
            "max_abs_log_mu_ratio": "max_i |log(mu_final(i) / mu_iteration(i))|",
        },
        "ktilde_name": definition.name,
        "catalog_path": display_path(catalog_path, root),
        "reference_path": display_path(reference_path, root),
        "trace_path": display_path(trace_path, root),
        "iterations": int(len(iterations)),
        "reference_l2_norm": reference_norm,
        "reference_linf_norm": reference_linf_norm,
        "unitary_fft_energy_scale": unitary_energy_scale,
        "final_relative_l2_error": float(metric_traces["relative_l2_error"][-1]),
        "final_relative_linf_error": float(metric_traces["relative_linf_error"][-1]),
        "final_lambda_ref_over_mu_m": float(metric_traces["lambda_ref_over_mu_m"][-1]),
        "final_max_abs_log_mu_ratio": float(metric_traces["max_abs_log_mu_ratio"][-1]),
        "final_difference_l2_norm": float(np.linalg.norm(final_difference, 2)),
        "final_difference_max_abs": float(np.max(np.abs(final_difference), initial=0.0)),
        "reference_matches_rerun": reference_matches,
        "validation_rtol": float(args.rtol),
        "validation_atol": float(args.atol),
        "reference_metadata": reference_metadata,
        "environment": collect_env_info(),
    }
    save_convergence_trace(trace_path, metadata_path, iterations, metric_traces, metadata)
    print("Convergence trace ready:", trace_path)
    print("Final reference matches rerun:", reference_matches)
    if not reference_matches:
        raise RuntimeError(
            "The convergence rerun did not match the saved final K-tilde within "
            f"rtol={args.rtol} and atol={args.atol}. See {metadata_path}."
        )


def main() -> None:
    """Run either the legacy deterministic replay or one independent trial."""

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Measure K-tilde convergence against a saved S10000 reference.")
    parser.add_argument(
        "--config",
        default="ktilde/weighted/config_convergence.json",
        help="Catalog containing the fixed S10000 reference K-tilde.",
    )
    parser.add_argument("--name", required=True, help="Exact fixed-reference K-tilde name.")
    parser.add_argument(
        "--output-dir",
        default="results/weighted/figures/ktilde_convergence",
        help="Legacy deterministic-trace output directory.",
    )
    parser.add_argument(
        "--trial-manifest",
        default="ktilde/weighted/config_convergence_trials.json",
        help="Checked-in manifest used when --trial is supplied.",
    )
    parser.add_argument("--trial", type=int, help="Run one independent manifest-defined trial.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate one trial without loading SD1.5.")
    parser.add_argument("--force", action="store_true", help="Replace this exact trace/trial output.")
    parser.add_argument("--rtol", type=float, default=1.0e-5, help="Legacy final-reference validation tolerance.")
    parser.add_argument("--atol", type=float, default=1.0e-8, help="Legacy final-reference validation tolerance.")
    args = parser.parse_args()
    if args.dry_run and args.trial is None:
        parser.error("--dry-run is available only with --trial.")

    from src.config import load_ktilde_catalog

    catalog_path = root / args.config
    catalog = load_ktilde_catalog(catalog_path)
    if args.name not in catalog:
        raise KeyError(f"K-tilde '{args.name}' is not defined in {args.config}.")
    if args.trial is None:
        _run_legacy_mode(root=root, args=args, catalog_path=catalog_path, catalog=catalog)
    else:
        _run_trial_mode(root=root, args=args, catalog_path=catalog_path, catalog=catalog)


if __name__ == "__main__":
    main()
