"""Configuration models and loaders for the SD1.5 reproducibility code."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

SAMPLING_METHODS: Dict[int, str] = {
    1: "cs",
    2: "mcs",
}

# Integer ids are persisted into result tables, while folder names are used on
# disk. Keeping both maps explicit avoids hidden assumptions in analysis code.
METHOD_FOLDER_NAMES: Dict[int, str] = {
    1: "cs",
    2: "mcs",
}

# Accept a few human-friendly aliases at config/CLI boundaries, then convert
# everything back to the canonical integer ids used by the runners.
METHOD_ALIASES: Dict[str, int] = {
    "1": 1,
    "cs": 1,
    "christoffel": 1,
    "2": 2,
    "mcs": 2,
}


def default_methods_enabled() -> Dict[str, bool]:
    """Return the default sampling-method toggle map for run configs."""

    return {name: False for name in SAMPLING_METHODS.values()}


def sampling_method_id(method: int | str) -> int:
    """Resolve a sampling-method identifier from either an integer or a string alias."""

    if isinstance(method, int):
        if method in SAMPLING_METHODS:
            return int(method)
        raise ValueError(f"Unknown sampling method id {method}.")

    token = str(method).strip().lower()
    if token in METHOD_ALIASES:
        return int(METHOD_ALIASES[token])
    raise ValueError(f"Unknown sampling method '{method}'. Expected one of: cs, mcs.")


def sampling_method_name(method: int | str) -> str:
    """Return the canonical long-form sampling-method name for a method identifier."""

    return SAMPLING_METHODS[sampling_method_id(method)]


def sampling_method_folder(method: int | str) -> str:
    """Return the folder label used for results produced by a sampling method."""

    return METHOD_FOLDER_NAMES[sampling_method_id(method)]


def normalize_methods_enabled(raw: Mapping[str, Any] | None) -> Dict[str, bool]:
    """Normalize the run-config method-toggle map into canonical boolean flags."""

    normalized = default_methods_enabled()
    if raw is None:
        return normalized

    # Config files may use aliases, but downstream runners expect canonical
    # method names in the toggle map.
    for key, value in raw.items():
        method_name = sampling_method_name(str(key))
        normalized[method_name] = bool(value)
    return normalized


def enabled_sampling_method_ids(
    methods_enabled: Mapping[str, Any],
    *,
    families: Sequence[str] | None = None,
) -> List[int]:
    """Return the enabled sampling-method ids."""

    family_set = {str(item).strip().lower() for item in (families or [])}
    # The older code had additional sampling families; the submission keeps the
    # classic Fourier-mask family only.
    if family_set.difference({"classic"}):
        raise ValueError("This submission only exposes the classic cs/mcs sampling family.")

    out: List[int] = []
    normalized = normalize_methods_enabled(methods_enabled)
    for method_id, method_name in SAMPLING_METHODS.items():
        if bool(normalized.get(method_name, False)):
            out.append(int(method_id))
    return out


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings for loading and executing the fixed SD1.5 pipeline."""

    torch_dtype: str = "float16"
    attention_slicing: str = ""
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class ImageConfig:
    """Image-shape settings used to validate the run against dataset and k-tilde artifacts."""

    height: int
    width: int


@dataclass(frozen=True)
class DatasetReference:
    """Reference to a named dataset artifact stored under the repo-level datasets folder."""

    name: str


@dataclass(frozen=True)
class KtildeReference:
    """Reference to a named k-tilde artifact stored under the repo-level ktilde folder."""

    name: str


@dataclass(frozen=True)
class GenerationConfig:
    """Reconstruction-generation settings for Stable Diffusion 1.5 runs."""

    num_steps: int
    guidance_scale: float = 7.5
    eta: float = 0.0


@dataclass(frozen=True)
class ReconstructionConfig:
    """Text-prompt settings used during reconstruction for each dataset item."""

    prompt: str = ""
    prompts: Optional[List[str]] = None


@dataclass(frozen=True)
class ReconstructionSolverConfig:
    """Parameters for repeated end-to-end backprop through the complete denoising chain."""

    sigma_y: float = 0.0
    init_from_meas_backproj: bool = False
    backproj_init_strength: float = 1.0
    outer_iterations: int = 25
    learning_rate: float = 5e-2
    lr_schedule: str = "constant"
    lr_warmup_iterations: int = 0
    lr_min_factor: float = 0.0
    latent_l2_penalty: float = 0.0
    normalize_grad: bool = False
    grad_clip: float = 0.0
    early_stop_patience: int = 0
    early_stop_min_rel_improvement: float = 0.0
    checkpoint_denoiser: bool = True
    log_every: int = 1


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling-method toggles used by experiment runners."""

    weighted_ls: bool
    methods_enabled: Dict[str, bool] = field(default_factory=default_methods_enabled)


@dataclass(frozen=True)
class SweepConfig:
    """Sampling-percentage sweep settings shared by all sampling-method runners."""

    sampling_perc_list: List[float]
    repeats_per_setting: int
    save_per_run_artifacts: bool


@dataclass(frozen=True)
class OptimConfig:
    """Adam moment settings reused by latent-space optimization."""

    nb_epochs: int = 1000
    lr_scale: float = 0.01
    adam_stepsize: float = 100.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    early_stop_loss: float = 1e-8
    log_every: int = 10


@dataclass(frozen=True)
class ReproConfig:
    """Randomness and determinism settings for reproducible runs."""

    seed: int
    deterministic: bool
    cudnn_benchmark: bool


@dataclass(frozen=True)
class OutputConfig:
    """Artifact-saving settings used by the runners."""

    save_mat: bool
    save_npz: bool
    save_json: bool
    save_images: bool
    plot_images: bool


@dataclass(frozen=True)
class RunConfig:
    """Top-level experiment config used by the CS, MCS, and run-all runners."""

    image: ImageConfig
    dataset: DatasetReference
    ktilde: KtildeReference
    runtime: RuntimeConfig
    gen_recon: GenerationConfig
    reconstruction: ReconstructionConfig
    reconstruction_solver: ReconstructionSolverConfig
    sampling: SamplingConfig
    sweep: SweepConfig
    optim: OptimConfig
    repro: ReproConfig
    output: OutputConfig


@dataclass(frozen=True)
class DatasetBuildConfig:
    """Configuration entry used to build or validate a dataset artifact."""

    name: str
    height: int
    width: int
    num_steps: int
    seed: int
    per_item_seed_offset: int
    prompts: List[str]
    guidance_scale: float = 7.5
    save_images: bool = True
    save_meta_json: bool = True


@dataclass(frozen=True)
class KtildeBuildConfig:
    """Configuration entry used to build or validate a named k-tilde artifact."""

    name: str
    height: int
    width: int
    num_steps: int
    seed: int
    max_samples: int
    guidance_scale: float = 7.5
    prompt: str = ""
    prompt_bank: Optional[List[str]] = None
    prompt_template: str = ""
    pair_same_prompt: bool = True


def _construct(cls, payload: Dict[str, Any]):
    """Instantiate a dataclass from a plain dictionary payload."""

    return cls(**payload)


def _normalize_runtime(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply defaults to the runtime config before dataclass construction."""

    normalized = dict(payload)
    normalized.setdefault("torch_dtype", "float16")
    normalized.setdefault("attention_slicing", "")
    normalized.setdefault("gradient_checkpointing", False)
    return normalized


def _normalize_guidance_scale(value: Any, *, field_name: str) -> float:
    """Normalize an SD-style guidance scale."""

    guidance_scale = float(7.5 if value is None else value)
    if guidance_scale < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return guidance_scale


def from_run_dict(payload: Dict[str, Any]) -> RunConfig:
    """Build a run config from a JSON dictionary."""

    normalized = dict(payload)
    normalized["runtime"] = _normalize_runtime(dict(normalized.get("runtime", {})))

    # Normalize nested dictionaries before dataclass construction so copied
    # configs can omit optional fields without changing runtime behavior.
    sampling_payload = dict(normalized.get("sampling", {}))
    sampling_payload.setdefault("weighted_ls", False)
    sampling_payload["methods_enabled"] = normalize_methods_enabled(sampling_payload.get("methods_enabled"))
    normalized["sampling"] = sampling_payload

    gen_recon_payload = dict(normalized.get("gen_recon", {}))
    gen_recon_payload["guidance_scale"] = _normalize_guidance_scale(
        gen_recon_payload.get("guidance_scale", 7.5),
        field_name="gen_recon.guidance_scale",
    )
    gen_recon_payload["eta"] = float(gen_recon_payload.get("eta", 0.0))
    normalized["gen_recon"] = gen_recon_payload

    reconstruction_payload = dict(normalized.get("reconstruction", {}))
    reconstruction_payload["prompt"] = str(reconstruction_payload.get("prompt", "") or "")
    prompts = reconstruction_payload.get("prompts")
    reconstruction_payload["prompts"] = None if prompts is None else ["" if item is None else str(item) for item in prompts]
    # Store only the supported reconstruction prompt fields. This strips legacy
    # keys from old configs while preserving the actual prompt schedule.
    normalized["reconstruction"] = {
        "prompt": reconstruction_payload["prompt"],
        "prompts": reconstruction_payload["prompts"],
    }

    if "dc_methods" in normalized:
        raise ValueError("Run configs now use 'reconstruction_solver' instead of 'dc_methods'.")
    solver_payload = dict(normalized.get("reconstruction_solver", {}))
    normalized["reconstruction_solver"] = {
        "sigma_y": float(solver_payload.get("sigma_y", 0.0)),
        "init_from_meas_backproj": bool(solver_payload.get("init_from_meas_backproj", False)),
        "backproj_init_strength": float(solver_payload.get("backproj_init_strength", 1.0)),
        "outer_iterations": int(solver_payload.get("outer_iterations", 25)),
        "learning_rate": float(solver_payload.get("learning_rate", 5e-2)),
        "lr_schedule": str(solver_payload.get("lr_schedule", solver_payload.get("learning_rate_schedule", "constant"))),
        "lr_warmup_iterations": int(solver_payload.get("lr_warmup_iterations", solver_payload.get("lr_warmup_iters", 0))),
        "lr_min_factor": float(solver_payload.get("lr_min_factor", solver_payload.get("minimum_learning_rate_factor", 0.0))),
        "latent_l2_penalty": float(solver_payload.get("latent_l2_penalty", 0.0)),
        "normalize_grad": bool(solver_payload.get("normalize_grad", False)),
        "grad_clip": float(solver_payload.get("grad_clip", 0.0)),
        "early_stop_patience": int(solver_payload.get("early_stop_patience", 0)),
        "early_stop_min_rel_improvement": float(solver_payload.get("early_stop_min_rel_improvement", 0.0)),
        "checkpoint_denoiser": bool(solver_payload.get("checkpoint_denoiser", True)),
        "log_every": int(solver_payload.get("log_every", 1)),
    }

    output_payload = dict(normalized.get("output", {}))
    # Keep artifact saving enabled by default because analysis notebooks expect
    # both compact result tables and per-run files.
    output_payload.setdefault("save_mat", False)
    output_payload.setdefault("save_npz", True)
    output_payload.setdefault("save_json", True)
    output_payload.setdefault("save_images", True)
    output_payload.setdefault("plot_images", False)
    normalized["output"] = output_payload

    sweep_payload = dict(normalized.get("sweep", {}))
    sweep_payload.setdefault("save_per_run_artifacts", True)
    normalized["sweep"] = sweep_payload

    optim_payload = dict(normalized.get("optim", {}))
    optim_payload.setdefault("nb_epochs", 1000)
    optim_payload.setdefault("lr_scale", 0.01)
    optim_payload.setdefault("adam_stepsize", 100.0)
    optim_payload.setdefault("adam_beta1", 0.9)
    optim_payload.setdefault("adam_beta2", 0.999)
    optim_payload.setdefault("adam_eps", 1e-8)
    optim_payload.setdefault("early_stop_loss", 1e-8)
    optim_payload.setdefault("log_every", 10)
    normalized["optim"] = optim_payload

    return RunConfig(
        image=_construct(ImageConfig, dict(normalized["image"])),
        dataset=_construct(DatasetReference, dict(normalized["dataset"])),
        ktilde=_construct(KtildeReference, dict(normalized["ktilde"])),
        runtime=_construct(RuntimeConfig, normalized["runtime"]),
        gen_recon=_construct(GenerationConfig, dict(normalized["gen_recon"])),
        reconstruction=_construct(ReconstructionConfig, normalized["reconstruction"]),
        reconstruction_solver=_construct(ReconstructionSolverConfig, normalized["reconstruction_solver"]),
        sampling=_construct(SamplingConfig, normalized["sampling"]),
        sweep=_construct(SweepConfig, normalized["sweep"]),
        optim=_construct(OptimConfig, dict(normalized["optim"])),
        repro=_construct(ReproConfig, dict(normalized["repro"])),
        output=_construct(OutputConfig, normalized["output"]),
    )


def _build_dataset_entry(name: str, payload: Mapping[str, Any]) -> DatasetBuildConfig:
    """Build a single dataset-catalog entry from its JSON payload."""

    return DatasetBuildConfig(
        name=str(name),
        height=int(payload["height"]),
        width=int(payload["width"]),
        num_steps=int(payload["num_steps"]),
        seed=int(payload["seed"]),
        per_item_seed_offset=int(payload.get("per_item_seed_offset", 0)),
        prompts=[str(item) for item in payload.get("prompts", [])],
        guidance_scale=_normalize_guidance_scale(
            payload.get("guidance_scale", 7.5),
            field_name=f"datasets.{name}.guidance_scale",
        ),
        save_images=bool(payload.get("save_images", True)),
        save_meta_json=bool(payload.get("save_meta_json", True)),
    )


def _build_ktilde_entry(name: str, payload: Mapping[str, Any]) -> KtildeBuildConfig:
    """Build a single k-tilde-catalog entry from its JSON payload."""

    # A prompt bank lets one k-tilde artifact average over a controlled prompt
    # set instead of a single text condition.
    prompt_bank_raw = payload.get("prompt_bank")
    prompt_bank = None if prompt_bank_raw is None else [str(item) for item in prompt_bank_raw]
    return KtildeBuildConfig(
        name=str(name),
        height=int(payload["height"]),
        width=int(payload["width"]),
        num_steps=int(payload["num_steps"]),
        seed=int(payload["seed"]),
        max_samples=int(payload["max_samples"]),
        guidance_scale=_normalize_guidance_scale(
            payload.get("guidance_scale", 7.5),
            field_name=f"ktilde.{name}.guidance_scale",
        ),
        prompt=str(payload.get("prompt", "")),
        prompt_bank=prompt_bank,
        prompt_template=str(payload.get("prompt_template", "")),
        pair_same_prompt=bool(payload.get("pair_same_prompt", True)),
    )


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a UTF-8 JSON file from disk."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_config(path: str | Path) -> RunConfig:
    """Load the run config used by the experiment runners."""

    return from_run_dict(load_json(path))


def load_dataset_catalog(path: str | Path) -> Dict[str, DatasetBuildConfig]:
    """Load the dataset catalog used by the dataset builder."""

    payload = load_json(path)
    datasets_payload = dict(payload.get("datasets", {}))
    return {
        str(name): _build_dataset_entry(str(name), dict(entry))
        for name, entry in datasets_payload.items()
    }


def load_ktilde_catalog(path: str | Path) -> Dict[str, KtildeBuildConfig]:
    """Load the k-tilde catalog used by the k-tilde builder."""

    payload = load_json(path)
    ktilde_payload = dict(payload.get("ktilde", {}))
    return {
        str(name): _build_ktilde_entry(str(name), dict(entry))
        for name, entry in ktilde_payload.items()
    }


def save_json(path: str | Path, payload: Any) -> None:
    """Write a Python object to a UTF-8 JSON file with stable formatting."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_config_to_dict(cfg: RunConfig) -> Dict[str, Any]:
    """Convert a run config to a plain dictionary for artifact serialization."""

    return asdict(cfg)
