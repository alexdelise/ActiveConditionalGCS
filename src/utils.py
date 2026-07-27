"""General utility helpers shared across dataset, k-tilde, and reconstruction workflows."""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import random as python_random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .config import ReproConfig


def resolve_ktilde_npz_path(ktilde_dir: str | Path, ktilde_name: str) -> Path:
    """Resolve one K-tilde from the supported artifact directories."""

    root = Path(ktilde_dir)
    filename = f"{str(ktilde_name).strip()}.npz"
    candidates = (
        root / "weighted" / "reference" / filename,
        root / "unweighted" / filename,
        root / filename,
    )
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"K-tilde '{ktilde_name}' was not found; searched: {searched}.")
    physical_matches = {path.resolve() for path in matches}
    if len(physical_matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"K-tilde '{ktilde_name}' is ambiguous across artifact directories: {locations}.")
    # Prefer the namespaced path when a local compatibility symlink points to
    # the same physical artifact.
    return matches[0]


def set_reproducibility(cfg: "ReproConfig") -> Dict[str, Any]:
    """Set all supported RNG sources from a run config and return the applied settings."""

    import numpy as np
    import torch

    python_random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Keep the CuDNN knobs explicit so saved run metadata reflects the active behavior.
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(cfg.deterministic)
    return {
        "seed": int(cfg.seed),
        "torch_deterministic": bool(cfg.deterministic),
        "cudnn_benchmark": bool(cfg.cudnn_benchmark),
    }


def set_seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a single operation such as dataset-item generation."""

    import numpy as np
    import torch

    python_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_env_info() -> Dict[str, Any]:
    """Collect concise environment metadata to save alongside experiment artifacts."""

    import torch

    from .constants import DEVICE

    info: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
    try:
        import diffusers  # type: ignore

        info["diffusers_version"] = diffusers.__version__
    except Exception as exc:  # pragma: no cover
        info["diffusers_version_error"] = repr(exc)
    try:
        import transformers  # type: ignore

        info["transformers_version"] = transformers.__version__
    except Exception as exc:  # pragma: no cover
        info["transformers_version_error"] = repr(exc)
    return info


def json_dump(path: str | Path, payload: Any) -> None:
    """Write JSON to disk using stable formatting and create parent directories as needed."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest for a text string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_empty_cuda_cache() -> None:
    """Clear the CUDA cache when available without failing on CPU-only environments."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Cache cleanup is best-effort and should never break the experiment flow.
        pass
