"""Prepare figure-reproduction archives without changing the original results."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("prompt_matched", "prompt_mismatched", "out_of_range")
LAWS = ("k0", "k1", "k2", "k4", "mcs", "inverse_square")
RATES = (0.01, 0.02, 0.03, 0.04, 0.05)
RUN_FILES = {
    "run_data.npz", "run_config.json", "dataset_item.json", "sampling_pattern.json",
    "run_summary.txt", "zero_filled_ifft.png",
}
CASE_FILES = {"dataset_ref.json", "ktilde_ref.json", "env_info.json", "run_config.json"}


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_metadata(value: object) -> object:
    """Store dataset and artifact references relative to the checkout."""

    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_metadata(item) for item in value]
    if isinstance(value, str):
        for marker in ("/ActiveConditionalGCS/",):
            if marker in value:
                return value.split(marker, 1)[1]
    return value


def run_files(scenario: str, *, ablation: bool) -> tuple[list[Path], int]:
    """Select the complete reported grid and reject missing or duplicate cells."""

    base = ROOT / "results" / "weighted"
    if ablation:
        base /= "ablation"
    base /= scenario
    selected: set[Path] = set()
    cells: Counter[tuple[object, ...]] = Counter()
    for law in (LAWS[:4] if ablation else LAWS):
        for case in sorted((base / law).iterdir()):
            if not case.is_dir():
                continue
            if ablation and not case.name.endswith(("_cfg3", "_cfg5", "_cfg7p5")):
                continue
            if not ablation and "__recover_" not in case.name:
                continue
            method = "cs" if law in LAWS[:4] else law
            for data_path in sorted((case / method).glob("item_*/samp_*/rep_*/run_data.npz")):
                with np.load(data_path, allow_pickle=False) as data:
                    ratio, repeat = float(data["samp_perc"]), int(data["repeat_id"])
                    if not any(np.isclose(ratio, rate, rtol=0, atol=5e-8) for rate in RATES) or repeat not in range(5):
                        raise ValueError(f"Unexpected sampling ratio or repeat: {data_path}")
                    if int(data["bp_completed_iterations"]) != 2000:
                        raise ValueError(f"Incomplete reconstruction: {data_path}")
                    for metric in ("lpips", "psnr_db", "ssim", "bp_best_loss"):
                        if not np.isfinite(float(data[metric])):
                            raise ValueError(f"Invalid {metric}: {data_path}")
                    cells[(law, case.name, int(data["item_id"]), ratio, repeat)] += 1
                for filename in (*RUN_FILES, f"recon_{method}.png"):
                    path = data_path.parent / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    selected.add(path)
            selected.update(path for path in case.iterdir() if path.is_file() and path.name in CASE_FILES)
    expected = 300 if ablation else 600
    if len(cells) != expected or any(count != 1 for count in cells.values()):
        raise ValueError(f"{scenario}: expected {expected} unique runs, found {len(cells)}")
    counts = Counter((key[0], key[1]) for key in cells)
    if any(count != 25 for count in counts.values()):
        raise ValueError(f"Incomplete law/recovery setting in {scenario}")
    return sorted(selected), expected


def studies() -> list[tuple[str, list[Path], int]]:
    """Return the six reconstruction archives and the Christoffel archive."""

    groups = []
    for ablation in (False, True):
        for scenario in SCENARIOS:
            files, runs = run_files(scenario, ablation=ablation)
            groups.append((f"{'ablation_' if ablation else ''}{scenario}", files, runs))
    files = [
        path for path in (ROOT / "ktilde" / "weighted").rglob("*")
        if path.is_file() and (path.suffix == ".npz" or path.name.endswith(".meta.json") or path.name.startswith("config_"))
    ]
    trace_root = ROOT / "results" / "weighted" / "ktilde" / "traces"
    traces = sorted(trace_root.glob("*/*.convergence.npz"))
    if len(traces) != 20:
        raise ValueError(f"Expected 20 convergence traces, found {len(traces)}")
    for path in traces:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if not meta.get("complete") or not np.array_equal(data["iteration"], np.arange(10, 10001, 10)):
                raise ValueError(f"Incomplete convergence trace: {path}")
        files.append(path)
        files.append(path.with_suffix(".meta.json"))
    # Include fixed targets so the archives can be checked without a prior checkout
    files.extend(path for path in (ROOT / "datasets").rglob("*") if path.is_file() and path.suffix in (".png", ".json"))
    groups.append(("christoffel", sorted(files), 0))
    return groups


def file_bytes(path: Path) -> bytes:
    """Normalize JSON sidecars while preserving checksum-linked numerical artifacts."""

    if path.suffix == ".json":
        return (json.dumps(portable_metadata(json.loads(path.read_text())), indent=2, sort_keys=True) + "\n").encode()
    return path.read_bytes()


def package(name: str, files: list[Path], output: Path) -> dict[str, object]:
    """Create one archive with a checksum manifest for every included file."""

    archive_path = output / f"{name}.tar.gz"
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}; use a new output directory")
    entries = []
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for path in files:
            contents = file_bytes(path)
            relative = path.relative_to(ROOT).as_posix()
            entries.append({"path": relative, "size": len(contents), "sha256": hashlib.sha256(contents).hexdigest()})
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(contents), 0o644
            archive.addfile(info, io.BytesIO(contents))
        contents = (json.dumps({"files": entries}, indent=2) + "\n").encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(contents)
        archive.addfile(info, io.BytesIO(contents))
    return {
        "name": name, "filename": archive_path.name, "archive_size_bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path), "file_count": len(entries),
        "unpacked_size_bytes": sum(int(entry["size"]) for entry in entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "releases" / "paper-results-v1")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    groups = studies()
    if args.inventory_only:
        for name, files, runs in groups:
            size = sum(path.stat().st_size for path in files)
            print(f"{name}: {runs} runs, {len(files)} files, {size / 2**20:.2f} MiB")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, files, runs in groups:
        print(f"Packaging {name}: {runs} runs, {len(files)} files", flush=True)
        asset = package(name, files, args.output)
        asset["reconstruction_count"] = runs
        assets.append(asset)
    catalog = {
        "schema_version": 1, "repository": "alexdelise/ActiveConditionalGCS",
        "release_tag": "paper-results-v1", "published": False,
        "main_reconstructions": 1800, "new_ablation_reconstructions": 900,
        "reused_main_ablation_rows": 300, "assets": assets,
    }
    (args.output / "data_manifest.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(json.dumps(catalog, indent=2))


if __name__ == "__main__":
    main()
