"""Validate every canonical unweighted split manifest through list-cases mode."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "unweighted"
SPLIT_RATES = {
    "first4": [0.00015625, 0.0003125, 0.000625, 0.00125],
    "last3": [0.0025, 0.005, 0.01],
}
EXPECTED_PRIORS = {
    "sample_k0_unconditioned": "Ktilde_SD15__fft__k0_512x512_S500_ns20",
    "sample_k1_daytime_beach": "Ktilde_SD15__fft__k1daytimebeach_512x512_S500_ns20",
    "sample_k2_sunset_beach": "Ktilde_SD15__fft__k2sunsetbeach_512x512_S500_ns20",
    "sample_k4_cat": "Ktilde_SD15__fft__k4cat_512x512_S500_ns20",
}
MAIN_RECOVERY_CONDITIONS = {"unprompted", "daytime_beach", "sunset_beach", "cat"}
ABLATION_RECOVERY_CONDITIONS = {
    "sunset_beach_cfg1",
    "sunset_beach_cfg1p5",
    "sunset_beach_cfg3",
    "sunset_beach_cfg5",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest(path: Path) -> tuple[str, int]:
    suite = load_json(path)
    base_path = ROOT / str(suite["base_config"])
    base = load_json(base_path)
    is_ablation = "ablation" in path.parts
    expected_conditions = ABLATION_RECOVERY_CONDITIONS if is_ablation else MAIN_RECOVERY_CONDITIONS
    split = "first4" if path.name.startswith("first4_") else "last3"
    prior = path.name.removeprefix(f"{split}_").removesuffix("_suite.json")

    sampling = dict(base["sampling"])
    if sampling.get("weighted_ls") is not False:
        raise ValueError(f"{base_path}: weighted_ls must remain disabled.")
    if sampling.get("fft_normalization", "backward") not in {"backward", None}:
        raise ValueError(f"{base_path}: unweighted FFT convention changed unexpectedly.")
    if list(base["sweep"]["sampling_perc_list"]) != SPLIT_RATES[split]:
        raise ValueError(f"{base_path}: unexpected {split} sampling grid.")
    if int(base["sweep"]["repeats_per_setting"]) != 5:
        raise ValueError(f"{base_path}: expected five repeats.")
    if not str(suite["tag"]).startswith("unweighted/"):
        raise ValueError(f"{path}: tag must remain in the unweighted namespace.")
    if len(suite["cases"]) != 4:
        raise ValueError(f"{path}: expected four recovery cases.")

    observed_conditions = {str(case["reconstruction_condition"]) for case in suite["cases"]}
    if observed_conditions != expected_conditions:
        raise ValueError(f"{path}: unexpected recovery grid {sorted(observed_conditions)}.")
    expected_ktilde = EXPECTED_PRIORS[prior]
    for case in suite["cases"]:
        ktilde_name = str(case.get("overrides", {}).get("ktilde", {}).get("name", base["ktilde"]["name"]))
        if ktilde_name != expected_ktilde:
            raise ValueError(f"{path}: expected {expected_ktilde}, got {ktilde_name}.")
        artifact = ROOT / "ktilde" / "unweighted" / f"{ktilde_name}.npz"
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing unweighted S500 artifact: {artifact}.")

    completed = subprocess.run(
        [
            sys.executable,
            "run_conditioning_regression.py",
            "--suite-config",
            str(path.relative_to(ROOT)),
            "--sampling-methods",
            "cs",
            "--list-cases",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    ready_count = completed.stdout.count("[ready]")
    if ready_count != 4:
        raise ValueError(f"{path}: list-cases resolved {ready_count} ready cases, expected four.")
    return ("ablation" if is_ablation else "main"), 4 * len(SPLIT_RATES[split]) * 5


def main() -> None:
    manifests = sorted(CONFIG_ROOT.rglob("first4_sample_*_suite.json"))
    manifests.extend(sorted(CONFIG_ROOT.rglob("last3_sample_*_suite.json")))
    if len(manifests) != 48:
        raise ValueError(f"Expected 48 unweighted split manifests, found {len(manifests)}.")
    totals = {"main": 0, "ablation": 0}
    for manifest in manifests:
        suite_kind, row_count = validate_manifest(manifest)
        totals[suite_kind] += row_count
        print(f"ok: {manifest.relative_to(ROOT)}")
    if totals != {"main": 1680, "ablation": 1680}:
        raise ValueError(f"Unexpected planned reconstruction totals: {totals}.")
    print(f"validated {len(manifests)} manifests; planned rows: {totals}")


if __name__ == "__main__":
    main()
