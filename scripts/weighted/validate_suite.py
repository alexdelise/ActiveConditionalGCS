"""Validate every weighted manifest and resolve it through list-cases mode."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "weighted"
EXPECTED_RATES = [0.00125, 0.0025, 0.005, 0.01, 0.025]
EXPECTED_PRIORS = {
    "sample_k0_unconditioned": "Ktilde_SD15__fft__k0_512x512_S10000_ns20",
    "sample_k1_daytime_beach": "Ktilde_SD15__fft__k1daytimebeach_512x512_S10000_ns20",
    "sample_k2_sunset_beach": "Ktilde_SD15__fft__k2sunsetbeach_512x512_S10000_ns20",
    "sample_k4_cat": "Ktilde_SD15__fft__k4cat_512x512_S10000_ns20",
}
MAIN_RECOVERY_CONDITIONS = {
    "unprompted",
    "daytime_beach",
    "sunset_beach",
    "cat",
}
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

    sampling = dict(base["sampling"])
    if sampling.get("weighted_ls") is not True:
        raise ValueError(f"{base_path}: weighted_ls must be enabled.")
    if sampling.get("fft_normalization") != "ortho":
        raise ValueError(f"{base_path}: fft_normalization must be ortho.")
    if float(sampling.get("probability_regularization_zeta", -1.0)) != 0.5:
        raise ValueError(f"{base_path}: zeta must equal 0.5.")
    if list(base["sweep"]["sampling_perc_list"]) != EXPECTED_RATES:
        raise ValueError(f"{base_path}: unexpected sampling grid.")
    if int(base["sweep"]["repeats_per_setting"]) != 5:
        raise ValueError(f"{base_path}: expected five repeats.")
    if not str(suite["tag"]).startswith("weighted/"):
        raise ValueError(f"{path}: tag must remain in the weighted namespace.")
    if len(suite["cases"]) != 4:
        raise ValueError(f"{path}: expected four recovery cases.")

    prior = path.name.removesuffix("_suite.json")
    expected_ktilde = EXPECTED_PRIORS[prior]
    observed_conditions = {str(case["reconstruction_condition"]) for case in suite["cases"]}
    if observed_conditions != expected_conditions:
        raise ValueError(f"{path}: unexpected recovery grid {sorted(observed_conditions)}.")
    for case in suite["cases"]:
        ktilde_name = str(case.get("overrides", {}).get("ktilde", {}).get("name", base["ktilde"]["name"]))
        if ktilde_name != expected_ktilde:
            raise ValueError(f"{path}: expected {expected_ktilde}, got {ktilde_name}.")
        if not (ROOT / "ktilde" / "weighted" / "reference" / f"{ktilde_name}.npz").is_file():
            raise FileNotFoundError(f"Missing S10000 artifact {ktilde_name}.npz.")

    relative_path = path.relative_to(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "run_conditioning_regression.py",
            "--suite-config",
            str(relative_path),
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
    return ("ablation" if is_ablation else "main"), 4 * len(EXPECTED_RATES) * 5


def main() -> None:
    manifests = sorted(CONFIG_ROOT.rglob("sample_*_suite.json"))
    if len(manifests) != 24:
        raise ValueError(f"Expected 24 weighted suite manifests, found {len(manifests)}.")
    totals = {"main": 0, "ablation": 0}
    for manifest in manifests:
        suite_kind, row_count = validate_manifest(manifest)
        totals[suite_kind] += row_count
        print(f"ok: {manifest.relative_to(ROOT)}")
    if totals != {"main": 1200, "ablation": 1200}:
        raise ValueError(f"Unexpected planned reconstruction totals: {totals}.")
    print(f"validated {len(manifests)} manifests; planned rows: {totals}")


if __name__ == "__main__":
    main()
