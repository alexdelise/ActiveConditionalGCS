"""Run one sampling-law and recovery setting over the complete experiment grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("prompt_matched", "prompt_mismatched", "out_of_range")
CS_PREFIXES = {
    "k0": "sample_k0_unconditioned",
    "k1": "sample_k1_daytime_beach",
    "k2": "sample_k2_sunset_beach",
    "k4": "sample_k4_cat",
}
RECOVERIES = ("unprompted", "daytime_beach", "sunset_beach", "cat")
CFG_LINES = {"1": "cfg1", "3": "cfg3", "5": "cfg5", "7.5": "cfg7p5"}


def configured_values_match(expected: dict, saved: dict) -> bool:
    """Compare configured values while allowing saved default fields."""

    return all(
        key in saved and (
            isinstance(saved[key], dict) and configured_values_match(value, saved[key])
            if isinstance(value, dict) else saved[key] == value
        )
        for key, value in expected.items()
    )


def reference_complete(command: list[str]) -> bool:
    """Check the main-study CFG 1 grid before loading a diffusion model."""

    import numpy as np

    suite = json.loads((ROOT / command[command.index("--suite-config") + 1]).read_text())
    expected = json.loads((ROOT / suite["base_config"]).read_text())
    name = command[command.index("--cases") + 1]
    case = next(case for case in suite["cases"] if case["name"] == name)
    for group, values in case.get("overrides", {}).items():
        if isinstance(values, dict):
            expected.setdefault(group, {}).update(values)
        else:
            expected[group] = values
    case_root = ROOT / command[command.index("--results-root") + 1] / suite["tag"] / name
    dataset = json.loads((ROOT / "datasets" / expected["dataset"]["name"] / "dataset_index.json").read_text())
    for item in dataset["items"]:
        for ratio in expected["sweep"]["sampling_perc_list"]:
            ratio_folder = f"samp_{ratio:.5f}".replace(".", "p")
            for repeat in range(5):
                run = case_root / "cs" / f"item_{int(item['item_id']):03d}" / ratio_folder / f"rep_{repeat:02d}"
                if not all((run / filename).is_file() for filename in ("run_data.npz", "run_summary.txt", "run_config.json")):
                    return False
                saved = json.loads((run / "run_config.json").read_text())
                # Resumed configurations record the remaining budget rather than the total
                solver = saved["reconstruction_solver"]
                solver["outer_iterations"] += int(solver.get("iteration_offset", 0))
                if not configured_values_match(expected, saved):
                    raise ValueError(f"Incompatible CFG 1 reference configuration: {run}")
                with np.load(run / "run_data.npz", allow_pickle=False) as data:
                    if (
                        int(data["bp_completed_iterations"]) != expected["reconstruction_solver"]["outer_iterations"]
                        or int(data["repeat_id"]) != repeat
                        or int(data["item_id"]) != int(item["item_id"])
                        or not np.isclose(float(data["samp_perc"]), ratio, rtol=0, atol=5e-8)
                    ):
                        return False
    return True


def resolved_command(mode: str, scenario: str, law: str, recovery: str) -> list[str]:
    """Return the existing suite runner command without altering optimization settings."""

    if mode == "ablation" and recovery == "1":
        # The CFG 1 sunset-beach reconstruction is the main-study reference
        mode, recovery = "main", "sunset_beach"
    method = "cs" if law in CS_PREFIXES else law
    if mode == "main":
        prefix = CS_PREFIXES[law] if method == "cs" else f"baseline_{law}"
        suffix = recovery if method != "cs" or recovery == "unprompted" else f"prompt_{recovery}"
        case = f"{prefix}__recover_{suffix}"
        suite = f"configs/weighted/{scenario}/{law}_suite.json"
        result_root = f"results/weighted/{scenario}"
    else:
        case = f"{CS_PREFIXES[law]}__recover_prompt_sunset_beach_{CFG_LINES[recovery]}"
        suite = f"configs/weighted/ablation/{scenario}/{law}_suite.json"
        result_root = f"results/weighted/ablation/{scenario}"
    return [
        sys.executable, str(ROOT / "run_conditioning_regression.py"),
        "--suite-config", suite, "--sampling-methods", method, "--cases", case,
        "--results-root", result_root, "--repeats-per-setting", "5",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("main", "ablation"))
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("sampling_law", choices=(*CS_PREFIXES, "mcs", "inverse_square"))
    parser.add_argument("recovery", help="Recovery prompt for main runs, or CFG 1, 3, 5, 7.5 for ablations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    choices = RECOVERIES if args.mode == "main" else tuple(CFG_LINES)
    if args.recovery not in choices:
        parser.error(f"recovery must be one of {', '.join(choices)}")
    if args.mode == "ablation" and args.sampling_law not in CS_PREFIXES:
        parser.error("the CFG ablation uses only the four Christoffel sampling laws")
    command = resolved_command(args.mode, args.scenario, args.sampling_law, args.recovery)
    if args.dry_run:
        suite_path = ROOT / command[command.index("--suite-config") + 1]
        suite = json.loads(suite_path.read_text())
        base = json.loads((ROOT / suite["base_config"]).read_text())
        name = command[command.index("--cases") + 1]
        case = next(case for case in suite["cases"] if case["name"] == name)
        print(json.dumps({
            "scenario": args.scenario, "sampling_law": args.sampling_law,
            "recovery": args.recovery, "case": case["name"],
            "ratios": base["sweep"]["sampling_perc_list"], "trials": 5,
            "expected_reconstructions": 25,
            "results_root": command[command.index("--results-root") + 1],
            "main_reference": args.mode == "ablation" and args.recovery == "1",
            "command": command,
        }, indent=2))
        return
    if args.mode == "ablation" and args.recovery == "1" and reference_complete(command):
        print("All CFG 1 reference reconstructions are present in main results; no reconstruction started")
        return
    os.chdir(ROOT)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
