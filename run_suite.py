"""Run a suite of tagged experiment cases built from a base run config plus overrides."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


RESOLVED_SUITE_MANIFEST_FILENAME = "resolved_suite_manifest.json"


def parse_names_csv(text: str | None) -> List[str]:
    """Parse a comma-separated list of case names."""

    if text is None:
        return []
    return [token.strip() for token in str(text).split(",") if token.strip()]


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge an override payload into a base JSON-like structure."""

    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = {str(key): deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            key_str = str(key)
            if key_str in merged:
                merged[key_str] = deep_merge(merged[key_str], value)
            else:
                merged[key_str] = deepcopy(value)
        return merged
    return deepcopy(override)


def selected_cases(cases: Sequence[Mapping[str, Any]], names: Sequence[str]) -> List[Mapping[str, Any]]:
    """Filter suite cases by name while preserving manifest order."""

    if not names:
        return [dict(case) for case in cases]

    name_set = {str(name) for name in names}
    selected = [dict(case) for case in cases if str(case.get("name", "")) in name_set]
    missing = sorted(name_set.difference({str(case.get("name", "")) for case in selected}))
    if missing:
        raise KeyError(f"Unknown suite case names: {missing}")
    return selected


def case_tag(base_tag: str, case_name: str) -> str:
    """Return the nested results tag for one suite case."""

    clean_base = str(base_tag).strip().strip("/")
    clean_case = str(case_name).strip().strip("/")
    if not clean_base:
        return clean_case
    return f"{clean_base}/{clean_case}"


def ktilde_artifact_path(project_root: Path, ktilde_name: str) -> Path:
    """Return the expected artifact path for one named k-tilde."""

    return project_root / "ktilde" / f"{str(ktilde_name).strip()}.npz"


def json_dump(path: Path, payload: Any) -> None:
    """Write JSON to disk with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    """Parse CLI arguments and run the selected suite cases."""

    root = Path(__file__).resolve().parent

    from src.config import from_run_dict, load_json, run_config_to_dict

    parser = argparse.ArgumentParser(description="Run a suite of experiment cases under a shared results tag.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/prompt_matched/sunset/sample_k0_unconditioned_suite.json",
        help="Path to the suite manifest JSON.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional override for the manifest results tag.")
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Optional comma-separated subset of case names to run.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the resolved case list and exit without running any reconstructions.",
    )
    parser.add_argument(
        "--skip-missing-ktilde",
        action="store_true",
        help="Skip cases whose named k-tilde artifact has not been built yet.",
    )
    args = parser.parse_args()

    suite_manifest = load_json(root / args.config)
    suite_tag = str(args.tag if args.tag is not None else suite_manifest.get("tag", "")).strip()
    if not suite_tag:
        raise ValueError("Suite manifest must define a non-empty 'tag', or one must be passed with --tag.")

    base_config_rel = str(suite_manifest.get("base_config", "")).strip()
    if not base_config_rel:
        raise ValueError("Suite manifest must define a non-empty 'base_config' path.")

    base_payload = load_json(root / base_config_rel)
    suite_cases_raw = suite_manifest.get("cases", [])
    if not isinstance(suite_cases_raw, list) or not suite_cases_raw:
        raise ValueError("Suite manifest must define a non-empty 'cases' list.")

    cases = selected_cases(suite_cases_raw, parse_names_csv(args.cases))
    suite_root = root / "results" / suite_tag

    resolved_cases: List[Dict[str, Any]] = []
    for case in cases:
        name = str(case.get("name", "")).strip()
        if not name:
            raise ValueError("Every suite case must define a non-empty 'name'.")

        overrides = dict(case.get("overrides", {}))
        merged_payload = deep_merge(base_payload, overrides)
        cfg = from_run_dict(merged_payload)
        reconstruction_prompt = str(cfg.reconstruction.prompt or "")
        artifact_path = ktilde_artifact_path(root, cfg.ktilde.name)
        resolved_cases.append(
            {
                "name": name,
                "description": str(case.get("description", "")),
                "sampling_condition": str(case.get("sampling_condition", "")),
                "reconstruction_condition": str(case.get("reconstruction_condition", "")),
                "tag": case_tag(suite_tag, name),
                "ktilde_name": cfg.ktilde.name,
                "ktilde_artifact_path": str(artifact_path),
                "ktilde_exists": bool(artifact_path.is_file()),
                "reconstruction_prompt": reconstruction_prompt,
                "run_config": run_config_to_dict(cfg),
            }
        )

    if args.list_cases:
        for case in resolved_cases:
            status = "ready" if bool(case["ktilde_exists"]) else "missing-ktilde"
            print(
                f"{case['name']}: {status} | tag={case['tag']} | "
                f"ktilde={case['ktilde_name']} | recon_prompt={case['reconstruction_condition']}"
            )
        return

    suite_root.mkdir(parents=True, exist_ok=True)
    json_dump(
        suite_root / RESOLVED_SUITE_MANIFEST_FILENAME,
        {
            "config_path": str((root / args.config).resolve()),
            "suite_tag": suite_tag,
            "base_config": base_config_rel,
            "cases": resolved_cases,
        },
    )

    from src.runner import run_enabled_methods

    suite_results: List[Dict[str, Any]] = []
    for case in resolved_cases:
        if args.skip_missing_ktilde and not bool(case["ktilde_exists"]):
            print(f"[skip] {case['name']} missing k-tilde: {case['ktilde_name']}")
            suite_results.append(
                {
                    "name": case["name"],
                    "tag": case["tag"],
                    "status": "skipped_missing_ktilde",
                    "ktilde_name": case["ktilde_name"],
                    "ktilde_artifact_path": case["ktilde_artifact_path"],
                }
            )
            continue

        cfg = from_run_dict(dict(case["run_config"]))
        print(
            f"[run] {case['name']} | tag={case['tag']} | "
            f"ktilde={case['ktilde_name']} | recon_prompt={case['reconstruction_condition']}"
        )
        outputs = run_enabled_methods(root, cfg, tag=str(case["tag"]))
        suite_results.append(
            {
                "name": case["name"],
                "tag": case["tag"],
                "status": "complete",
                "ktilde_name": case["ktilde_name"],
                "outputs": outputs,
            }
        )
        json_dump(suite_root / "suite_results.json", {"suite_tag": suite_tag, "results": suite_results})

    json_dump(suite_root / "suite_results.json", {"suite_tag": suite_tag, "results": suite_results})


if __name__ == "__main__":
    main()
