"""Run a conditioning experiment suite over prompt/sampling cases."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


RECONSTRUCTION_SOLVER = "sd15_backprop"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
RESOLVED_SUITE_MANIFEST_FILENAME = "resolved_suite_manifest.json"


def parse_names_csv(text: str | None) -> List[str]:
    """Parse a comma-separated list into a stripped token list."""

    if text is None:
        return []
    return [token.strip() for token in str(text).split(",") if token.strip()]


def parse_sampling_methods(text: str | None) -> List[int]:
    """Resolve an optional sampling-method override list."""

    requested = parse_names_csv(text)
    if not requested:
        return []

    # Import lazily so --help remains cheap and config validation stays local to
    # this runner.
    from src.config import sampling_method_id

    return [int(sampling_method_id(token)) for token in requested]


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge JSON-like payloads."""

    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = {str(key): deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            key_str = str(key)
            if key_str in merged:
                # Suite cases override only the fields that differ from the base
                # experiment config, for example prompt or k-tilde name.
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

    pieces = [str(base_tag).strip("/"), str(case_name).strip("/")]
    return "/".join(piece for piece in pieces if piece)


def json_dump(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def ktilde_artifact_path(project_root: Path, ktilde_name: str) -> Path | None:
    """Return the expected artifact path, or None for a non-CS run."""

    name = str(ktilde_name).strip()
    if not name:
        return None
    from src.utils import resolve_ktilde_npz_path

    try:
        return resolve_ktilde_npz_path(project_root / "ktilde", name)
    except FileNotFoundError:
        return project_root / "ktilde" / f"{name}.npz"


def main() -> None:
    """Parse CLI arguments and run the requested experiment suite."""

    root = Path(__file__).resolve().parent

    from src.config import from_run_dict, load_json, run_config_to_dict

    parser = argparse.ArgumentParser(description="Run a conditioning experiment suite.")
    parser.add_argument(
        "--suite-config",
        type=str,
        default="configs/unweighted/prompt_mismatched/sunset/sample_k0_unconditioned_suite.json",
        help="Path to the suite manifest JSON.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional override for the top-level results tag.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help=(
            "Optional directory under which the tagged result folder is created. "
            "Relative paths are resolved from the project root; the legacy default "
            "is the repository's results directory."
        ),
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Optional comma-separated subset of suite cases to run.",
    )
    parser.add_argument(
        "--sampling-families",
        type=str,
        default=None,
        help="Optional comma-separated sampling families: classic, baseline, or vd.",
    )
    parser.add_argument(
        "--sampling-methods",
        type=str,
        default=None,
        help=(
            "Optional comma-separated exact sampling methods, e.g. "
            "'cs', 'mcs', or 'inverse_square'."
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the resolved case grid and exit without running reconstructions.",
    )
    parser.add_argument(
        "--skip-missing-ktilde",
        action="store_true",
        help="Skip cases whose named k-tilde artifact has not been built yet.",
    )
    parser.add_argument(
        "--sampling-percentages",
        type=str,
        default=None,
        help=(
            "Optional comma-separated execution shard. The saved canonical "
            "configuration retains its complete sampling sweep."
        ),
    )
    parser.add_argument(
        "--repeats-per-setting",
        type=int,
        default=None,
        help="Optional positive repeat-count override, intended for isolated smoke tests.",
    )
    args = parser.parse_args()

    suite_manifest = load_json(root / args.suite_config)
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

    selected = selected_cases(suite_cases_raw, parse_names_csv(args.cases))
    families = parse_names_csv(args.sampling_families)
    method_ids = parse_sampling_methods(args.sampling_methods)
    if families and method_ids:
        # Families and explicit methods are two ways to select samplers; allowing
        # both would make the output grid ambiguous.
        raise ValueError("--sampling-families and --sampling-methods are mutually exclusive.")

    execution_sampling_percentages = None
    if args.sampling_percentages is not None:
        execution_sampling_percentages = [
            float(value) for value in parse_names_csv(args.sampling_percentages)
        ]
        if not execution_sampling_percentages or any(
            value <= 0.0 or value > 1.0
            for value in execution_sampling_percentages
        ):
            raise ValueError("--sampling-percentages must contain values in (0, 1].")

    execution_repeats_per_setting = args.repeats_per_setting
    if (
        execution_repeats_per_setting is not None
        and int(execution_repeats_per_setting) <= 0
    ):
        raise ValueError("--repeats-per-setting must be positive.")

    results_base = Path(args.results_root) if args.results_root is not None else root / "results"
    if not results_base.is_absolute():
        results_base = root / results_base
    results_base = results_base.resolve()
    top_level_root = results_base / suite_tag

    resolved_cases: List[Dict[str, Any]] = []
    for case in selected:
        name = str(case.get("name", "")).strip()
        if not name:
            raise ValueError("Every suite case must define a non-empty 'name'.")

        overrides = dict(case.get("overrides", {}))
        merged_payload = deep_merge(base_payload, overrides)
        cfg = from_run_dict(merged_payload)
        if execution_sampling_percentages is not None:
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
                    "Execution sampling percentages must be present in the "
                    f"canonical sweep; missing {missing_rates} for case {name!r}."
                )
        if (
            execution_repeats_per_setting is not None
            and int(execution_repeats_per_setting)
            > int(cfg.sweep.repeats_per_setting)
        ):
            raise ValueError(
                "--repeats-per-setting cannot exceed the canonical repeat count "
                f"({cfg.sweep.repeats_per_setting}) for case {name!r}."
            )
        from src.config import enabled_sampling_method_ids

        case_method_ids = (
            [int(value) for value in method_ids]
            if method_ids
            else enabled_sampling_method_ids(cfg.sampling.methods_enabled, families=families or None)
        )
        ktilde_required = 1 in case_method_ids
        artifact_path = ktilde_artifact_path(root, cfg.ktilde.name)
        ktilde_exists = bool(artifact_path is not None and artifact_path.is_file())
        # Resolve each case up front so --list-cases and manifests show the exact
        # merged config and whether its k-tilde artifact is available.
        resolved_cases.append(
            {
                "name": name,
                "description": str(case.get("description", "")),
                "sampling_condition": str(case.get("sampling_condition", "")),
                "sampling_label": str(case.get("sampling_label", case.get("sampling_condition", ""))),
                "sampling_rank": int(case.get("sampling_rank", 0)),
                "reconstruction_condition": str(case.get("reconstruction_condition", "")),
                "reconstruction_label": str(case.get("reconstruction_label", case.get("reconstruction_condition", ""))),
                "recon_rank": int(case.get("recon_rank", 0)),
                "tag": case_tag(suite_tag, name),
                "reconstruction_solver": RECONSTRUCTION_SOLVER,
                "ktilde_name": cfg.ktilde.name,
                "ktilde_required": bool(ktilde_required),
                "ktilde_artifact_path": "" if artifact_path is None else str(artifact_path),
                "ktilde_exists": bool(ktilde_exists),
                "run_config": run_config_to_dict(cfg),
            }
        )

    if args.list_cases:
        for case in resolved_cases:
            status = (
                "ready"
                if not bool(case["ktilde_required"]) or bool(case["ktilde_exists"])
                else "missing-ktilde"
            )
            print(
                f"{case['name']} [{status}] | "
                f"sampling={case['sampling_condition']} | recon={case['reconstruction_condition']} | "
                f"ktilde={case['ktilde_name']}"
            )
        return

    top_level_root.mkdir(parents=True, exist_ok=True)
    json_dump(
        top_level_root / RESOLVED_SUITE_MANIFEST_FILENAME,
        {
            # Store both the suite manifest and each merged run config for
            # reproducibility audits after the run completes.
            "config_path": str((root / args.suite_config).resolve()),
            "suite_tag": suite_tag,
            "reconstruction_solver": RECONSTRUCTION_SOLVER,
            "base_config": base_config_rel,
            "sampling_families": families,
            "sampling_method_ids": method_ids,
            "execution_sampling_percentages": execution_sampling_percentages,
            "execution_repeats_per_setting": execution_repeats_per_setting,
            "results_root": str(results_base),
            "cases": resolved_cases,
        },
    )

    from src.runner import run_enabled_methods, run_methods

    suite_results: List[Dict[str, Any]] = []
    for case in resolved_cases:
        if (
            args.skip_missing_ktilde
            and bool(case["ktilde_required"])
            and not bool(case["ktilde_exists"])
        ):
            print(f"[skip] {case['name']} missing k-tilde: {case['ktilde_name']}")
            suite_results.append(
                {
                    "name": case["name"],
                    "tag": case["tag"],
                    "reconstruction_solver": RECONSTRUCTION_SOLVER,
                    "status": "skipped_missing_ktilde",
                    "ktilde_name": case["ktilde_name"],
                    "ktilde_artifact_path": case["ktilde_artifact_path"],
                }
            )
            continue
        if bool(case["ktilde_required"]) and not bool(case["ktilde_exists"]):
            raise FileNotFoundError(
                f"{case['name']} requires missing k-tilde: {case['ktilde_artifact_path']}"
            )

        cfg = from_run_dict(dict(case["run_config"]))
        print(
            f"[run] {case['name']} | tag={case['tag']} | "
            f"ktilde={case['ktilde_name']} | recon={case['reconstruction_condition']}"
        )
        if method_ids:
            # Per-script paper launches pass --sampling-methods cs, which keeps
            # one sampling distribution per cluster job.
            outputs = run_methods(
                root,
                cfg,
                tag=str(case["tag"]),
                method_ids=method_ids,
                sampling_percentages=execution_sampling_percentages,
                repeats_per_setting=execution_repeats_per_setting,
                results_base=results_base,
            )
        else:
            outputs = run_enabled_methods(
                root,
                cfg,
                tag=str(case["tag"]),
                families=families or None,
                sampling_percentages=execution_sampling_percentages,
                repeats_per_setting=execution_repeats_per_setting,
                results_base=results_base,
            )
        suite_results.append(
            {
                "name": case["name"],
                "tag": case["tag"],
                "reconstruction_solver": RECONSTRUCTION_SOLVER,
                "status": "complete",
                "sampling_condition": case["sampling_condition"],
                "reconstruction_condition": case["reconstruction_condition"],
                "ktilde_name": case["ktilde_name"],
                "outputs": outputs,
            }
        )
        json_dump(top_level_root / "suite_results.json", {"reconstruction_solver": RECONSTRUCTION_SOLVER, "results": suite_results})

    json_dump(top_level_root / "suite_results.json", {"reconstruction_solver": RECONSTRUCTION_SOLVER, "results": suite_results})
    json_dump(
        top_level_root / EXPERIMENT_MANIFEST_FILENAME,
        {
            "suite_tag": suite_tag,
            "suite_config": str((root / args.suite_config).resolve()),
            "reconstruction_solver": RECONSTRUCTION_SOLVER,
            "sampling_families": families,
            "sampling_method_ids": method_ids,
            "execution_sampling_percentages": execution_sampling_percentages,
            "execution_repeats_per_setting": execution_repeats_per_setting,
            "results_root": str(results_base),
            "resolved_suite_manifest_path": str((top_level_root / RESOLVED_SUITE_MANIFEST_FILENAME).resolve()),
            "cases": resolved_cases,
        },
    )


if __name__ == "__main__":
    main()
