#!/usr/bin/env python
"""Build the six unweighted MCS/inverse-square suite manifests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("prompt_matched", "prompt_mismatched", "out_of_range")
SCHEMES = {
    "mcs": {
        "label": r"$\mu_{\mathrm{MCS}}$",
        "description": "uniform MCS",
    },
    "inverse_square": {
        "label": r"$\mu_{\mathrm{IS}}$",
        "description": "pure inverse-square",
    },
}
RECOVERIES = (
    ("unprompted", "Unprompted", "", 0),
    ("daytime_beach", "Daytime Beach", "daytime beach", 1),
    ("sunset_beach", "Sunset Beach", "sunset beach", 2),
    ("cat", "Cat", "cat", 3),
)


def suite_payload(family: str, scheme: str) -> dict:
    info = SCHEMES[scheme]
    cases = []
    for recovery, recovery_label, prompt, rank in RECOVERIES:
        methods = {
            "cs": False,
            "mcs": scheme == "mcs",
            "inverse_square": scheme == "inverse_square",
        }
        cases.append(
            {
                "name": f"baseline_{scheme}__recover_{recovery}",
                "description": (
                    f"Recover the {family.replace('_', ' ')} signal with "
                    f"{info['description']} sampling and "
                    f"{recovery.replace('_', ' ')} recovery."
                ),
                "sampling_condition": scheme,
                "sampling_label": info["label"],
                "sampling_rank": 4 if scheme == "mcs" else 5,
                "reconstruction_condition": recovery,
                "reconstruction_label": recovery_label,
                "recon_rank": rank,
                "overrides": {
                    "ktilde": {"name": ""},
                    "sampling": {
                        "weighted_ls": False,
                        "fft_normalization": "backward",
                        "probability_regularization_zeta": 0.0,
                        "methods_enabled": methods,
                    },
                    "reconstruction": {
                        "prompt": prompt,
                        "prompts": None,
                    },
                },
            }
        )
    return {
        "base_config": f"configs/unweighted/{family}/sunset/base.json",
        "tag": f"unweighted/{family}/sunset/baselines/{scheme}",
        "use_dc_presets": False,
        "baseline_sampling_method": scheme,
        "cases": cases,
    }


def main() -> None:
    for family in FAMILIES:
        output_dir = ROOT / "configs" / "unweighted" / "baselines" / family / "sunset"
        output_dir.mkdir(parents=True, exist_ok=True)
        for scheme in SCHEMES:
            output_path = output_dir / f"{scheme}_suite.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(suite_payload(family, scheme), handle, indent=2)
                handle.write("\n")
            print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
