#!/usr/bin/env python3
"""Build the nine checked-in weighted baseline suite manifests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "configs" / "weighted" / "baselines"

FAMILIES = ("prompt_matched", "prompt_mismatched", "out_of_range")
RECOVERIES = {
    "unprompted": ("", "Unprompted", 0),
    "daytime_beach": ("daytime beach", "Daytime Beach", 1),
    "sunset_beach": ("sunset beach", "Sunset Beach", 2),
    "cat": ("cat", "Cat", 3),
}
SCHEMES = {
    "mcs": {
        "method": "mcs",
        "sampling_condition": "mcs",
        "sampling_label": "$\\mu_{\\mathrm{MCS}}$",
        "sampling_rank": 4,
        "description": "uniform MCS",
    },
    "inverse_square": {
        "method": "inverse_square",
        "sampling_condition": "inverse_square",
        "sampling_label": "$\\mu_{\\mathrm{IS}}$",
        "sampling_rank": 5,
        "description": "pure inverse-square variable-density",
    },
    "vdhh": {
        "method": "vdhh",
        "sampling_condition": "vdhh",
        "sampling_label": "$\\mu_{\\mathrm{VDHH}}$",
        "sampling_rank": 6,
        "description": "design-weighted half-disk/half-outside VDHH",
    },
}


def sampling_override(active_method: str) -> dict:
    return {
        "weighted_ls": True,
        "fft_normalization": "ortho",
        "probability_regularization_zeta": 0.0,
        "methods_enabled": {
            "cs": False,
            "mcs": active_method == "mcs",
            "vdhh": active_method == "vdhh",
            "inverse_square": active_method == "inverse_square",
        },
        "vd_params": {
            "vdhh": {
                "lowfreq_scale": 2.0,
                "max_disk_fraction": 0.5,
            },
            "inverse_square": {},
        },
    }


def build_suite(family: str, scheme: str, spec: dict) -> dict:
    cases = []
    for recovery, (prompt, label, rank) in RECOVERIES.items():
        cases.append(
            {
                "name": f"baseline_{scheme}__recover_{recovery}",
                "description": (
                    f"Recover the {family.replace('_', '-')} signal with "
                    f"{spec['description']} sampling and {label.lower()} recovery."
                ),
                "sampling_condition": spec["sampling_condition"],
                "sampling_label": spec["sampling_label"],
                "sampling_rank": spec["sampling_rank"],
                "reconstruction_condition": recovery,
                "reconstruction_label": label,
                "recon_rank": rank,
                "overrides": {
                    "ktilde": {"name": ""},
                    "sampling": sampling_override(str(spec["method"])),
                    "reconstruction": {
                        "prompt": prompt,
                        "prompts": None,
                    },
                },
            }
        )
    return {
        "base_config": f"configs/weighted/{family}/sunset/base.json",
        "tag": f"weighted/{family}/sunset/baselines/{scheme}",
        "use_dc_presets": False,
        "baseline_sampling_method": str(spec["method"]),
        "cases": cases,
    }


def main() -> None:
    for family in FAMILIES:
        destination = CONFIG_ROOT / family / "sunset"
        destination.mkdir(parents=True, exist_ok=True)
        for scheme, spec in SCHEMES.items():
            path = destination / f"{scheme}_suite.json"
            path.write_text(
                json.dumps(build_suite(family, scheme, spec), indent=2) + "\n",
                encoding="utf-8",
            )
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
