#!/usr/bin/env python
"""Build unweighted MCS and inverse-square CFG-ablation suites."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("prompt_matched", "prompt_mismatched", "out_of_range")
SCHEMES = {
    "mcs": {
        "label": r"$\mu_{\mathrm{MCS}}$",
        "description": "uniform MCS",
        "rank": 5,
    },
    "inverse_square": {
        "label": r"$\mu_{\mathrm{IS}}$",
        "description": "pure inverse-square",
        "rank": 6,
    },
}
CFG_SETTINGS = (
    ("cfg1", "CFG 1", 1.0, 1),
    ("cfg1p5", "CFG 1.5", 1.5, 2),
    ("cfg3", "CFG 3", 3.0, 3),
    ("cfg5", "CFG 5", 5.0, 4),
)


def suite_payload(family: str, scheme: str) -> dict:
    info = SCHEMES[scheme]
    methods = {
        "cs": False,
        "mcs": scheme == "mcs",
        "vdhh": False,
        "inverse_square": scheme == "inverse_square",
    }
    cases = []
    for key, label, cfg_scale, rank in CFG_SETTINGS:
        cases.append(
            {
                "name": f"baseline_{scheme}__recover_prompt_sunset_beach_{key}",
                "description": (
                    f"Recover the {family.replace('_', ' ')} signal with "
                    f"{info['description']} sampling and sunset beach recovery "
                    f"at {label}."
                ),
                "sampling_condition": scheme,
                "sampling_label": info["label"],
                "sampling_rank": info["rank"],
                "reconstruction_condition": f"sunset_beach_{key}",
                "reconstruction_label": label,
                "recon_rank": rank,
                "overrides": {
                    "gen_recon": {"guidance_scale": cfg_scale},
                    "ktilde": {"name": ""},
                    "sampling": {
                        "weighted_ls": False,
                        "fft_normalization": "backward",
                        "probability_regularization_zeta": 0.0,
                        "methods_enabled": methods,
                        "vd_params": {
                            "vdhh": {
                                "lowfreq_scale": 2.0,
                                "max_disk_fraction": 0.5,
                            },
                            "inverse_square": {},
                        },
                    },
                    "reconstruction": {
                        "prompt": "sunset beach",
                        "prompts": None,
                    },
                },
            }
        )
    return {
        "base_config": f"configs/unweighted/{family}/sunset/base.json",
        "tag": f"unweighted/ablation/{family}/sunset/{scheme}_cfg_ablation",
        "use_dc_presets": False,
        "baseline_sampling_method": scheme,
        "cases": cases,
    }


def main() -> None:
    for family in FAMILIES:
        output_dir = (
            ROOT
            / "configs"
            / "unweighted"
            / "ablation"
            / "baselines"
            / family
            / "sunset"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for scheme in SCHEMES:
            output_path = output_dir / f"{scheme}_suite.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(suite_payload(family, scheme), handle, indent=2)
                handle.write("\n")
            print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
