"""Validate and copy weighted main references into weighted CFG analyses."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PRIOR_KTILDE = {
    "sample_k0_unconditioned": "Ktilde_SD15__fft__k0_512x512_S10000_ns20",
    "sample_k1_daytime_beach": "Ktilde_SD15__fft__k1daytimebeach_512x512_S10000_ns20",
    "sample_k2_sunset_beach": "Ktilde_SD15__fft__k2sunsetbeach_512x512_S10000_ns20",
    "sample_k4_cat": "Ktilde_SD15__fft__k4cat_512x512_S10000_ns20",
}
FAMILIES = {"prompt_matched", "prompt_mismatched", "out_of_range"}
COMPATIBLE_BLOCKS = (
    "image",
    "dataset",
    "runtime",
    "reconstruction_solver",
    "sampling",
    "sweep",
    "optim",
    "repro",
    "output",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_reference_config(
    source_config: dict[str, Any],
    ablation_base: dict[str, Any],
    *,
    expected_ktilde: str,
    expected_prompt: str,
) -> None:
    for block in COMPATIBLE_BLOCKS:
        if source_config.get(block) != ablation_base.get(block):
            raise ValueError(f"Weighted reference is incompatible in config block {block!r}.")
    if source_config.get("ktilde", {}).get("name") != expected_ktilde:
        raise ValueError("Weighted reference uses the wrong S10000 K-tilde artifact.")
    if float(source_config.get("gen_recon", {}).get("guidance_scale", -1.0)) != 7.5:
        raise ValueError("Weighted reference must use recovery CFG 7.5.")
    if str(source_config.get("reconstruction", {}).get("prompt", "")) != expected_prompt:
        raise ValueError("Weighted reference uses the wrong recovery prompt.")


def sync_reference(
    *,
    source: Path,
    destination: Path,
    ablation_base: dict[str, Any],
    expected_ktilde: str,
    expected_prompt: str,
    copy: bool,
) -> str:
    source_config_path = source / "run_config.json"
    if not source_config_path.is_file():
        return f"missing: {source}"
    validate_reference_config(
        load_json(source_config_path),
        ablation_base,
        expected_ktilde=expected_ktilde,
        expected_prompt=expected_prompt,
    )
    if copy:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return f"copied: {source} -> {destination}"
    return f"compatible: {source}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--prior", required=True, choices=sorted(PRIOR_KTILDE))
    parser.add_argument(
        "--split",
        choices=("first3", "last2"),
        default=None,
        help="Optional weighted rate split whose main references should be reused.",
    )
    parser.add_argument("--copy", action="store_true", help="Copy compatible references after validation.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    split_prior = f"{args.split}_{args.prior}" if args.split else args.prior
    main_root = root / "results" / "weighted" / args.family / "sunset" / split_prior
    target_root = (
        root
        / "results"
        / "weighted"
        / "ablation"
        / args.family
        / "sunset"
        / split_prior
    )
    ablation_base = load_json(root / "configs" / "weighted" / "ablation" / args.family / "sunset" / "base.json")
    expected_ktilde = PRIOR_KTILDE[args.prior]

    mappings = (
        (
            main_root / f"{args.prior}__recover_unprompted",
            target_root / "reference_unconditioned",
            "",
        ),
        (
            main_root / f"{args.prior}__recover_prompt_sunset_beach",
            target_root / "reference_cfg7p5",
            "sunset beach",
        ),
    )
    for source, destination, prompt in mappings:
        print(
            sync_reference(
                source=source,
                destination=destination,
                ablation_base=ablation_base,
                expected_ktilde=expected_ktilde,
                expected_prompt=prompt,
                copy=bool(args.copy),
            )
        )


if __name__ == "__main__":
    main()
