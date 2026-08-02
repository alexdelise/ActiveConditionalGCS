#!/usr/bin/env python
"""Refresh LPIPS and baseline support in the unweighted CFG notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = ROOT / "analyze_results" / "unweighted" / "ablation"
FAMILIES = ("prompt_matched", "prompt_mismatched", "out_of_range")


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def set_source(cell: dict[str, Any], text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def main() -> None:
    for family in FAMILIES:
        path = NOTEBOOK_ROOT / f"{family}_cfg_ablation.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []

        setup = source_text(notebook["cells"][1])
        if "import sd15_recovery_analysis as recovery" not in setup:
            setup = setup.replace(
                "import sd15_cfg_ablation_analysis as cfgviz\n"
                "importlib.reload(cfgviz)",
                "import sd15_cfg_ablation_analysis as cfgviz\n"
                "import sd15_recovery_analysis as recovery\n"
                "importlib.reload(cfgviz)\n"
                "importlib.reload(recovery)",
            )
        set_source(notebook["cells"][1], setup)

        load_code = source_text(notebook["cells"][3])
        lpips_block = (
            "LPIPS_TABLE = recovery.ensure_lpips_metrics(\n"
            "    SD15_ROOT,\n"
            "    result_namespace='unweighted',\n"
            "    artifact_roots=[\n"
            f"        SD15_ROOT / 'results' / 'unweighted' / '{family}' / 'sunset',\n"
            f"        SD15_ROOT / 'results' / 'unweighted' / 'ablation' / '{family}' / 'sunset',\n"
            "    ],\n"
            "    device='cpu',\n"
            ")\n\n"
        )
        if "LPIPS_TABLE = recovery.ensure_lpips_metrics(" not in load_code:
            load_code = lpips_block + load_code
        load_code = load_code.replace(
            "'psnr_db', 'ssim', 'pixel_mae', 'case_root'",
            "'psnr_db', 'ssim', 'lpips', 'pixel_mae', 'case_root'",
        )
        set_source(notebook["cells"][3], load_code)

        metric_markdown = source_text(notebook["cells"][4])
        metric_markdown = metric_markdown.replace(
            "PSNR and SSIM versus sampling ratio",
            "PSNR, SSIM, and LPIPS versus sampling ratio",
        )
        set_source(notebook["cells"][4], metric_markdown)

        introduction = source_text(notebook["cells"][0])
        if "uniform MCS" not in introduction:
            introduction += (
                "\nThe analysis also ingests uniform MCS and pure inverse-square "
                "sampling when those baseline rows are available.\n"
            )
        set_source(notebook["cells"][0], introduction)

        notebook.setdefault("metadata", {}).pop("widgets", None)
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
