#!/usr/bin/env python
"""Refresh the three unweighted main-result notebooks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT / "analyze_results" / "unweighted" / "main"
NOTEBOOKS = {
    "prompt_matched": "Prompt-Matched In-Range",
    "prompt_mismatched": "Prompt-Mismatched In-Range",
    "out_of_range": "Out-of-Range",
}


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def set_source(cell: dict[str, Any], text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def setup_code(family: str) -> str:
    return f"""from pathlib import Path
import importlib
import sys

from IPython.display import display

NOTEBOOK_DIR = Path.cwd().resolve()
for search_root in [NOTEBOOK_DIR, *NOTEBOOK_DIR.parents]:
    helper_dir = search_root / 'analyze_results'
    helper_path = helper_dir / 'sd15_recovery_analysis.py'
    if helper_path.exists():
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        break
    for child in search_root.iterdir():
        if not child.is_dir():
            continue
        helper_dir = child / 'analyze_results'
        helper_path = helper_dir / 'sd15_recovery_analysis.py'
        if helper_path.exists():
            if str(helper_dir) not in sys.path:
                sys.path.insert(0, str(helper_dir))
            break
    else:
        continue
    break
else:
    raise FileNotFoundError('Could not find sd15_recovery_analysis.py from the notebook cwd.')

import sd15_recovery_analysis as recovery
recovery = importlib.reload(recovery)

SD15_ROOT = recovery.find_sd15_root(NOTEBOOK_DIR)
UNWEIGHTED_BASE_TAG = 'unweighted/{family}/sunset'
SAMPLING_METHODS = list(recovery.UNWEIGHTED_MAIN_SAMPLING_METHODS)
ALLOWED_SAMPLING_PERC = set(recovery.DEFAULT_ALLOWED_SAMPLING_PERCENTAGES)
OUTPUT_ROOT = SD15_ROOT / 'results'

# LPIPS is calculated by the shared analysis pipeline and saved incrementally.
LPIPS_TABLE = recovery.ensure_lpips_metrics(
    SD15_ROOT,
    result_namespace='unweighted',
    artifact_roots=[SD15_ROOT / 'results' / UNWEIGHTED_BASE_TAG],
    device='cpu',
)
analysis, COMPLETION_TABLE = recovery.load_unweighted_main_analysis(
    SD15_ROOT,
    base_tag=UNWEIGHTED_BASE_TAG,
    output_root=OUTPUT_ROOT,
    include_partial=True,
)
ROWS = analysis.rows
MEAN_TABLE = analysis.mean_table
ACTIVE_TAG = analysis.active_tag
LOADED_TAGS = analysis.loaded_tags
OUTPUT_DIR = analysis.output_dir / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPLETION_PATH = OUTPUT_DIR / 'unweighted_main_completion.csv'
COMPLETION_TABLE.to_csv(COMPLETION_PATH, index=False)

print(f'Active tag: {{ACTIVE_TAG}}')
print(f'Loaded source tags: {{LOADED_TAGS}}')
print(f'Loaded {{len(ROWS)}} / 840 expected run rows.')
lpips_count = int(ROWS['lpips'].notna().sum()) if 'lpips' in ROWS else 0
print(f'Loaded {{lpips_count}} LPIPS values.')
display(
    COMPLETION_TABLE.groupby(
        ['sampling_method', 'sampling_condition'],
        as_index=False,
    )[['observed', 'expected', 'left']].sum()
)
display(MEAN_TABLE)
if MEAN_TABLE.empty:
    print('No recovery rows found yet. Run the suite first, then rerun this notebook.')
"""


def main() -> None:
    for family, title in NOTEBOOKS.items():
        path = ANALYSIS_ROOT / f"{family}_results.ipynb"
        with path.open("r", encoding="utf-8") as handle:
            notebook = copy.deepcopy(json.load(handle))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []

        set_source(
            notebook["cells"][0],
            f"""# SD1.5 {title} Recovery Results

This notebook analyzes the original unweighted reconstruction pipeline across
the four empirical Christoffel sampling laws, uniform MCS, and pure
inverse-square sampling. All six distributions use the original seven sampling
ratios and five trials. PSNR, SSIM, LPIPS, and per-pixel MAE are loaded and
plotted through the shared analysis code. Outputs remain under
the corresponding experiment's `results/unweighted/<scenario>/sunset/figures/`
directory.
""",
        )
        set_source(notebook["cells"][1], setup_code(family))
        for cell in notebook["cells"]:
            text = source_text(cell)
            text = text.replace(
                "metric curves for `psnr_db`, `ssim`, and `pixel_mae`",
                "metric curves for `psnr_db`, `ssim`, `lpips`, and `pixel_mae`",
            )
            text = text.replace(
                "selected from the loaded rows by PSNR first and SSIM second",
                "selected from the loaded rows by minimum LPIPS, with maximum PSNR as the tie-breaker",
            )
            if (
                "METRIC_OUTPUTS = recovery.export_metric_figures(" in text
                and "    combine_sampling_methods=True,\n" not in text
            ):
                text = text.replace(
                    "    show=True,\n)",
                    "    combine_sampling_methods=True,\n    show=True,\n)",
                    1,
                )
            if "GRID_OUTPUTS = recovery.export_recovery_grids(" in text:
                text = text.replace(
                    "    sampling_method=SAMPLING_METHODS[0],",
                    "    sampling_method=None,",
                )
            set_source(cell, text)

        notebook.setdefault("metadata", {}).pop("widgets", None)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(notebook, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
