"""Build clean weighted counterparts of the seven SD1.5 analysis notebooks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT / "analyze_results"
WEIGHTED_ROOT = ANALYSIS_ROOT / "weighted"
WEIGHTED_RATES = "{0.00125, 0.0025, 0.005, 0.01, 0.025}"
SHARED_RATES = "{0.00125, 0.0025, 0.005, 0.01}"

MAIN_NOTEBOOKS = {
    "prompt_matched": {
        "source": "unweighted/main/prompt_matched_results.ipynb",
        "old_tag": "unweighted/prompt_matched/sunset",
        "weighted_tag": "weighted/prompt_matched/sunset",
        "title": "Prompt-Matched In-Range",
    },
    "prompt_mismatched": {
        "source": "unweighted/main/prompt_mismatched_results.ipynb",
        "old_tag": "unweighted/prompt_mismatched/sunset",
        "weighted_tag": "weighted/prompt_mismatched/sunset",
        "title": "Prompt-Mismatched In-Range",
    },
    "out_of_range": {
        "source": "unweighted/main/out_of_range_results.ipynb",
        "old_tag": "unweighted/out_of_range/sunset",
        "weighted_tag": "weighted/out_of_range/sunset",
        "title": "Out-of-Range",
    },
}

ABLATION_NOTEBOOKS = {
    "prompt_matched": {
        "source": "unweighted/ablation/prompt_matched_cfg_ablation.ipynb",
        "weighted_experiment": "weighted_prompt_matched_in_range",
        "original_experiment": "prompt_matched_in_range",
        "title": "Prompt-Matched In-Range",
    },
    "prompt_mismatched": {
        "source": "unweighted/ablation/prompt_mismatched_cfg_ablation.ipynb",
        "weighted_experiment": "weighted_prompt_mismatched_in_range",
        "original_experiment": "prompt_mismatched_in_range",
        "title": "Prompt-Mismatched In-Range",
    },
    "out_of_range": {
        "source": "unweighted/ablation/out_of_range_cfg_ablation.ipynb",
        "weighted_experiment": "weighted_out_of_range",
        "original_experiment": "out_of_range",
        "title": "Out-of-Range",
    },
}


def read_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def clean_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(notebook)
    for cell in cleaned.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    metadata = cleaned.setdefault("metadata", {})
    metadata.pop("widgets", None)
    return cleaned


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def set_source(cell: dict[str, Any], text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": f"md-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": f"code-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def trial_convergence_code() -> str:
    return """TRIAL_COMPLETION = exp.ktilde_convergence_trial_completion_table(SD15_ROOT)
CONVERGENCE_FIGURE_DIR = SD15_ROOT / 'results' / 'weighted' / 'figures' / 'ktilde_convergence'
CONVERGENCE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
COMPLETION_PATH = CONVERGENCE_FIGURE_DIR / 'ktilde_convergence_five_trial_completion.csv'
TRIAL_COMPLETION.to_csv(COMPLETION_PATH, index=False)
display(TRIAL_COMPLETION)

CONVERGENCE_FIGURE_PATHS = {}
if not TRIAL_COMPLETION['status'].eq('complete').all():
    print('The five-trial convergence figures are pending. Run scripts/weighted/ktilde_convergence/list_all.sh to inspect all 20 jobs.')
else:
    CONVERGENCE_TRACES = exp.load_ktilde_convergence_trial_traces(SD15_ROOT)
    CONVERGENCE_SUMMARIES = exp.summarize_ktilde_convergence_trials(
        CONVERGENCE_TRACES,
        confidence_level=0.95,
    )
    CONVERGENCE_FIGURE_PATHS = exp.export_ktilde_convergence_trial_figure_set(
        CONVERGENCE_SUMMARIES,
        output_dir=CONVERGENCE_FIGURE_DIR,
        file_format='pdf',
        metrics=list(exp.KTILDE_TRIAL_CONVERGENCE_METRICS),
        show=True,
    )
    display(pd.Series({key: str(value) for key, value in CONVERGENCE_FIGURE_PATHS.items()}))
CONVERGENCE_FIGURE_PATHS
"""


def trial_convergence_markdown() -> str:
    return """## Five-Trial Algorithm 1 K-Tilde Convergence

The four saved S10000 artifacts remain fixed references. For every prompt, five
new S10000 estimates use disjoint latent-seed blocks and record relative
$\\ell_2$ error, relative $\\ell_\\infty$ error, the regularized Lambda
max-ratio, and the regularized maximum log-probability ratio every 10
iterations. The probability-based metrics use $\\zeta=1/2$.

Each curve is the arithmetic mean of the five trial values at that iteration.
Shading is the 95% Student-$t$ confidence interval computed on the original
metric scale; the mean and bounds are then displayed on the existing
logarithmic $y$-axis. The completion table remains visible while any of the 20
jobs is missing or incomplete.
"""


def weighted_main_setup_code(weighted_tag: str) -> str:
    return f"""from pathlib import Path
import importlib
import sys

import pandas as pd

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
WEIGHTED_BASE_TAG = '{weighted_tag}'
SAMPLING_METHODS = list(recovery.WEIGHTED_MAIN_SAMPLING_METHODS)
ALLOWED_SAMPLING_PERC = set(recovery.WEIGHTED_MAIN_RATES)
EXCLUDED_SAMPLING_CONDITIONS = set()
OUTPUT_ROOT = SD15_ROOT / 'results' / 'figures'

LPIPS_TABLE = recovery.ensure_lpips_metrics(
    SD15_ROOT,
    device='cpu',
)
analysis, COMPLETION_TABLE = recovery.load_weighted_main_analysis(
    SD15_ROOT,
    base_tag=WEIGHTED_BASE_TAG,
    output_root=OUTPUT_ROOT,
    include_partial=True,
)
ROWS = analysis.rows
MEAN_TABLE = analysis.mean_table
ACTIVE_TAG = analysis.active_tag
LOADED_TAGS = analysis.loaded_tags
OUTPUT_DIR = analysis.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPLETION_PATH = OUTPUT_DIR / 'weighted_main_completion.csv'
COMPLETION_TABLE.to_csv(COMPLETION_PATH, index=False)

print(f'Active tag: {{ACTIVE_TAG}}')
print(f'Loaded source tags: {{LOADED_TAGS}}')
print(f'Loaded {{len(ROWS)}} / 700 expected run rows.')
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


def main_comparison_code(old_tag: str) -> str:
    return f"""# This is a complete-pipeline comparison: S10000, zeta, weighting, and FFT normalization all change.
SHARED_PAPER_RATES = {SHARED_RATES}
WEIGHTED_ONLY_RATE = 0.025
original_analysis = recovery.load_recovery_analysis(
    SD15_ROOT,
    tag_group_candidates=recovery.split_tag_group_candidates('{old_tag}'),
    sampling_methods=SAMPLING_METHODS,
    allowed_sampling_percentages=SHARED_PAPER_RATES,
    excluded_sampling_conditions=EXCLUDED_SAMPLING_CONDITIONS,
    output_root=OUTPUT_ROOT,
)

comparison_keys = [
    'sampling_method',
    'sampling_condition',
    'reconstruction_condition',
    'samp_perc',
]
comparison_metrics = ['psnr_db', 'ssim', 'pixel_mae', 'grain', 'runtime_sec']
comparison_columns = comparison_keys + comparison_metrics
weighted_shared = (
    MEAN_TABLE[
        MEAN_TABLE['samp_perc'].astype(float).isin(SHARED_PAPER_RATES)
    ][comparison_columns].copy()
    if not MEAN_TABLE.empty
    else pd.DataFrame(columns=comparison_columns)
)
original_shared = (
    original_analysis.mean_table[comparison_columns].copy()
    if not original_analysis.mean_table.empty
    else pd.DataFrame(columns=comparison_columns)
)
PIPELINE_COMPARISON = weighted_shared.merge(
    original_shared,
    on=comparison_keys,
    how='outer',
    suffixes=('_weighted', '_original'),
    indicator=True,
)
for metric in comparison_metrics:
    PIPELINE_COMPARISON[f'{{metric}}_weighted_minus_original'] = (
        PIPELINE_COMPARISON[f'{{metric}}_weighted'] - PIPELINE_COMPARISON[f'{{metric}}_original']
    )

RATE_SCOPE = pd.DataFrame(
    {{
        'sampling_ratio': sorted(ALLOWED_SAMPLING_PERC),
        'comparison_scope': [
            'shared with original paper' if rate in SHARED_PAPER_RATES else 'weighted-only'
            for rate in sorted(ALLOWED_SAMPLING_PERC)
        ],
    }}
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_COMPARISON.to_csv(OUTPUT_DIR / 'weighted_vs_original_shared_rates.csv', index=False)
RATE_SCOPE.to_csv(OUTPUT_DIR / 'rate_scope.csv', index=False)
display(RATE_SCOPE)
display(PIPELINE_COMPARISON)
"""


def build_main_notebooks() -> None:
    for family, spec in MAIN_NOTEBOOKS.items():
        notebook = clean_notebook(read_notebook(ANALYSIS_ROOT / spec["source"]))
        set_source(
            notebook["cells"][0],
            f"""# Weighted S10000 {spec['title']} Recovery Results

This notebook is the weighted counterpart of the original recovery analysis. It uses the unitary Fourier operator and weighted least squares for the four S10000 source laws, uniform MCS, pure inverse-square sampling, and VDHH. All seven distributions use five sampling ratios and five trials. PSNR, SSIM, and LPIPS receive sampling-ratio sweeps. Outputs stay under `results/weighted/figures/`.

The four shared rates are compared with the original paper results as a complete-pipeline comparison. The `0.025` rate is explicitly labeled weighted-only.
""",
        )
        set_source(notebook["cells"][1], weighted_main_setup_code(spec["weighted_tag"]))
        for cell in notebook["cells"]:
            text = source_text(cell)
            text = text.replace(
                "metric curves for `psnr_db`, `ssim`, and `pixel_mae`",
                "metric curves for `psnr_db`, `ssim`, `lpips`, and `pixel_mae`",
            )
            if "METRIC_OUTPUTS = recovery.export_metric_figures(" in text:
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
        notebook["cells"].extend(
            [
                markdown_cell(
                    """## Complete-Pipeline Comparison with Original Results

The table below compares only the four shared sampling ratios. Differences cannot be attributed to weighting alone because the S10000 secant budget, $\\zeta=1/2$ regularization, and unitary Fourier convention also change.
"""
                ),
                code_cell(main_comparison_code(spec["old_tag"])),
            ]
        )
        write_notebook(WEIGHTED_ROOT / "main" / f"{family}_results.ipynb", notebook)


def ablation_comparison_code(original_experiment: str) -> str:
    return f"""# The comparison is pipeline-level; it is not an isolated estimate of the weighting effect.
SHARED_PAPER_RATES = {SHARED_RATES}
WEIGHTED_ONLY_RATE = 0.025
original_rows = cfgviz.load_cfg_ablation_rows(
    SD15_ROOT,
    experiment='{original_experiment}',
    sampling_percentages=SHARED_PAPER_RATES,
)

comparison_keys = ['distribution_key', 'line_condition', 'samp_perc']
comparison_metrics = ['psnr_db', 'ssim', 'pixel_mae']
weighted_summary = (
    rows[rows['samp_perc'].astype(float).isin(SHARED_PAPER_RATES)]
    .groupby(comparison_keys, dropna=False)[comparison_metrics]
    .mean()
    .reset_index()
)
original_summary = (
    original_rows.groupby(comparison_keys, dropna=False)[comparison_metrics]
    .mean()
    .reset_index()
)
PIPELINE_COMPARISON = weighted_summary.merge(
    original_summary,
    on=comparison_keys,
    how='outer',
    suffixes=('_weighted', '_original'),
    indicator=True,
)
for metric in comparison_metrics:
    PIPELINE_COMPARISON[f'{{metric}}_weighted_minus_original'] = (
        PIPELINE_COMPARISON[f'{{metric}}_weighted'] - PIPELINE_COMPARISON[f'{{metric}}_original']
    )

RATE_SCOPE = pd.DataFrame(
    {{
        'sampling_ratio': list(cfgviz.WEIGHTED_SAMPLING_PERCENTAGES),
        'comparison_scope': [
            'shared with original paper' if rate in SHARED_PAPER_RATES else 'weighted-only'
            for rate in cfgviz.WEIGHTED_SAMPLING_PERCENTAGES
        ],
    }}
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_COMPARISON.to_csv(OUTPUT_DIR / 'weighted_vs_original_shared_rates.csv', index=False)
RATE_SCOPE.to_csv(OUTPUT_DIR / 'rate_scope.csv', index=False)
display(RATE_SCOPE)
display(PIPELINE_COMPARISON)
"""


def build_ablation_notebooks() -> None:
    for family, spec in ABLATION_NOTEBOOKS.items():
        notebook = clean_notebook(read_notebook(ANALYSIS_ROOT / spec["source"]))
        set_source(
            notebook["cells"][0],
            f"""# Weighted S10000 {spec['title']} CFG Recovery Ablation

This notebook analyzes the weighted recovery-only CFG ablation at the same five sampling ratios and five trials as the weighted main experiment. CFG 1, 1.5, 3, and 5 are new runs; compatible unconditioned and CFG 7.5 references come only from the weighted main suite after strict manifest validation.

The four shared paper rates are compared as a complete-pipeline comparison, and `0.025` is labeled weighted-only.
""",
        )
        setup = source_text(notebook["cells"][1])
        original_experiment_line = next(
            line for line in setup.splitlines() if line.startswith("EXPERIMENT = ")
        )
        setup = setup.replace(
            original_experiment_line,
            f"EXPERIMENT = '{spec['weighted_experiment']}'",
        )
        output_line = next(line for line in setup.splitlines() if line.startswith("OUTPUT_DIR = "))
        setup = setup.replace(
            output_line,
            f"OUTPUT_DIR = SD15_ROOT / 'results' / 'weighted' / 'figures' / 'ablation' / '{family}' / 'sunset'",
        )
        setup = setup.replace(
            "import os\nimport sys",
            "import os\nimport sys\n\nimport pandas as pd",
        )
        set_source(notebook["cells"][1], setup)

        load_code = source_text(notebook["cells"][3])
        load_code = load_code.replace(
            "sync_report = cfgviz.sync_main_references(SD15_ROOT, experiment=EXPERIMENT)",
            "sync_report = cfgviz.sync_main_references(SD15_ROOT, experiment=EXPERIMENT, dry_run=True)",
        )
        load_code = load_code.replace(
            "rows = cfgviz.load_cfg_ablation_rows(SD15_ROOT, experiment=EXPERIMENT)",
            "rows = cfgviz.load_cfg_ablation_rows(\n"
            "    SD15_ROOT,\n"
            "    experiment=EXPERIMENT,\n"
            "    sampling_percentages=cfgviz.WEIGHTED_SAMPLING_PERCENTAGES,\n"
            ")",
        )
        set_source(notebook["cells"][3], load_code)
        set_source(
            notebook["cells"][2],
            """## Load Results

The weighted launcher performs strict manifest validation before it copies compatible unconditioned and CFG 7.5 references. This notebook reports the corresponding unsplit copy plan in dry-run mode, then loads all available weighted rows. The count table should reach five repeats for every sampling distribution, recovery line, and sampling ratio.
""",
        )
        notebook["cells"].extend(
            [
                markdown_cell(
                    """## Complete-Pipeline Comparison with Original Results

Only the four shared rates enter this comparison. The `0.025` ablation rows remain weighted-only.
"""
                ),
                code_cell(ablation_comparison_code(spec["original_experiment"])),
            ]
        )
        write_notebook(WEIGHTED_ROOT / "ablation" / f"{family}_cfg_ablation.ipynb", notebook)


def build_lambda_notebook() -> None:
    notebook = clean_notebook(
        read_notebook(ANALYSIS_ROOT / "unweighted" / "main" / "ktilde_lambda_comparison.ipynb")
    )
    set_source(
        notebook["cells"][0],
        """# Weighted S10000 K-Tilde / Lambda Comparison

This notebook audits the four S10000 source laws used by the weighted experiment suite. It converts stored Fourier energies to the unitary convention and computes Lambda tables with

$$\\mu_{1/2}(i)=\\tfrac12\\widetilde\\mu_{\\mathrm{S10000}}(i)+\\tfrac{1}{2n}.$$

The five-trial convergence study keeps raw K-tilde errors while applying $\\zeta=1/2$ to its sampling-law and Lambda diagnostics. Figures are written under `results/weighted/figures/`.
""",
    )
    setup = source_text(notebook["cells"][1])
    setup = setup.replace(
        "CATALOG = exp.load_ktilde_catalog(SD15_ROOT)",
        "S10000_CONFIG = SD15_ROOT / 'ktilde' / 'weighted' / 'config_convergence.json'\n"
        "ZETA = 0.5\n"
        "CATALOG = exp.load_ktilde_catalog(SD15_ROOT, config_path=S10000_CONFIG)",
    )
    setup = setup.replace(
        "TABLES = exp.build_lambda_tables(SD15_ROOT, skip_missing=True)",
        "TABLES = exp.build_lambda_tables(\n"
        "    SD15_ROOT,\n"
        "    config_path=S10000_CONFIG,\n"
        "    probability_regularization_zeta=ZETA,\n"
        "    skip_missing=True,\n"
        ")",
    )
    setup = setup.replace(
        "SD15_ROOT / 'ktilde' / 'unweighted' / f'{name}.npz'",
        "SD15_ROOT / 'ktilde' / 'weighted' / 'reference' / f'{name}.npz'",
    )
    setup = setup.replace(
        "**FFT normalization.** ",
        "**Weighted source-law audit.** ",
    )
    setup = setup.replace(
        '"The mismatch table is unchanged because this scale cancels."',
        '"Every sampling column uses zeta=1/2 regularization; this changes the absolute compatibility values."',
    )
    setup_lines = []
    for line in setup.splitlines(keepends=True):
        if "\\widetilde" in line:
            quote_index = line.index('"')
            line = f"{line[:quote_index]}r{line[quote_index:]}"
        setup_lines.append(line)
    setup = "".join(setup_lines)
    set_source(notebook["cells"][1], setup)

    for cell in notebook["cells"]:
        text = source_text(cell)
        text = text.replace(
            "results' / 'figures' / 'ktilde_convergence'",
            "results' / 'weighted' / 'figures' / 'ktilde_convergence'",
        )
        text = text.replace(
            "results' / 'figures' / 'lambda_figures'",
            "results' / 'weighted' / 'figures' / 'lambda_figures'",
        )
        text = text.replace(
            "results' / 'unweighted' / 'figures' / 'lambda_figures'",
            "results' / 'weighted' / 'figures' / 'lambda_figures'",
        )
        set_source(cell, text)
        if cell.get("id") == "explain-02":
            set_source(
                cell,
                """## Export Lambda Figures

This cell exports the absolute Lambda heatmap, the individual sampling-law
plots, and the compact sampling-law row. The displayed `FIGURE_PATHS` series
is the checklist of generated paper figures.
""",
            )
        if cell.get("id") == "ktilde-convergence-explain":
            set_source(cell, trial_convergence_markdown())
        if cell.get("id") == "ktilde-convergence-export":
            set_source(cell, trial_convergence_code())

    notebook["cells"] = notebook["cells"][:6]
    notebook["cells"].extend(
        [
            markdown_cell(
                """## Regularized S10000 Probability Audit

This audit records both the raw artifact statistics and the exact $\\zeta=1/2$ law used by reconstruction and Lambda analysis.
"""
            ),
            code_cell(
                """import numpy as np

probability_rows = []
for role, info in TABLES['bank'].items():
    raw = np.asarray(info['raw_probabilities'], dtype=np.float64).reshape(-1)
    effective = np.asarray(info['probabilities'], dtype=np.float64).reshape(-1)
    expected = 0.5 * raw + 0.5 / raw.size
    np.testing.assert_allclose(effective, expected, rtol=0.0, atol=2e-16)
    probability_rows.append(
        {
            'role': role,
            'artifact': info['name'],
            'n': raw.size,
            'zeta': info['probability_regularization_zeta'],
            'raw_sum': raw.sum(),
            'raw_min': raw.min(),
            'raw_max': raw.max(),
            'regularized_sum': effective.sum(),
            'regularized_min': effective.min(),
            'regularized_max': effective.max(),
            'required_floor': 0.5 / raw.size,
            'floor_satisfied': bool(effective.min() >= 0.5 / raw.size),
        }
    )
PROBABILITY_AUDIT = pd.DataFrame(probability_rows)
AUDIT_PATH = SD15_ROOT / 'results' / 'weighted' / 'figures' / 'ktilde_lambda' / 'probability_audit.csv'
AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
PROBABILITY_AUDIT.to_csv(AUDIT_PATH, index=False)
display(PROBABILITY_AUDIT)
AUDIT_PATH
"""
            ),
        ]
    )
    write_notebook(WEIGHTED_ROOT / "main" / "ktilde_lambda_comparison.ipynb", notebook)


def main() -> None:
    build_main_notebooks()
    build_ablation_notebooks()
    build_lambda_notebook()
    print(f"Built weighted analysis notebooks under {WEIGHTED_ROOT}")


if __name__ == "__main__":
    main()
