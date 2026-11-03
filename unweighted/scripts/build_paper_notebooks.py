"""Generate the unweighted and selected-comparison notebooks."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
SETUP = '''from pathlib import Path
import sys
from IPython.display import display

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / 'run_conditioning_regression.py').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'analyze_results')]
from unweighted.analyze_results import analysis as study
import sd15_recovery_analysis as recovery
import sd15_cfg_ablation_analysis as cfg
'''


def write_notebook(path, sections):
    cells = []
    for kind, source in sections:
        cell = dict(id=f"cell-{len(cells):03d}", cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    data = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + '\n')


def main():
    for ablation in (False, True):
        folder = 'ablation' if ablation else 'main'
        suffix = 'cfg_ablation' if ablation else 'results'
        for scenario in study_scenarios():
            setup = SETUP + f"\nROWS = study.load_scenario('{scenario}', ablation={ablation})\nOUTPUT_DIR = ROOT / 'unweighted/results/{'ablation/' if ablation else ''}{scenario}/sunset/figures'\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\nprint(f'Loaded {{len(ROWS)}} / {560 if ablation else 840} observations')\ndisplay(ROWS.groupby([{repr('distribution_key' if ablation else 'sampling_condition')}, {repr('line_condition' if ablation else 'reconstruction_condition')}]).size().unstack())\n"
            curves = ("study.plot_cfg_curves(ROWS, OUTPUT_DIR, show=True)" if ablation else
                      "recovery.METRIC_LABELS['bp_best_loss'] = 'Best Unweighted Loss'\nrecovery.export_metric_figures(ROWS, OUTPUT_DIR, sweep_metrics=study.METRICS, combined_metrics=(), combine_sampling_methods=True, sampling_xscale='log', show=True)")
            panels = ("cfg.plot_reconstruction_panel(ROWS, sd15_root=ROOT, output_dir=OUTPUT_DIR, samp_perc=.01, show=True)" if ablation else
                      "recovery.export_recovery_grids(ROWS, ROOT, OUTPUT_DIR, sampling_method=None, sampling_percentage=.01, show=True)")
            forest = "study.plot_scenario_forest(ROWS, OUTPUT_DIR, ablation=" + str(ablation) + ", show=True)"
            write_notebook(ROOT / f'unweighted/analyze_results/{folder}/{scenario}_{suffix}.ipynb', [
                ('markdown', f'# Unweighted {scenario.replace("_", " ")}'+(' CFG ablation' if ablation else '')+'\n\nReconstruction metrics across seven sampling ratios and five trials.\n'),
                ('code', setup), ('markdown', '## Reconstruction metrics\n'), ('code', curves),
                ('markdown', '## Recovered images\n'), ('code', panels),
                ('markdown', '## Optimization trajectories\n'), ('code', 'study.export_traces(ROWS, OUTPUT_DIR, show=True)'),
                ('markdown', '## Aggregate metrics\n'), ('code', forest)])
        setup = SETUP + f"\nFRAMES = [study.load_scenario(s, ablation={ablation}) for s in study.SCENARIOS]\nOUTPUT_DIR = ROOT / 'unweighted/results/{'ablation/' if ablation else ''}summary/figures'\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        name = 'ablation/unweighted_ablation_forest.ipynb' if ablation else 'unweighted_main_forest.ipynb'
        write_notebook(ROOT / f'unweighted/analyze_results/{name}', [
            ('markdown', '# Unweighted aggregate results\n\nAggregate reconstruction metrics across all three experiments.\n'),
            ('code', setup), ('code', f'study.plot_combined_forest(FRAMES, OUTPUT_DIR, ablation={ablation}, show=True)')])
    setup = '''from pathlib import Path
import sys
from IPython.display import display, Image
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / 'run_conditioning_regression.py').is_file())
sys.path[:0] = [str(ROOT / 'analyze_results'), str(ROOT / 'analyze_results/paper_figures')]
'''
    sections = [('markdown', '# Selected reconstruction comparisons\n\nRebuild selected reconstruction comparisons from saved runs.\n'), ('code', setup)]
    for module, title in [('build_ood_sampling_comparison', 'Out-of-range sampling comparison'), ('build_compact_reconstructions', 'Weighted reconstruction comparisons'), ('build_same_mask_comparison', 'Weighted and unweighted comparison'), ('build_unweighted_hallucination_gallery', 'Selected unweighted reconstructions')]:
        sections.extend([('markdown', '## '+title+'\n'), ('code', f'import {module} as builder\nbuilder.main()\nfor preview in sorted(builder.OUTPUT.glob("*.png")):\n    display(Image(filename=str(preview), width=1100))')])
    write_notebook(ROOT / 'analyze_results/paper_figures/paper_figures.ipynb', sections)


def study_scenarios():
    return ('prompt_matched', 'prompt_mismatched', 'out_of_range')


if __name__ == '__main__':
    main()
