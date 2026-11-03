# Result Analysis

Shared loaders, metric summaries, uncertainty calculations, and plotting
functions are located directly in this directory. The
[weighted notebooks](weighted/README.md) reproduce the main reconstruction
figures, CFG ablations, and Christoffel studies. The
[unweighted notebooks](../unweighted/analyze_results/README.md) follow the same
organization. [Selected reconstruction comparisons](paper_figures/paper_figures.ipynb)
are grouped with their builders in [paper_figures/](paper_figures/).

Download the saved data using [the release tools](../scripts/release/README.md).
Run each notebook from its own directory or the repository root. Figures use
the shared Computer Modern/LaTeX style and are saved as PDFs under each
experiment's results folder.

To execute a notebook and save its execution output outside Git, use:

```text
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir=[output-directory] [notebook-path]
```

Execute all analysis notebooks from the repository root:

```bash
mkdir -p results/notebook_runs
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output-dir=results/notebook_runs \
  analyze_results/weighted/*/*_results.ipynb \
  analyze_results/weighted/ablation/*_cfg_ablation.ipynb \
  analyze_results/weighted/weighted_main_forest.ipynb \
  analyze_results/weighted/ablation/weighted_ablation_forest.ipynb \
  analyze_results/weighted/ktilde/ktilde_analysis.ipynb \
  unweighted/analyze_results/main/*_results.ipynb \
  unweighted/analyze_results/ablation/*_cfg_ablation.ipynb \
  unweighted/analyze_results/unweighted_main_forest.ipynb \
  unweighted/analyze_results/ablation/unweighted_ablation_forest.ipynb \
  unweighted/analyze_results/ktilde/ktilde_analysis.ipynb \
  analyze_results/paper_figures/paper_figures.ipynb
```
