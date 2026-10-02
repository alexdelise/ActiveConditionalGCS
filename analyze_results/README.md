# Result Analysis

Shared loaders, metric summaries, uncertainty calculations, and plotting
functions are located directly in this directory. The
[weighted notebooks](weighted/README.md) reproduce the main reconstruction
figures, CFG ablations, and Christoffel studies.

Download the saved data using [the release tools](../scripts/release/README.md).
Run each notebook from its own directory or the repository root. Figures use
the shared Computer Modern/LaTeX style and are saved as PDFs under each
experiment's results folder.

To execute all nine public notebooks from the repository root and keep their
execution outputs outside Git:

```bash
mkdir -p results/notebook_runs
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output-dir=results/notebook_runs \
  analyze_results/weighted/*/*_results.ipynb \
  analyze_results/weighted/ablation/*_cfg_ablation.ipynb \
  analyze_results/weighted/weighted_main_forest.ipynb \
  analyze_results/weighted/ablation/weighted_ablation_forest.ipynb \
  analyze_results/weighted/ktilde/ktilde_analysis.ipynb
```
