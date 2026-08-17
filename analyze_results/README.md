# Result Analysis

Shared loaders, validation routines, metric summaries, uncertainty estimates,
and plotting functions are located directly in this directory. Experiment
notebooks are organized by reconstruction objective:

- [unweighted/](unweighted/)
- [weighted/](weighted/)

Every notebook reads from its matching experiment directory under
[../results/](../results/) and writes generated PDFs and tables into a local
`figures/` folder beside that experiment's results.
