# Saved Results

The figure-reproduction data is published in the
[`paper-results-v1` GitHub Release](https://github.com/alexdelise/ActiveConditionalGCS/releases/tag/paper-results-v1).
The [data manifest](data_manifest.json) records archive sizes, checksums, file
counts, and reconstruction counts.

| Archive | Reconstructions | Compressed size |
| --- | ---: | ---: |
| Prompt matched | 600 | 563.02 MiB |
| Prompt mismatched | 600 | 523.81 MiB |
| Out of range | 600 | 642.52 MiB |
| Prompt-matched CFG ablation | 300 | 303.22 MiB |
| Prompt-mismatched CFG ablation | 300 | 285.60 MiB |
| Out-of-range CFG ablation | 300 | 349.41 MiB |
| Christoffel studies and targets | — | 138.82 MiB |
| **Total** | **2,700** | **2.74 GiB** |

## Contents

The release contains one archive for each main scenario, one for each
CFG-ablation scenario, and one for the Christoffel studies. It includes 1,800
main reconstructions and 900 new ablation reconstructions. Ablation analysis
reads its 300 CFG 1 reference rows directly from the main results.

Each reconstruction includes metrics, the complete saved optimization trace,
sampled frequencies and probabilities, the recovered image, the zero-filled
image, and metadata. The Christoffel archive includes the fixed sampling laws,
sampling-CFG estimates, cross-class estimates, 20 independent convergence
trials, scalar convergence traces, and fixed target datasets.

Optimizer states, latent checkpoints, and generated figures are not included.

## Download and Verify

Run from the repository root:

```bash
python scripts/release/download_results.py
```

The downloader checks each archive and every file inside it before installing
the data. It does not replace different existing local files. To select a
separate destination, use:

```text
python scripts/release/download_results.py --destination [checkout-path]
```

Example:

```bash
python scripts/release/download_results.py --destination results/download_verification
```

The [analysis guide](../../analyze_results/weighted/README.md) lists the
notebooks to run after downloading the data.
