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

The complete local main results occupy approximately 6.01 GiB by file size,
and the local CFG-ablation results occupy 2.85 GiB. The release omits redundant
exports and files unnecessary for visualization; it does not remove them from
the local results.

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

Optimizer states, latent checkpoints, discarded settings, diagnostics,
redundant aggregate CSVs, and generated figures are not included. Numerical
artifacts are retained unchanged, including their checksum-linked provenance.
JSON sidecar paths are made relative to the repository where applicable.

## Download and Verify

Run from the repository root:

```bash
python scripts/release/download_results.py
```

The downloader checks each archive and every file inside it before installing
the data. It does not replace different existing local files. Use an empty
destination when verifying data separately:

```bash
python scripts/release/download_results.py --destination /path/to/checkout
```

For locally prepared archives, use:

```bash
python scripts/release/download_results.py \
  --from-directory results/releases/paper-results-v1 \
  --destination /path/to/empty/checkout
```

The [analysis guide](../../analyze_results/weighted/README.md) lists the
notebooks to run after downloading the data.

## Prepare a Release

```bash
python scripts/release/package_results.py --inventory-only
python scripts/release/package_results.py --output results/releases/paper-results-v1
```

Packaging validates the completed reconstruction grid and writes archives
under the ignored results directory. It never changes the source results.
An existing archive is not overwritten. Choose a new output directory to
prepare another version.

Upload the archives as GitHub Release assets, not as Git commits. Include the
generated `data_manifest.json` with the release and update the checked-in
manifest after publication. Set its `published` flag only when the matching
assets are accessible. No upload is performed by the packaging command.
