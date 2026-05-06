# Datasets

This folder contains the image artifacts referenced by the paper experiments.

- `sunset_beach_signal_sd15_512x512`: prompt-matched in-range signal.
- `sunset_sandy_coast_signal_sd15_512x512`: prompt-mismatched in-range signal.
- `out_of_range_512x512`: out-of-range sunset image.

Each dataset folder contains `gt_000.png`, `dataset_index.json`, and `meta_000.json`.

The suite runner loads these artifacts directly. Regenerating the SD1.5-generated datasets is possible through:

```bash
python build_dataset.py --name sunset_beach_signal_sd15_512x512
python build_dataset.py --name sunset_sandy_coast_signal_sd15_512x512
```

The `out_of_range_512x512` artifact is copied data and should be treated as a fixed input.
