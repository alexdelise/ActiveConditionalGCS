"""Build the fixed-trial out-of-range comparison from saved reconstructions."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "analyze_results"))
from paper_metrics import saved_reconstruction_metrics, lpips_lookup

TEX_ROOT = PROJECT / "analyze_results/tex"
os.environ["TEXINPUTS"] = str(TEX_ROOT) + os.pathsep + os.environ.get("TEXINPUTS", "")
RESULTS = PROJECT / "results/weighted/out_of_range"
OUTPUT = PROJECT / "results/paper_figures/weighted/out_of_range"
RATIOS = (.01, .02, .03, .04, .05)
METHODS = (
    ("k2", "sample_k2_sunset_beach__recover_prompt_sunset_beach", "cs",
     r"$\widetilde\mu_{c_{\mathrm{sb}}}$"),
    ("inverse_square", "baseline_inverse_square__recover_sunset_beach", "inverse_square",
     r"$\mu_{\mathrm{IS}}$"),
    ("mcs", "baseline_mcs__recover_sunset_beach", "mcs", r"$\mu_{\mathrm{MCS}}$"),
)


def main():
    cached = lpips_lookup(PROJECT)
    values = np.empty((5, 3, 5))
    losses = np.empty_like(values)
    paths = {}
    seeds = {}
    for trial in range(5):
        for row, (law, case, method, _) in enumerate(METHODS):
            for col, ratio in enumerate(RATIOS):
                directory = (RESULTS / law / case / method / "item_000" /
                             f"samp_{ratio:.5f}".replace(".", "p") / f"rep_{trial:02d}")
                relative = directory.relative_to(PROJECT).as_posix()
                metric = saved_reconstruction_metrics(
                    directory, lpips_fallback=float(cached.get(relative, float("nan"))))
                values[trial, row, col] = metric["lpips"]
                path = directory / f"recon_{method}.png"
                assert path.is_file(), path
                paths[trial, row, col] = path
                with np.load(directory / "run_data.npz", allow_pickle=False) as data:
                    seeds[trial, row, col] = int(data["rep_seed"].item())
                    losses[trial, row, col] = metric["bp_best_loss"]
    assert np.isfinite(values).all()
    assert np.isfinite(losses).all() and (losses >= 0).all()
    # Equal weight to each method/ratio cell; standardization prevents high-variance
    # cells from dominating selection. Select one entire trial, never individual panels.
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    scores = np.mean(((values - mean) / np.maximum(std, 1e-12)) ** 2, axis=(1, 2))
    trial = 0  # Trial 1, fixed throughout at the author's request.
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
                         "font.family": "serif",
                         "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 21,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(14.8, 7.0))
    grid = fig.add_gridspec(3, 7, width_ratios=[1, .38, 1, 1, 1, 1, 1],
                           left=.005, right=.995, bottom=.015, top=.92,
                           wspace=.035, hspace=.06)
    records = []
    for row, (law, _, _, label) in enumerate(METHODS):
        if row == 1:
            truth = fig.add_subplot(grid[row, 0])
            truth.imshow(Image.open(PROJECT / "datasets/out_of_range_512x512/gt_000.png"))
            truth.set_axis_off()
            truth.set_title("Ground Truth", fontsize=21, pad=11)
        for col, ratio in enumerate(RATIOS):
            ax = fig.add_subplot(grid[row, col + 2])
            image = Image.open(paths[trial, row, col]).convert("RGB")
            assert image.size == (512, 512)
            ax.imshow(image, interpolation="none")
            ax.set_axis_off()
            annotation = (f"LPIPS {values[trial, row, col]:.3f}\n"
                          f"W. Loss {losses[trial, row, col]:.2e}")
            ax.text(.015, .985, annotation, transform=ax.transAxes, ha="left", va="top",
                    color="white", fontsize=11.25, linespacing=1.,
                    bbox={"boxstyle": "square,pad=0.2", "facecolor": "black",
                          "edgecolor": "none", "alpha": .55})
            if row == 0:
                ax.set_title(rf"$m/n={ratio:.2f}$", fontsize=21, pad=11)
            if col == 0:
                ax.text(-.07, .5, label, transform=ax.transAxes, ha="right", va="center",
                        fontsize=24)
            records.append({"sampling": law, "ratio": ratio, "trial": trial + 1,
                            "repeat_id": trial, "seed": seeds[trial, row, col],
                            "lpips": float(values[trial, row, col]),
                            "bp_best_loss": float(losses[trial, row, col]),
                            "image": paths[trial, row, col].relative_to(PROJECT).as_posix()})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "sampling_comparison.pdf", dpi=300, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "2600", "-png",
                    str(OUTPUT / "sampling_comparison.pdf"),
                    str(OUTPUT / "sampling_comparison")], check=True)
    metadata = {"selection": "Trial 1 fixed throughout at the author's request",
                "trial_scores": scores.tolist(), "selected_trial": trial + 1, "panels": records}
    (OUTPUT / "sampling_comparison_selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"selected_trial": trial + 1, "trial_scores": scores.tolist()}, indent=2))


if __name__ == "__main__":
    main()
