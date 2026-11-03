"""Build the compact weighted reconstruction comparisons."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "analyze_results"))
from paper_metrics import saved_reconstruction_metrics, lpips_lookup

OUTPUT = PROJECT / "results/paper_figures/weighted/reconstruction"
os.environ["TEXINPUTS"] = str(PROJECT / "analyze_results/tex") + os.pathsep + os.environ.get("TEXINPUTS", "")
SCENARIOS = {
    "prompt_matched": "sunset_beach_signal_sd15_512x512",
    "prompt_mismatched": "sunset_sandy_coast_signal_sd15_512x512",
    "out_of_range": "out_of_range_512x512",
}
METHODS = (
    ("k2", "cs", r"$\widetilde\mu_{c_{\mathrm{sb}}}$"),
    ("mcs", "mcs", r"$\mu_{\mathrm{MCS}}$"),
    ("inverse_square", "inverse_square", r"$\mu_{\mathrm{IS}}$"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
                         "font.family": "serif", "font.serif": ["Computer Modern Roman"],
                         "mathtext.fontset": "cm", "savefig.facecolor": "white"})
    records = []
    for scenario, dataset in SCENARIOS.items():
        result = PROJECT / "results/weighted" / scenario
        lpips = lpips_lookup(PROJECT)
        fig = plt.figure(figsize=(18, 10.5))
        grid = fig.add_gridspec(3, 5,
                               left=.085, right=.995, bottom=.005, top=.94,
                               wspace=.025, hspace=.075)
        for row, (law, method, label) in enumerate(METHODS):
            truth = fig.add_subplot(grid[row, 0])
            truth.imshow(Image.open(PROJECT / "datasets" / dataset / "gt_000.png").convert("RGB"))
            truth.set_axis_off()
            if row == 0:
                truth.set_title("Ground Truth", fontsize=20, pad=10)
            truth.text(-.075, .5, label, transform=truth.transAxes, ha="right", va="center", fontsize=24)
            truth.add_patch(Rectangle((0, 0), 1, 1, transform=truth.transAxes, fill=False,
                                      edgecolor="#bbbbbb", linewidth=.6, clip_on=False))
            for col, prompt in enumerate(("unprompted", "daytime_beach", "sunset_beach", "cat")):
                prompt_text = {"unprompted": "", "daytime_beach": "daytime beach",
                               "sunset_beach": "sunset beach", "cat": "cat"}[prompt]
                case = (("sample_k2_sunset_beach__recover_unprompted" if prompt == "unprompted"
                         else f"sample_k2_sunset_beach__recover_prompt_{prompt}") if method == "cs"
                        else f"baseline_{method}__recover_{prompt}")
                candidates = []
                for repeat in range(5):
                    run = result / law / case / method / "item_000/samp_0p03000" / f"rep_{repeat:02d}"
                    with np.load(run / "run_data.npz", allow_pickle=False) as saved:
                        assert np.isclose(saved["samp_perc"].item(), .03)
                        assert int(saved["repeat_id"].item()) == repeat
                    metric = saved_reconstruction_metrics(
                        run, lpips_fallback=lpips.get(run.relative_to(PROJECT).as_posix(), float("nan")))
                    assert np.isfinite(list(metric.values())).all()
                    candidates.append((metric["lpips"], -metric["psnr_db"], repeat, run, metric))
                _, _, repeat, run, metric = min(candidates, key=lambda r: r[:3])
                config = json.loads((run / "run_config.json").read_text())
                assert config["gen_recon"]["guidance_scale"] == 1
                assert config["sampling"]["weighted_ls"] is True
                assert config["reconstruction"]["prompt"] == prompt_text
                path = run / f"recon_{method}.png"
                image = Image.open(path).convert("RGB")
                assert image.size == (512, 512)
                ax = fig.add_subplot(grid[row, col + 1])
                ax.imshow(image, interpolation="none")
                ax.set_axis_off()
                if row == 0:
                    ax.set_title('$c_r=\\texttt{"' + prompt_text + '"}$', fontsize=20, pad=10)
                annotation = (f"LPIPS {metric['lpips']:.3f}\nW. Loss {metric['bp_best_loss']:.2e}\n"
                              f"PSNR {metric['psnr_db']:.2f} dB\nSSIM {metric['ssim']:.3f}")
                ax.text(.015, .985, annotation, transform=ax.transAxes, ha="left", va="top",
                        color="white", fontsize=10.5, linespacing=1.,
                        bbox={"boxstyle": "square,pad=0.2", "facecolor": "black", "edgecolor": "none", "alpha": .55})
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                      edgecolor="#bbbbbb", linewidth=.6, clip_on=False))
                records.append({"scenario": scenario, "sampling": law, "recovery_prompt": prompt,
                    "sampling_ratio": .03, "repeat_id": repeat, "selection": "Lowest LPIPS, then highest PSNR",
                    "image": path.relative_to(PROJECT).as_posix(), "sha256": sha(path), **metric})
        output = OUTPUT / f"compact_{scenario}_reconstructions.pdf"
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=.025)
        plt.close(fig)
        subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "1600", "-png", str(output),
                        str(output.with_suffix(""))], check=True)
        print("Built", output.name, flush=True)
    (OUTPUT / "compact_reconstructions_sources.json").write_text(json.dumps(
        {"panels": records}, indent=2) + "\n")
    print("Validated all 36 selected reconstructions.")


if __name__ == "__main__":
    main()
