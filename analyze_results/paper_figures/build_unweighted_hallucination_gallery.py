"""Build a three-row, 1%-sampling gallery without modifying saved images."""
from pathlib import Path
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
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "analyze_results"))
from paper_metrics import saved_reconstruction_metrics, lpips_lookup

OUTPUT = PROJECT / "results/paper_figures/unweighted/summary"
os.environ["TEXINPUTS"] = str(PROJECT / "analyze_results/tex") + os.pathsep + os.environ.get("TEXINPUTS", "")
SCENARIOS = (
    ("prompt_matched", "matched", "In-Range Prompt-Matched"),
    ("prompt_mismatched", "mismatched", "In-Range Prompt-Mismatched"),
    ("out_of_range", "ood", "Out-of-Range"),
)
# Selected saved trials, inspected against the targets. Trial indices are zero-based.
SUNSET_EXAMPLES = {
    "prompt_matched": (2, 2),
    "prompt_mismatched": (2, 2),
    "out_of_range": (2, 4),
}
DATASETS = {"prompt_matched": "sunset_beach_signal_sd15_512x512",
            "prompt_mismatched": "sunset_sandy_coast_signal_sd15_512x512",
            "out_of_range": "out_of_range_512x512"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_path(scenario, method, prompt, ratio, repeat):
    split = "first4" if ratio < .0025 else "last3"
    tag = f"{split}_sample_k2_sunset_beach" if method == "cs" else f"{split}_mcs_recover_{prompt}"
    case = (f"sample_k2_sunset_beach__recover_prompt_{prompt}" if method == "cs"
            else f"baseline_mcs__recover_{prompt}")
    return (PROJECT / "unweighted/results" / scenario / "sunset" / tag / case / method /
            "item_000" / f"samp_{ratio:.5f}".replace(".", "p") / f"rep_{repeat:02d}")


def main():
    lpips = lpips_lookup(PROJECT, unweighted=True)
    selected = json.loads((Path(__file__).with_name("selected_hallucinations.json")).read_text())
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
                         "font.family": "serif", "font.serif": ["Computer Modern Roman"],
                         "mathtext.fontset": "cm", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 5, figsize=(15, 9.3))
    fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.96, wspace=.025, hspace=.075)
    records = []
    for row, (scenario, short, _) in enumerate(SCENARIOS):
        truth = PROJECT / "datasets" / DATASETS[scenario] / "gt_000.png"
        axes[row, 0].imshow(Image.open(truth).convert("RGB"))
        axes[row, 0].set_axis_off()
        if row == 0:
            axes[row, 0].set_title("Ground Truth", fontsize=18.75, pad=8)
        cat_examples = []
        for method in ("cs", "mcs"):
            record = next(r for r in selected if r["scenario"] == scenario and
                          r["sampling"] == ("k2_sunset_beach" if method == "cs" else method) and r["recovery_prompt"] == "cat")
            image = PROJECT / record["image"]
            assert sha(image) == record["sha256"]
            cat_examples.append((image.parent, method, "Selected by visual inspection", None))
        first, second = SUNSET_EXAMPLES[scenario]
        settings = [
            (run_path(scenario, "mcs", "sunset_beach", .01, first), "mcs",
             "Sunset-beach example selected for invented scene content, not blur", None),
            (run_path(scenario, "cs", "sunset_beach", .01, second), "cs",
             "Selected for altered trees, shoreline, or scene structure", None),
            cat_examples[0],
            cat_examples[1],
        ]
        for column, (run, sampler, reason, talk_image) in enumerate(settings, start=1):
                config = json.loads((run / "run_config.json").read_text())
                assert config["sampling"]["weighted_ls"] is False
                assert config["gen_recon"]["guidance_scale"] == 7.5
                assert config["reconstruction_solver"]["outer_iterations"] == 500
                prompt = config["reconstruction"]["prompt"]
                prompt_code = {"cat": "ca", "sunset beach": "sb", "daytime beach": "db"}[prompt]
                with np.load(run / "run_data.npz", allow_pickle=False) as saved:
                    ratio = float(saved["samp_perc"].item())
                    repeat = int(saved["repeat_id"].item())
                    seed = int(saved["rep_seed"].item())
                    count = int(saved["m_coeffs"].item())
                    psnr = float(saved["psnr_db"].item())
                    ssim = float(saved["ssim"].item())
                assert np.isclose(ratio, .01)
                path = run / f"recon_{sampler}.png"
                ax = axes[row, column]
                image = Image.open(path).convert("RGB")
                assert image.size == (512, 512)
                ax.imshow(image, interpolation="none")
                ax.set_axis_off()
                sampling = r"$\widetilde{\mu}_{c_{\mathrm{sb}}}$" if sampler == "cs" else r"$\mu_{\mathrm{MCS}}$"
                if row == 0:
                    ax.set_title(sampling + r", " + rf"$c_r=c_{{\mathrm{{{prompt_code}}}}}$",
                                 fontsize=18.75, pad=8)
                metric = saved_reconstruction_metrics(
                    run, lpips_fallback=float(lpips.get(run.relative_to(PROJECT).as_posix(), float("nan"))))
                perceptual = metric["lpips"]
                psnr, ssim = metric["psnr_db"], metric["ssim"]
                loss = metric["bp_best_loss"]
                annotation = (f"LPIPS {perceptual:.3f}\nU. Loss {loss:.2e}\n"
                              f"PSNR {psnr:.2f} dB\nSSIM {ssim:.3f}")
                # Match the effective annotation size of the full-grid panels
                # when both PDFs are inserted at the manuscript's full text width.
                ax.text(.015, .985, annotation, transform=ax.transAxes, ha="left", va="top",
                        color="white", fontsize=11.25, linespacing=1.,
                        bbox={"boxstyle": "square,pad=0.2", "facecolor": "black", "edgecolor": "none", "alpha": .55})
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                      edgecolor="#bbbbbb", linewidth=.6, clip_on=False))
                records.append({"row": row + 1, "column": column + 1,
                    "scenario": scenario, "sampling": "k2_sunset_beach" if sampler == "cs" else "mcs",
                    "recovery_prompt": prompt, "sampling_ratio": ratio, "repeat_id": repeat,
                    "seed": seed, "selection": reason, "image": path.relative_to(PROJECT).as_posix(),
                    "sha256": sha(path),
                    "lpips": perceptual, "psnr_db": psnr, "ssim": ssim,
                    "bp_best_loss": loss,
                    "loss_source": "Saved best optimization loss converted by m/n"})
        axes[row, 0].add_patch(Rectangle((0, 0), 1, 1, transform=axes[row, 0].transAxes,
                                       fill=False, edgecolor="#bbbbbb", linewidth=.6, clip_on=False))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "selected_hallucinations.pdf"
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "2400", "-png", str(output),
                    str(output.with_suffix(""))], check=True)
    (OUTPUT / "selected_hallucinations_sources.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Built 12 selected examples: the fixed saved reconstructions.")


if __name__ == "__main__":
    main()
