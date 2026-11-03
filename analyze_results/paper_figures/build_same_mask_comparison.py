"""Build a fixed-trial Christoffel OOD comparison with two recovery prompts.

The saved Christoffel laws, masks, and optimizer protocols differ by study;
retain these details in the provenance.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "analyze_results")]
from paper_metrics import saved_reconstruction_metrics, lpips_lookup

OUTPUT = PROJECT / "results/paper_figures/unweighted/summary"
os.environ["TEXINPUTS"] = str(PROJECT / "analyze_results/tex") + os.pathsep + os.environ.get("TEXINPUTS", "")


def main():
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
                         "font.family": "serif", "font.serif": ["Computer Modern Roman"],
                         "mathtext.fontset": "cm", "savefig.facecolor": "white"})
    weighted = lpips_lookup(PROJECT)
    unweighted = lpips_lookup(PROJECT, unweighted=True)
    truth = PROJECT / "datasets/out_of_range_512x512/gt_000.png"
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    fig.subplots_adjust(left=.055, right=.995, bottom=.01, top=.92, wspace=.025, hspace=.08)
    records = []
    for row, study in enumerate(("weighted", "unweighted")):
        axes[row, 0].imshow(Image.open(truth).convert("RGB"))
        axes[row, 0].set_axis_off()
        axes[row, 0].text(-.1, .5, "Weighted" if row == 0 else "Unweighted",
                          transform=axes[row, 0].transAxes, ha="center", va="center",
                          rotation=90, fontsize=30)
        if row == 0:
            axes[row, 0].set_title("Ground Truth", fontsize=22.5, pad=9)
        panels = [(law, prefix, label, recovery)
                  for law, prefix, label in (
                      ("k0", "sample_k0_unconditioned", r"$\widetilde\mu_{c_{\mathrm{uc}}}$"),
                      ("k2", "sample_k2_sunset_beach", r"$\widetilde\mu_{c_{\mathrm{sb}}}$"))
                  for recovery in ("uc", "sb")]
        for col, (law, prefix, label, recovery) in enumerate(panels, 1):
            method = "cs"
            case = prefix + ("__recover_unprompted" if recovery == "uc" else "__recover_prompt_sunset_beach")
            weighted_run = PROJECT / f"results/weighted/out_of_range/{law}/{case}/cs/item_000/samp_0p01000/rep_00"
            if recovery == "uc":
                unweighted_run = PROJECT / f"unweighted/results/out_of_range/sunset/last3_{prefix}/{case}/cs/item_000/samp_0p01000/rep_00"
            else:
                unweighted_run = PROJECT / f"unweighted/results/ablation/out_of_range/sunset/last3_{prefix}/{case}_cfg1/cs/item_000/samp_0p01000/rep_00"
            run = weighted_run if study == "weighted" else unweighted_run
            config = json.loads((run / "run_config.json").read_text())
            with np.load(run / "run_data.npz", allow_pickle=False) as data:
                count = int(data["m_coeffs"])
                seed = int(data["rep_seed"])
                psnr = float(data["psnr_db"])
                ssim = float(data["ssim"])
                assert int(data["repeat_id"]) == 0 and np.isclose(float(data["samp_perc"]), .01)
                sampling_method = int(data["samp_method"])
            assert config["reconstruction"]["prompt"] == ("" if recovery == "uc" else "sunset beach")
            assert config["gen_recon"]["guidance_scale"] == (7.5 if study == "unweighted" and recovery == "uc" else 1.)
            # Recreate the actual unweighted mask and compare its full index set.
            from src.sampling import build_sampling_pattern
            from src.ktilde import load_ktilde_probabilities, regularize_sampling_probabilities
            probability = None
            if method == "cs":
                raw, _, _ = load_ktilde_probabilities(PROJECT / "ktilde", config["ktilde"]["name"])
                probability = regularize_sampling_probabilities(
                    raw, config["sampling"].get("probability_regularization_zeta", 0.))
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                indices, _, _ = build_sampling_pattern(512*512, count, sampling_method, probability, H=512, W=512)
            finally:
                np.random.set_state(state)
            with np.load(weighted_run / "run_data.npz", allow_pickle=False) as data:
                expected = data["sampled_indices"]
                assert int(data["rep_seed"]) == seed
            if study == "weighted":
                np.testing.assert_array_equal(np.sort(indices), np.sort(expected))
            image = run / f"recon_{method}.png"
            cache = weighted if row == 0 else unweighted
            metric = saved_reconstruction_metrics(
                run, lpips_fallback=float(cache.get(run.relative_to(PROJECT).as_posix(), float("nan"))))
            lpips, loss = metric["lpips"], metric["bp_best_loss"]
            psnr, ssim = metric["psnr_db"], metric["ssim"]
            ax = axes[row, col]
            ax.imshow(Image.open(image).convert("RGB"), interpolation="none")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(label + ", " + rf"$c_r=c_{{\mathrm{{{recovery}}}}}$", fontsize=22.5, pad=9)
            loss_label = "W. Loss" if row == 0 else "U. Loss"
            annotation = (f"LPIPS {lpips:.3f}\n{loss_label} {loss:.2e}\n"
                          f"PSNR {psnr:.2f} dB\nSSIM {ssim:.3f}")
            ax.text(.015, .985, annotation, transform=ax.transAxes, ha="left", va="top",
                    color="white", fontsize=13.5, linespacing=1.,
                    bbox={"boxstyle":"square,pad=0.2", "facecolor":"black", "edgecolor":"none", "alpha":.55})
            records.append({"study":study, "sampling":law, "repeat_id":0, "seed":seed,
                            "m_coeffs":count, "sampling_ratio":.01,
                            "recovery_prompt":config["reconstruction"]["prompt"],
                            "recovery_cfg":config["gen_recon"]["guidance_scale"],
                            "mask_sha256":hashlib.sha256(np.sort(indices).tobytes()).hexdigest(),
                            "weighted_mask_overlap":float(np.intersect1d(indices, expected).size/count),
                            "image":image.relative_to(PROJECT).as_posix(),
                            "image_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
                            "lpips":lpips,"bp_best_loss":loss,"psnr_db":psnr,"ssim":ssim,
                            "loss_source":"Saved best optimization loss; unweighted values converted by m/n"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "weighted_unweighted_ood_comparison"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "1800", "-png",
                    str(stem.with_suffix(".pdf")), str(stem)], check=True)
    stem.with_suffix(".json").write_text(json.dumps({
        "selection":"Trial 1 fixed, not selected by image quality",
        "scope":"Same target, trial seed and 1% sampling; conditional recovery CFG 1; empty-prompt unweighted references use nominal CFG 7.5; Christoffel masks and optimizer protocols differ",
        "panels":records},indent=2)+"\n")
    print("Built Christoffel OOD comparison with two sampling and two recovery prompts, trial 1.")


if __name__ == "__main__":
    main()
