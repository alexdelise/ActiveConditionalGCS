"""Render the fixed target images for the repository overview."""

from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analyze_results"))
import sd15_conditioning_experiment as experiment


def main() -> None:
    targets = (
        ("sunset_beach_signal_sd15_512x512", "In-Range Prompt-Matched"),
        ("sunset_sandy_coast_signal_sd15_512x512", "In-Range Prompt-Mismatched"),
        ("out_of_range_512x512", "Out-of-Range"),
    )
    with plt.rc_context(experiment.SD15_PRESENTATION_RC):
        figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
        for axis, (dataset, title) in zip(axes, targets):
            # Display the complete target without changing its crop or color scale
            with Image.open(ROOT / "datasets" / dataset / "gt_000.png") as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(title, fontsize=18, pad=10)
            axis.set_axis_off()
        figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.91, wspace=0.035)
        output = ROOT / "assets" / "target_images.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        # Rasterize the TeX PDF with Poppler so PNG output does not require dvipng
        with tempfile.TemporaryDirectory(prefix="gcs-target-panel-") as temporary:
            pdf = Path(temporary) / "targets.pdf"
            figure.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
            subprocess.run([
                "pdftoppm", "-png", "-singlefile", "-r", "150", str(pdf), str(output.with_suffix("")),
            ], check=True)
        plt.close(figure)
        print(output)


if __name__ == "__main__":
    main()
