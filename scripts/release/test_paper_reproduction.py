"""Offline regression checks for the paper's metric and aggregation conventions."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "analyze_results")]
from paper_metrics import saved_reconstruction_metrics, lpips_lookup
from scripts.release import download_results as download
from unweighted.analyze_results import analysis
import sd15_recovery_analysis as recovery
import sd15_cfg_ablation_analysis as cfg


class PaperReproductionTests(unittest.TestCase):
    def test_selective_ablation_download_includes_main_and_support(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = []
            for name, requires in (("unweighted_main", ["unweighted_support"]),
                                   ("unweighted_ablation", ["unweighted_main"]),
                                   ("unweighted_support", [])):
                contents = name.encode()
                filename = name + ".tar.gz"
                (root / filename).write_bytes(contents)
                assets.append(dict(name=name, filename=filename, requires=requires, file_count=1,
                                   archive_size_bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest()))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(published=True, repository="example/repo", release_tag="data", assets=assets)))
            argv = ["download", "--manifest", str(manifest), "--from-directory", str(root),
                    "--destination", str(root / "checkout"), "--assets", "unweighted_ablation"]
            with patch.object(sys, "argv", argv), patch.object(download, "extract_verified", return_value=1) as extract:
                download.main()
            self.assertEqual({call.args[0].name for call in extract.call_args_list},
                             {asset["filename"] for asset in assets})

    def test_conversion_matches_the_actual_fourier_operator_and_reductions(self):
        import torch
        from src.sampling import MeasurementOperator
        from src.reconstruction import measurement_loss
        with patch("src.sampling.DEVICE", torch.device("cpu")):
            indices = np.array([0, 5, 13, 24, 37, 48, 63])
            old = MeasurementOperator(indices, N=64, C=3, H=8, W=8, fft_normalization="backward")
            new = MeasurementOperator(indices, N=64, C=3, H=8, W=8, fft_normalization="ortho")
            image = torch.arange(192, dtype=torch.float32).reshape(1, 3, 8, 8) / 192
            target = torch.zeros(21, dtype=torch.complex64)
            legacy, _, _ = measurement_loss(image, measurements=target, measurement_operator=old,
                sigma_y=0., ls_weights=None, loss_reduction="mean")
            canonical, _, _ = measurement_loss(image, measurements=target, measurement_operator=new,
                sigma_y=0., ls_weights=None, loss_reduction="measurement_sum_channel_mean")
            torch.testing.assert_close(legacy * len(indices)/64, canonical)

    def test_saved_best_not_final_or_png_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_config.json").write_text(json.dumps({"sampling": {"weighted_ls": True}}))
            np.savez(run / "run_data.npz", psnr_db=20., ssim=.5, lpips=.3,
                     bp_best_loss=2., final_data_loss=9., H=8, W=8, m_coeffs=7)
            metric = saved_reconstruction_metrics(run, lpips_fallback=.8)
            self.assertEqual(metric["bp_best_loss"], 2.)
            self.assertEqual(metric["lpips"], .3)

    def test_unweighted_uses_actual_integer_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_config.json").write_text(json.dumps({"sampling": {"weighted_ls": False}}))
            np.savez(run / "run_data.npz", psnr_db=20., ssim=.5, bp_best_loss=4., H=8, W=8, m_coeffs=7)
            self.assertEqual(saved_reconstruction_metrics(run, lpips_fallback=.3)["bp_best_loss"], 4.*7/64)

    def test_missing_metric_fails_instead_of_recomputing(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_config.json").write_text(json.dumps({"sampling": {"weighted_ls": False}}))
            np.savez(run / "run_data.npz", psnr_db=20., ssim=.5, bp_best_loss=4., H=8, W=8, m_coeffs=7)
            with self.assertRaises(ValueError):
                saved_reconstruction_metrics(run)

    def test_loss_conversion_cannot_be_applied_twice(self):
        frame = pd.DataFrame({"m_coeffs": [41, 2621], "bp_best_loss": [3., 4.]})
        converted = analysis.convert_loss_convention(frame)
        np.testing.assert_allclose(converted.bp_best_loss, np.array([3.*41, 4.*2621]) / 262144)
        with self.assertRaises(ValueError):
            analysis.convert_loss_convention(converted)

    def test_legacy_sidecar_paths_are_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "unweighted/results/metrics/lpips.csv"
            sidecar.parent.mkdir(parents=True)
            pd.DataFrame({"artifact_relpath": ["results/unweighted/a"], "lpips": [.3]}).to_csv(sidecar, index=False)
            self.assertEqual(lpips_lookup(root, unweighted=True), {"unweighted/results/a": .3})

    def test_interval_uses_five_trial_averages_not_ten_independent_rows(self):
        frame = pd.DataFrame([dict(sampling="s", recovery="r", repeat_id=j, samp_perc=ratio, lpips=float(j + offset))
                              for j in range(5) for ratio, offset in ((.01, 0), (.02, 10))])
        summary = recovery.build_aggregate_forest_summary(frame, row_column="sampling", line_column="recovery",
                    row_keys=["s"], line_keys=["r"], expected_sampling_ratios=[.01, .02], metrics=["lpips"])
        means = np.arange(5, dtype=float) + 5
        self.assertEqual(summary.iloc[0].trial_count, 5)
        self.assertAlmostEqual(summary.iloc[0].lpips_mean, means.mean())
        self.assertAlmostEqual(summary.iloc[0].lpips_ci, t.ppf(.975, 4) * means.std(ddof=1)/np.sqrt(5))

    def test_cfg_controls_are_read_from_main_not_copied(self):
        distribution = cfg.experiment_distributions("out_of_range")[0]
        groups = cfg.case_root_candidate_groups(ROOT, distribution, "cfg7p5")
        self.assertTrue(all("reference_cfg7p5" not in str(p) for group in groups for p in group))
        self.assertTrue(all("/ablation/" not in str(p) for group in groups for p in group))


if __name__ == "__main__":
    unittest.main()
