"""Check saved metrics, trial-level intervals, and reproduced paper assets."""
from pathlib import Path
import argparse
import base64
import collections
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "analyze_results")]
import importlib.util
from paper_metrics import saved_reconstruction_metrics, lpips_lookup
from unweighted.analyze_results import analysis as unweighted
import sd15_recovery_analysis as recovery


def load_weighted_rows(*, ablation=False):
    frames = []
    for scenario in unweighted.SCENARIOS:
        path = ROOT / "analyze_results/weighted" / ("ablation" if ablation else scenario) / "analysis.py"
        spec = importlib.util.spec_from_file_location(f"verify_weighted_{scenario}_{ablation}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frame = (module.load_rows(scenario) if ablation else module.load_rows()).copy()
        expected = module.EXPECTED_ROWS_PER_SCENARIO if ablation else module.EXPECTED_ROWS
        if len(frame) != expected or not np.isfinite(frame[list(unweighted.METRICS)].to_numpy(float)).all():
            raise ValueError(f"Incomplete weighted results: {scenario}, ablation={ablation}")
        frame["scenario"] = scenario
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_calculations():
    main, ablation = unweighted.load_results()
    studies = [("weighted_main", load_weighted_rows(), False),
               ("weighted_cfg", load_weighted_rows(ablation=True), False),
               ("unweighted_main", pd.concat(main), True),
               ("unweighted_cfg", pd.concat(ablation), True)]
    lookups = {flag: lpips_lookup(ROOT, unweighted=flag) for flag in (False, True)}
    saved_cache = {}
    pairs = observations = 0
    for name, rows, uw in studies:
        if "artifact_relpath" not in rows:
            rows = rows.copy()
            rows["artifact_relpath"] = rows["_run_data_path"].map(lambda p: Path(p).parent.relative_to(ROOT).as_posix())
        cfg = name.endswith("cfg")
        row_key = "distribution_key" if name == "unweighted_cfg" else "sampling_condition"
        line_key = "line_condition" if cfg else "reconstruction_condition"
        ratios = unweighted.RATIOS if uw else (.01, .02, .03, .04, .05)
        for row in rows.itertuples():
            path = row.artifact_relpath
            if path not in saved_cache:
                saved_cache[path] = saved_reconstruction_metrics(ROOT / path,
                    lpips_fallback=lookups[uw].get(path, float("nan")))
            for metric, value in saved_cache[path].items():
                np.testing.assert_allclose(getattr(row, metric), value, rtol=1e-12, atol=1e-12)
            observations += 1
        for scenario, frame in rows.groupby("scenario", sort=False):
            summary = recovery.build_aggregate_forest_summary(frame, row_column=row_key, line_column=line_key,
                row_keys=list(frame[row_key].unique()), line_keys=list(frame[line_key].unique()), expected_sampling_ratios=ratios)
            summary = summary.set_index([row_key, line_key])
            for keys, group in frame.groupby([row_key, line_key], sort=False):
                assert len(group) == 5 * len(ratios), (name, scenario, keys)
                assert not group.duplicated(["samp_perc", "repeat_id"]).any()
                for metric in unweighted.METRICS:
                    trial = np.array([group[group.repeat_id.eq(j)][metric].to_numpy(float).mean() for j in range(5)])
                    expected_ci = t.ppf(.975, 4) * trial.std(ddof=1) / np.sqrt(5)
                    np.testing.assert_allclose(summary.loc[keys, f"{metric}_mean"], trial.mean(), rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(summary.loc[keys, f"{metric}_ci"], expected_ci, rtol=1e-12, atol=1e-12)
                    pairs += 1
        print(name, len(rows), "saved observations and intervals PASS", flush=True)
    return dict(observations=observations, distinct_runs=len(saved_cache), forest_estimates=pairs)


def check_panels():
    base = ROOT / "results/paper_figures"
    groups = [(base / "weighted/out_of_range/sampling_comparison_selection.json", "main"),
              (base / "weighted/reconstruction/compact_reconstructions_sources.json", "compact"),
              (base / "unweighted/summary/weighted_unweighted_ood_comparison.json", "comparison"),
              (base / "unweighted/summary/selected_hallucinations_sources.json", "gallery")]
    by_pdf = collections.defaultdict(list)
    lookups = {flag: lpips_lookup(ROOT, unweighted=flag) for flag in (False, True)}
    count = 0
    for source, kind in groups:
        data = json.loads(source.read_text())
        records = data["panels"] if isinstance(data, dict) else data
        for record in records:
            image = ROOT / record["image"]
            uw = record["image"].startswith("unweighted/")
            metric = saved_reconstruction_metrics(image.parent, lpips_fallback=lookups[uw].get(str(image.parent.relative_to(ROOT)), float("nan")))
            for key in unweighted.METRICS:
                if key in record:
                    np.testing.assert_allclose(record[key], metric[key], rtol=1e-12, atol=1e-12)
            expected_hash = record.get("sha256", record.get("image_sha256"))
            if expected_hash:
                assert hashlib.sha256(image.read_bytes()).hexdigest() == expected_hash
            if kind == "compact":
                pdf = source.parent / f'compact_{record["scenario"]}_reconstructions.pdf'
            elif kind == "gallery":
                pdf = source.parent / "selected_hallucinations.pdf"
            elif kind == "main":
                pdf = source.parent / "sampling_comparison.pdf"
            else:
                pdf = source.with_suffix(".pdf")
            by_pdf[pdf].append((record, kind, uw))
            count += 1
    for pdf, records in by_pdf.items():
        text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
        actual = collections.Counter(re.findall(r"(LPIPS|[WU]\. Loss|PSNR|SSIM)\s+([-+0-9.e]+)", text))
        expected = collections.Counter()
        for record, kind, uw in records:
            expected[("LPIPS", f'{record["lpips"]:.3f}')] += 1
            expected[("U. Loss" if uw else "W. Loss", f'{record["bp_best_loss"]:.2e}')] += 1
            if kind != "main":
                expected[("PSNR", f'{record["psnr_db"]:.2f}')] += 1
                expected[("SSIM", f'{record["ssim"]:.3f}')] += 1
        assert actual == expected, (pdf, expected-actual, actual-expected)
    print(count, "rendered panels PASS", flush=True)
    return count


def check_manuscript(manuscript):
    mapping = {
        "results/weighted/summary/figures/aggregate_metric_forest_combined.pdf": "figures/weighted/summary/aggregate_metric_forest_combined.pdf",
        "results/weighted/ablation/summary/figures/aggregate_cfg_forest_combined.pdf": "figures/weighted/ablation/aggregate_cfg_forest_combined.pdf",
        "unweighted/results/summary/figures/aggregate_metric_forest_combined.pdf": "figures/unweighted/summary/aggregate_metric_forest_combined.pdf",
        "unweighted/results/ablation/summary/figures/aggregate_cfg_forest_combined.pdf": "figures/unweighted/summary/aggregate_cfg_forest_combined.pdf",
    }
    for filename, reference in (
        ("ktilde_sampling_distribution_row_uc_sb_db_ca.pdf", "self_distributions.pdf"),
        ("ktilde_lambda_absolute_heatmap.pdf", "self_compatibility.pdf"),
        ("ktilde_convergence_five_trial_combined.pdf", "convergence_combined.pdf"),
        ("ktilde_cross_class_sampling_laws_zeta_0p0.pdf", "cross_distributions.pdf"),
        ("ktilde_cross_class_lambda_zeta_0p0.pdf", "cross_compatibility.pdf"),
        ("ktilde_cfg_sampling_distributions_zeta_0p0.pdf", "cfg_distributions.pdf"),
        ("ktilde_cfg_lambda_panels_zeta_0.pdf", "cfg_compatibility.pdf"),
    ):
        mapping[f"results/weighted/ktilde/figures/{filename}"] = f"figures/lambda_ktilde/{reference}"
    for path in (ROOT / "results/paper_figures").rglob("*.pdf"):
        relative = path.relative_to(ROOT / "results/paper_figures")
        mapping[str(path.relative_to(ROOT))] = str(Path("figures") / relative)
    with tempfile.TemporaryDirectory(prefix="gcs-svg-check-") as directory:
        for source, reference in mapping.items():
            trees = []
            for index, pdf in enumerate((ROOT / source, manuscript / reference)):
                svg = Path(directory) / f"{index}.svg"
                subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
                trees.append(ET.parse(svg))
            for attr in ("d", "transform", "fill", "stroke"):
                values = [collections.Counter(e.get(attr) for e in tree.iter() if e.get(attr)) for tree in trees]
                assert values[0] == values[1], (source, attr)
            raster = []
            for tree in trees:
                images = collections.Counter()
                for element in tree.iter():
                    if element.tag.endswith("}image"):
                        href = element.get("{http://www.w3.org/1999/xlink}href", "")
                        if href.startswith("data:image/"):
                            with Image.open(io.BytesIO(base64.b64decode(href.split(",", 1)[1]))) as image:
                                images[(image.size, hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest())] += 1
                raster.append(images)
            assert raster[0] == raster[1], (source, "embedded images")
            print("Figure drawing MATCH", reference, flush=True)
    return dict(figures=len(mapping))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, help="Optional local manuscript for exact figure comparisons")
    parser.add_argument("--calculations-only", action="store_true", help="Check data and intervals before rendering notebooks")
    args = parser.parse_args()
    report = check_calculations()
    if not args.calculations_only:
        report["panels"] = check_panels()
    if args.manuscript and not args.calculations_only:
        report.update(check_manuscript(args.manuscript))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
