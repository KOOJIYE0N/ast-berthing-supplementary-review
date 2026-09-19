"""Verify retrieved depth arrays against the supplied experiment records."""

import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def average_ranks(values):
    _, indices, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2)[indices]


def spearman_correlation(x, y):
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "supplementary/depth_no_msaa"
META = ROOT / "metadata"
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
expected = json.loads((META / "depth_validation_reference.json").read_text())
relative = json.loads((META / "depth_relative_metrics.json").read_text())
sizes = json.loads((META / "depth_file_sizes.json").read_text())
report = json.loads((DATA / "report.json").read_text())
calibration = json.loads((DATA / "calibration.json").read_text())
plan = json.loads((DATA / "plan.json").read_text())
assert plan == report["plan"] == calibration["plan"]
assert calibration["coefficients"] == report["coefficients"]
files = sorted(DATA.glob("*.npz"))
assert len(files) == 160 and {p.name for p in files} == set(expected["files"])
rows = {r["file"]: r for r in report["rows"]}
relative_rows = {(r["seed"], r["view"], r["condition"]): r for r in relative["rows"]}
schema = {
    "rgb": ((256, 256, 3), "uint8"),
    "prediction": ((224, 224), "float32"),
    "native_gt_m": ((600, 800), "float32"),
    "native_mask": ((600, 800), "bool"),
    "nearest_gt_m": ((224, 224), "float32"),
    "bilinear_gt_m": ((224, 224), "float32"),
    "target_mask": ((224, 224), "bool"),
    "qpos": ((6,), "float64"),
}
hashes = {}
samples = defaultdict(list)
scene_geometry = {}
target_metrics = []
max_metric_residual = 0.0
checks = 0

for path in files:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected["files"][path.name], path.name
    assert path.stat().st_size == sizes[path.name]
    hashes[path.name] = digest
    row = rows[path.name]
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == set(schema)
        for name, (shape, dtype) in schema.items():
            assert data[name].shape == shape and str(data[name].dtype) == dtype, (path.name, name)
            assert np.isfinite(data[name]).all()
        rgb, prediction, gt, mask = [data[n] for n in ["rgb", "prediction", "nearest_gt_m", "target_mask"]]
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb_sha256"]
        assert int(mask.sum()) == row["target_pixels"]
        assert int(data["native_mask"].sum()) == row["native_target_pixels"]
        nearest = np.asarray(Image.fromarray(data["native_gt_m"]).resize((256, 256), Image.Resampling.NEAREST))[16:240, 16:240]
        target = np.asarray(Image.fromarray(data["native_mask"].astype("uint8")).resize((256, 256), Image.Resampling.NEAREST))[16:240, 16:240].astype(bool)
        assert np.array_equal(nearest, gt) and np.array_equal(target, mask)
        assert rgb[16:240, 16:240].shape[:2] == prediction.shape
        if row["condition"] == "occlusion":
            assert np.all(rgb[64:192, 64:192] == 0)
        geometry = tuple(hashlib.sha256(data[n].tobytes()).hexdigest()
                         for n in ["native_gt_m", "native_mask", "qpos"])
        group = row["split"], row["seed"], row["view"]
        if group in scene_geometry:
            assert geometry == scene_geometry[group]
        scene_geometry[group] = geometry
        valid = np.isfinite(gt) & (gt > 0) & (gt < .999 * expected["far_m"])
        if row["split"] == "calibration":
            grid = np.zeros_like(valid)
            grid[::4, ::4] = True
            samples[row["view"]].append((prediction[valid & grid], gt[valid & grid]))
            continue
        target_valid = valid & mask
        stored = row["metrics"]["target"]
        assert int(target_valid.sum()) == stored["pixels"]
        if not target_valid.any():
            assert row["view"] == "headview"
            target_metrics.append({"seed": row["seed"], "view": row["view"],
                                   "condition": row["condition"], "pixels": 0})
            continue
        p = prediction[target_valid].astype(float)
        g = gt[target_valid].astype(float)
        a, b = calibration["coefficients"][row["view"]]
        inverse = a * p + b
        ok = inverse > 0
        calibrated, truth = 1 / inverse[ok], g[ok]
        delta = np.log(calibrated) - np.log(truth)
        metrics = {
            "inverse_rank_correlation": spearman_correlation(p, 1 / g),
            "positive_metric_coverage": float(ok.mean()),
            "rmse_m": float(np.sqrt(np.mean((calibrated - truth) ** 2))),
            "absrel": float(np.mean(abs(calibrated - truth) / truth)),
            "scale_invariant_log_rmse": float(np.sqrt(max(0, np.mean(delta ** 2) - np.mean(delta) ** 2))),
        }
        for k, value in metrics.items():
            residual = abs(value - stored[k])
            assert np.isclose(value, stored[k], rtol=1e-9, atol=1e-9), (path.name, k)
            max_metric_residual = max(max_metric_residual, residual)
            checks += 1
        positive = p > 0
        raw_delta = np.log(p[positive]) - np.log(1 / g[positive])
        raw_silog = float(np.sqrt(max(0, np.mean(raw_delta ** 2) - np.mean(raw_delta) ** 2)))
        stored_relative = relative_rows[(row["seed"], row["view"], row["condition"])]["metrics"]["target"]
        assert np.isclose(raw_silog, stored_relative["raw_inverse_depth_scale_invariant_log_rmse"], rtol=1e-9, atol=1e-9)
        metrics["raw_inverse_depth_silog"] = raw_silog
        target_metrics.append({"seed": row["seed"], "view": row["view"],
                               "condition": row["condition"], "pixels": len(p), **metrics})

rebuilt = {}
for view, data in samples.items():
    p = np.concatenate([x for x, _ in data]).astype(float)
    g = np.concatenate([y for _, y in data]).astype(float)
    coefficients = np.linalg.lstsq(np.stack([p, np.ones_like(p)], axis=1), 1 / g, rcond=None)[0]
    assert np.allclose(coefficients, calibration["coefficients"][view], rtol=1e-12, atol=1e-12)
    rebuilt[view] = {"calibration_pixels": len(p), "coefficients": coefficients.tolist(),
                     "max_coefficient_difference": float(np.max(abs(coefficients - calibration["coefficients"][view])))}

nominal = {}
for view in ["topview", "frontview", "headview", "sideview"]:
    rs = [r for r in target_metrics if r["view"] == view and r["condition"] == "nominal"]
    assert len(rs) == 12
    valid = [r for r in rs if r["pixels"] > 0]
    nominal[view] = {"test_scenes": len(rs), "scenes_with_target": len(valid)}
    for metric in ["inverse_rank_correlation", "rmse_m", "absrel", "raw_inverse_depth_silog"]:
        nominal[view][metric + "_scene_mean"] = float(np.mean([r[metric] for r in valid])) if valid else None

result = {
    "status": "PASS", "npz_files": len(files),
    "npz_bytes": sum(p.stat().st_size for p in files), "sha256_verified_files": len(hashes),
    "metadata_files_match_supplied_records": True,
    "array_schema_and_native_resize_crop_verified": True,
    "calibration_rebuilt_from_calibration_scenes_only": rebuilt,
    "target_metric_checks": checks, "maximum_target_metric_residual": max_metric_residual,
    "nominal_target_depth": nominal, "files": hashes,
    "limits": ["No policy training, rollout, or new inference was run.",
               "Synthetic RGB occlusion is not a physical 3D occluder.",
               "Material/light parameter changes do not establish strong-reflection robustness.",
               "Head view has no target pixels in the supplied scenes.",
               "The checks do not establish strong-reflection robustness or policy performance."],
}
(OUT / "depth_receipt_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: result[k] for k in ["status", "npz_files", "npz_bytes", "sha256_verified_files",
      "target_metric_checks", "maximum_target_metric_residual", "nominal_target_depth"]}, indent=2))
