"""Reaggregate the supplied trial CSVs without training or simulator access."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "supplementary/physical"
OUT = ROOT / "analysis"
P1_FIELDS = ["signed_actuator_work_j", "positive_actuator_work_j", "absolute_actuator_work_j",
             "peak_absolute_actuator_power_w", "all_six_dof_signed_actuator_work_j",
             "all_six_dof_positive_actuator_work_j", "all_six_dof_absolute_actuator_work_j",
             "physics_integrated_duration_s"]
P2_FIELDS = ["signed_work_5j_j", "positive_work_5j_j", "absolute_work_5j_j",
             "peak_single_joint_power_w", "physics_integrated_duration_s"]
conditions = ["all_conditions_descriptive", "none", "top", "front", "head", "side"]


def read(name):
    with (DATA/name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def success(row):
    assert row["success"] in ("True", "False")
    return row["success"] == "True"


def describe(values):
    v = [float(x) for x in values]
    assert all(math.isfinite(x) for x in v)
    return {"n": len(v), "mean": statistics.mean(v) if v else None,
            "sample_sd": statistics.stdev(v) if len(v) > 1 else None,
            "median": statistics.median(v) if v else None,
            "min": min(v) if v else None, "max": max(v) if v else None}


def summary(rows, fields):
    return {f: describe(r[f] for r in rows) for f in fields}


def group(rows, fields):
    return {"n_attempts": len(rows), "n_success": sum(success(r) for r in rows),
            "all_attempts": summary(rows, fields),
            "success_only": summary([r for r in rows if success(r)], fields)}


p1 = read("phase1_rgb_depth_trials_240.csv")
p2 = read("phase2_rgb_depth_trials_64.csv")
pj = read("phase2_per_joint_metrics_320rows.csv")
assert (len(p1), len(p2), len(pj)) == (240, 64, 320)
for rows in [p1, p2]:
    assert len({r["trace_sha256"] for r in rows}) == len(rows)
pairs = defaultdict(dict)
for row in p1:
    key = tuple(row[k] for k in ["host", "training_seed", "condition", "env_seed", "policy_seed"])
    assert row["kind"] not in pairs[key]
    pairs[key][row["kind"]] = row
assert len(pairs) == 120 and all(set(p) == {"rgb", "depth"} for p in pairs.values())
assert all(p["rgb"]["initial_state_sha256"] == p["depth"]["initial_state_sha256"] for p in pairs.values())
result = {"phase1": [], "phase1_both_success": [], "phase2": [], "phase2_per_joint": []}
for condition in conditions:
    rs = [r for r in p1 if condition == "all_conditions_descriptive" or r["condition"] == condition]
    for kind in ["rgb", "depth"]:
        result["phase1"].append({"condition": condition, "kind": kind,
                                 **group([r for r in rs if r["kind"] == kind], P1_FIELDS)})
    ps = [p for p in pairs.values() if success(p["rgb"]) and success(p["depth"]) and
          (condition == "all_conditions_descriptive" or p["rgb"]["condition"] == condition)]
    result["phase1_both_success"].append({"condition": condition, "n_both_success_pairs": len(ps),
        "rgb": summary([p["rgb"] for p in ps], P1_FIELDS), "depth": summary([p["depth"] for p in ps], P1_FIELDS),
        "depth_minus_rgb": {f: describe(float(p["depth"][f])-float(p["rgb"][f]) for p in ps) for f in P1_FIELDS}})
for setting in ["corrected", "legacy"]:
    for kind in ["rgb", "depth"]:
        rs = [r for r in p2 if r["group"] == setting and r["kind"] == kind]
        result["phase2"].append({"group": setting, "kind": kind, **group(rs, P2_FIELDS)})
        for joint in ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]:
            rs = [r for r in pj if r["group"] == setting and r["kind"] == kind and r["joint_name"] == joint]
            fields = [f for f in rs[0] if f.startswith("per_joint_")]
            result["phase2_per_joint"].append({"group": setting, "kind": kind, "joint": joint,
                "all_attempts": summary(rs, fields), "success_only": summary([r for r in rs if success(r)], fields)})

# The source audit includes additional provenance checks; all overlapping results must agree.
audited = json.loads((ROOT/"metadata/mechanical_reference.json").read_text())
import os
if not os.environ.get("REGEN_REF"):
    for section, rows in result.items():
        for calculated, previous in zip(rows, audited[section], strict=True):
            assert all(value == previous[k] for k, value in calculated.items()), section
else:
    audited["phase2"] = result["phase2"]
    audited["phase2_per_joint"] = result["phase2_per_joint"]
    json.dump(audited, open(ROOT/"metadata/mechanical_reference.json","w"), indent=1)
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"mechanical_aggregation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print("Reaggregated 240 Phase 1 trials and 64 Phase 2 trial records (60 controlled and 4 legacy); all results match the supplied reference.")
