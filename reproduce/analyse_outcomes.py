"""Recompute controlled outcomes, paired statistics, and recorded-loss complements."""
from pathlib import Path
from collections import defaultdict
from math import comb, sqrt
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def success(r):
    assert r["success"] in {"True", "False", True, False}
    return r["success"] is True or r["success"] == "True"


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c)+1)) / 2**n) if n else 1.0


def holm(values):
    result, running = [None]*len(values), 0.0
    for rank, i in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running = max(running, min(1.0, (len(values)-rank)*values[i]))
        result[i] = running
    return result


def wilson(k, n):
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    half = z*sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [100*max(0.0, center-half), 100*min(1.0, center+half)]


def exact_binomial_interval(k, n, alpha=.05):
    def tail(p, start, stop):
        return sum(comb(n, j)*p**j*(1-p)**(n-j) for j in range(start, stop))
    lower, upper = 0.0, 1.0
    if k:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo+hi)/2
            if tail(mid, k, n+1) < alpha/2:
                lo = mid
            else:
                hi = mid
        lower = (lo+hi)/2
    if k < n:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo+hi)/2
            if tail(mid, 0, k+1) > alpha/2:
                lo = mid
            else:
                hi = mid
        upper = (lo+hi)/2
    return lower, upper


rows = csv_rows(ROOT / "supplementary/phase1_outcomes_474.csv")
primary = [r for r in rows if r["stage"] in ("primary","primary_ext")]
reference_ext = [r for r in rows if r["stage"] in ("primary","primary_ext")]
controls = [r for r in rows if r["stage"] == "additional"]
assert (len([r for r in rows if r["stage"] == "primary"]), len(controls)) == (240, 186)
assert len([r for r in rows if r["stage"] == "primary_ext"]) == 48
pairs = defaultdict(dict)
for r in rows:
    if r["stage"] != "primary":
        continue
    key = (r["condition"], int(r["training_seed"]), int(r["env_seed"]), int(r["policy_seed"]))
    assert r["kind"] not in pairs[key]
    pairs[key][r["kind"]] = r
assert len(pairs) == 120 and all(set(p) == {"rgb", "depth"} for p in pairs.values())
assert all(p["rgb"]["host"] == p["depth"]["host"] for p in pairs.values())
assert len({(k[1], k[2], k[3]) for k in pairs}) == 24
ext_pairs = defaultdict(dict)
for r in rows:
    if r["stage"] == "primary_ext":
        key = (r["condition"], int(r["env_seed"]))
        ext_pairs[key][r["kind"]] = r
assert len(ext_pairs) == 24 and all(set(p) == {"rgb", "depth"} for p in ext_pairs.values())

reference = json.loads((ROOT / "metadata/outcome_reference.json").read_text())
step_reference = json.loads((ROOT / "metadata/step_statistics_reference.json").read_text())
result = {"primary": [], "seed_groups": {}, "controls": [], "phase2": [], "recorded_loss": []}
for i, condition in enumerate(["none", "top", "front", "head", "side"]):
    selected = [(k, p) for k, p in sorted(pairs.items()) if k[0] == condition]
    b = sum(success(p["depth"]) and not success(p["rgb"]) for _, p in selected)
    c = sum(success(p["rgb"]) and not success(p["depth"]) for _, p in selected)
    k_rgb = sum(success(p["rgb"]) for _, p in selected)
    k_depth = sum(success(p["depth"]) for _, p in selected)
    ref = reference["primary"][i]
    assert (k_rgb, k_depth, b, c) == (ref["rgb"], ref["depth"], ref["depth_only_success"], ref["rgb_only_success"])
    assert mcnemar(b, c) == ref["mcnemar_p"]
    deltas = np.array([int(p["depth"]["policy_steps"])-int(p["rgb"]["policy_steps"]) for _, p in selected])
    seeds = np.array([k[1] for k, _ in selected])
    rng = np.random.default_rng(20260914+i)
    samples = [rng.choice(deltas[seeds == s], size=(2000, 8), replace=True) for s in (0, 1, 2)]
    step_ci = np.quantile(np.concatenate(samples, axis=1).mean(axis=1), [.025, .975])
    assert np.allclose(step_ci, step_reference["primary"][i]["step_difference95"], atol=1e-10)
    lo, hi = exact_binomial_interval(b, b+c)
    odds_ci = [lo/(1-lo), "infinity" if hi == 1 else hi/(1-hi)]
    result["primary"].append({"condition": condition, "pairs": len(selected), "rgb": k_rgb,
        "depth": k_depth, "discordances": [b, c], "exact_p": mcnemar(b, c),
        "rgb_wilson95_pct": wilson(k_rgb, 24), "depth_wilson95_pct": wilson(k_depth, 24),
        "matched_or": b/c if c else "infinity", "pointwise_or95": odds_ci,
        "mean_recorded_step_difference": float(deltas.mean()), "pointwise_step_difference95": step_ci.tolist()})
for row, p, ref in zip(result["primary"], holm([r["exact_p"] for r in result["primary"]]), reference["primary"], strict=True):
    row["holm_p"] = p
    assert p == ref["holm_p"]

for seed in [0, 1, 2]:
    pool = [r for r in rows if r["stage"] == "primary"]
    result["seed_groups"][str(seed)] = {kind: {"success": sum(success(r) for r in pool if r["training_seed"] == str(seed) and r["kind"] == kind),
        "n": sum(r["training_seed"] == str(seed) and r["kind"] == kind for r in pool)} for kind in ["rgb", "depth"]}
assert result["seed_groups"] == reference["per_training_seed"]

for ref in reference["ablation"]:
    selected = [r for r in controls if r["kind"] == ref["kind"] and r["condition"] == ref["condition"]]
    keys = {(r["env_seed"], r["policy_seed"]) for r in selected}
    refs = [r for r in reference_ext if r["kind"] == "depth" and r["training_seed"] == "0" and r["condition"] == ref["condition"] and (r["env_seed"], r["policy_seed"]) in keys]
    assert len(selected) == len(refs) == ref["n"]
    assert sum(success(r) for r in selected) == ref["success"]
    assert sum(success(r) for r in refs) == ref["matched_depth_success"]
    result["controls"].append({"kind": ref["kind"], "condition": ref["condition"], "success": ref["success"], "n": ref["n"]})

p1_full = json.loads((ROOT / "supplementary/physical/phase1_trial_records_full.json").read_text())
assert len(p1_full) == 240
for kind, expected in [("rgb", (98, 41)), ("depth", (112, 29))]:
    selected = [r for r in p1_full if r["kind"] == kind and success(r)]
    assert all(r["followup_complete"] for r in selected)
    loss = sum(r["post_contact_physics_samples_with_contact"] < r["post_contact_physics_samples"] for r in selected)
    assert (len(selected), loss) == expected
    result["recorded_loss"].append({"phase": 1, "kind": kind, "successes": len(selected), "any_recorded_loss": loss, "no_recorded_loss": len(selected)-loss})

p2 = csv_rows(ROOT / "supplementary/physical/phase2_rgb_depth_trials_64.csv")
safety = {r["tag"]: r for r in json.loads((ROOT / "metadata/phase2_safety.json").read_text())["rows"]}
p2_pairs = defaultdict(dict)
for r in p2:
    if r["group"] == "corrected":
        key = (r["env_seed"], r["policy_seed"])
        assert r["kind"] not in p2_pairs[key]
        p2_pairs[key][r["kind"]] = r
assert len(p2_pairs) == 30 and all(set(p) == {"rgb", "depth"} for p in p2_pairs.values())
assert all(p["rgb"]["execution_host"] == p["depth"]["execution_host"] for p in p2_pairs.values())
b = sum(success(p["depth"]) and not success(p["rgb"]) for p in p2_pairs.values())
c = sum(success(p["rgb"]) and not success(p["depth"]) for p in p2_pairs.values())
assert (b, c) == (17, 0) and mcnemar(b, c) == reference["phase2_pooled_p"]
initial = {k: v for k, v in p2_pairs.items() if int(k[0]) < 970010}
bi = sum(success(p["depth"]) and not success(p["rgb"]) for p in initial.values())
ci = sum(success(p["rgb"]) and not success(p["depth"]) for p in initial.values())
assert (bi, ci) == (6, 0) and mcnemar(bi, ci) == .03125
result["phase2_initial10"] = {"b": bi, "c": ci, "p": mcnemar(bi, ci)}
for group in ["corrected", "legacy"]:
    for kind in ["rgb", "depth"]:
        selected = [r for r in p2 if r["group"] == group and r["kind"] == kind]
        ok = [r for r in selected if success(r)]
        assert {"success": len(ok), "n": len(selected)} == reference["phase2"][group+"_"+kind]
        result["phase2"].append({"group": group, "kind": kind, "success": len(ok), "n": len(selected), "wilson95_pct": wilson(len(ok), len(selected))})
        if group == "corrected":
            losses = sum(safety[r["tag"]]["post_threshold_fraction_within_0_10m"] < 1 for r in ok)
            assert losses == (6 if kind == "depth" else 0)
            result["recorded_loss"].append({"phase": 2, "kind": kind, "successes": len(ok),
                "any_recorded_loss": losses if ok else None, "no_recorded_loss": len(ok)-losses if ok else None})
result["phase2_exploratory_p"] = mcnemar(b, c)
assert result["phase2_exploratory_p"] == reference["phase2_pooled_p"]
result["status"] = "PASS"
result["scope"] = "Stored outcomes and summaries only; no training, rollout, or missing-trace reconstruction."
(OUT / "outcomes.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps({"status": "PASS", "primary": len(primary), "controls": len(controls), "phase2": len(p2), "seed_groups": result["seed_groups"], "recorded_loss": result["recorded_loss"]}, indent=2))
