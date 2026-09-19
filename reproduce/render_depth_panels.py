"""Reformat unchanged saved diagnostics as readable four-view panels."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "supplementary/depth_no_msaa"
OUT = ROOT / "analysis/figures"
OUT.mkdir(parents=True, exist_ok=True)
VERIFY = ROOT / "metadata/depth_validation_reference.json"
inventory = json.loads(VERIFY.read_text())["files"]
cal = json.loads((DATA / "calibration.json").read_text())["coefficients"]
views = ["topview", "frontview", "headview", "sideview"]
conditions = {"nominal": "figure_depth_diagnostic", "light_material_stress": "figure_depth_lighting",
              "occlusion": "figure_depth_occlusion"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .4})
cmap = plt.get_cmap("magma").copy()
cmap.set_bad("#aaaaaa")
norm = LogNorm(vmin=.25, vmax=20, clip=False)
audit = {"scene": 977101, "selection": "first test seed, unchanged", "panels_per_figure": 12,
         "display_range_m": [.25, 20], "pixel_aspect": 1, "source_files": [], "outputs": {}}

for condition, stem in conditions.items():
    # Four cameras across, three modalities down; axes are physically square.
    fig = plt.figure(figsize=(7.16, 5.38), facecolor="white")
    left, gap, cell_w = .105, .016, .204
    cell_h = cell_w * 7.16 / 5.38
    for col, view in enumerate(views):
        name = f"test_977101_{view}_{condition}.npz"
        path = DATA / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == inventory[name]
        with np.load(path, allow_pickle=False) as d:
            rgb = d["rgb"][16:240, 16:240].copy()
            inverse = cal[view][0] * d["prediction"].astype(float) + cal[view][1]
            valid = np.isfinite(inverse) & (inverse > 0)
            pred = np.full(inverse.shape, np.nan)
            pred[valid] = 1 / inverse[valid]
            gt, mask = d["nearest_gt_m"].copy(), d["target_mask"].copy()
        fig.text(left + col * (cell_w + gap) + cell_w / 2, .974,
                 view.replace("view", "").capitalize(), ha="center", va="center", weight="bold")
        for row, (label, array) in enumerate(zip(["RGB", "Predicted\ndepth", "GT\ndepth"], [rgb, pred, gt])):
            bottom = .945 - cell_h - row * (cell_h + .018)
            if col == 0:
                fig.text(.048, bottom + cell_h / 2, label, ha="center", va="center", fontsize=8)
            ax = fig.add_axes([left + col * (cell_w + gap), bottom, cell_w, cell_h])
            if row == 0:
                ax.imshow(array, interpolation="nearest")
            else:
                ax.imshow(np.ma.masked_invalid(array), cmap=cmap, norm=norm, interpolation="nearest")
                if mask.any():
                    ax.contour(mask.astype(float), levels=[.5], colors=["#00d9ea"], linewidths=.5)
            ax.set(xticks=[], yticks=[], xlim=(-.5, 223.5), ylim=(223.5, -.5))
            for spine in ax.spines.values():
                spine.set_color("#777777")
            assert abs(ax.get_position().width * 7.16 - ax.get_position().height * 5.38) < 1e-8
        audit["source_files"].append({"name": name, "sha256": digest, "target_pixels": int(mask.sum()),
                                      "invalid_pixels": int((~valid).sum())})
    bar = fig.add_axes([.35, .045, .36, .016])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, orientation="horizontal",
                      extend="both", ticks=[.25, .5, 1, 2, 5, 10, 20])
    cb.ax.set_xticklabels(["0.25", "0.5", "1", "2", "5", "10", "20"])
    cb.ax.tick_params(labelsize=8, length=2, pad=2)
    fig.text(.33, .053, "Depth (m, log scale)", ha="right", va="center", fontsize=8)
    fig.text(.74, .053, "Gray: invalid", ha="left", va="center", fontsize=8)
    fig.savefig(OUT / f"{stem}.png", dpi=450)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    audit["outputs"][stem] = hashlib.sha256((OUT / f"{stem}.png").read_bytes()).hexdigest()
(ROOT / "analysis/depth_figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps({"figures": list(conditions.values()), "panels_each": 12, "source_npz_unchanged": True}))
