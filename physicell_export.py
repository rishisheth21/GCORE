#!/usr/bin/env python3
"""
S.M.A.R.C. GBM Pipeline — Phase 3: PhysiCell Full-Domain Export (v3)
physicell_export.py

Exports the FULL quarter-domain [0,7350]um x [0,7350]um at 100um spacing.
Grid: 74x74 = 5476 voxels (cell centers at 50, 150, ..., 7300 um).
O2 computed directly from Gaussian formula using r=sqrt(x^2+y^2) from origin.

Parameters (unchanged from o2_model.py):
  sigma = 3.0 mm, beta = 92.0, O2_max = 38.0 mmHg, rho_0 = 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.expanduser("~/smarc_gbm")
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#333333",
    "axes.labelcolor":    "#111111",
    "axes.titleweight":   "bold",
    "axes.titlesize":     12,
    "axes.labelsize":     10,
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "grid.color":         "#DDDDDD",
    "grid.linewidth":     0.6,
    "font.family":        "DejaVu Sans",
    "savefig.facecolor":  "white",
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
})

# ─── Model constants ──────────────────────────────────────────────────────────
SIGMA_MM  = 3.0    # mm
BETA      = 92.0   # mmHg per density unit
O2_MAX    = 38.0   # mmHg
RHO_0     = 1.0    # peak tumor density
R_HYP_MM  = 4.30   # mm — hypoxic boundary (O2 = 5 mmHg)
R_HYP_UM  = 4300.0 # um
R_TUMOR_MM= 7.35   # mm — tumor boundary (rho > 0.05)
R_TUMOR_UM= 7350.0 # um

# ─── Full-domain grid parameters ──────────────────────────────────────────────
# Quarter-domain: x=[0, 7350]um, y=[0, 7350]um
# 74 nodes per axis at 100um spacing → cell centers at 50, 150, ..., 7300 um
DOMAIN_UM  = 7350.0   # um — total domain extent
SPACING_UM = 100.0    # um — voxel size
N_NODES    = 74       # nodes per axis (74 * 100 = 7400um covers [0, 7350]um)
# Node centers: 50, 150, ..., 7300 um (half-step inset from each edge)
# Use DOMAIN_UM + SPACING_UM/2 as stop to include 7300 um endpoint
x_nodes_um = np.arange(SPACING_UM / 2, DOMAIN_UM + SPACING_UM / 2 - 1e-9, SPACING_UM)   # 50..7300, 74 pts
y_nodes_um = np.arange(SPACING_UM / 2, DOMAIN_UM + SPACING_UM / 2 - 1e-9, SPACING_UM)

# Recalculate N_NODES from actual array sizes
N_X = len(x_nodes_um)
N_Y = len(y_nodes_um)
print(f"Full-domain grid: {N_X} x {N_Y} = {N_X * N_Y} nodes at {SPACING_UM:.0f} um spacing")
print(f"  x range: {x_nodes_um[0]:.1f} ... {x_nodes_um[-1]:.1f} um")
print(f"  y range: {y_nodes_um[0]:.1f} ... {y_nodes_um[-1]:.1f} um")

# ─── Build 2D grid ────────────────────────────────────────────────────────────
PX_UM, PY_UM = np.meshgrid(x_nodes_um, y_nodes_um, indexing="ij")  # (N_X, N_Y)

# ─── Compute O2 using Gaussian formula ────────────────────────────────────────
# O2 = clip(O2_max - beta * rho_0 * exp(-R^2 / (2*sigma^2)), 0, O2_max)
# R is distance from origin in mm; coordinates are in um -> convert to mm
PX_MM = PX_UM / 1000.0
PY_MM = PY_UM / 1000.0
R2_MM2 = PX_MM**2 + PY_MM**2   # radial distance squared from origin (mm^2)
O2_GRID = np.clip(O2_MAX - BETA * RHO_0 * np.exp(-R2_MM2 / (2.0 * SIGMA_MM**2)),
                  0.0, O2_MAX)

print(f"\nO2 grid stats:")
print(f"  min: {O2_GRID.min():.4f} mmHg")
print(f"  max: {O2_GRID.max():.4f} mmHg")

# ─── Gradient check ───────────────────────────────────────────────────────────
o2_range = O2_GRID.max() - O2_GRID.min()
print(f"\nGradient check: O2 range = {o2_range:.4f} mmHg")
if o2_range < 5.0:
    print("ERROR: O2 field is nearly flat! Aborting.")
    import sys; sys.exit(1)
print("  O2 shows meaningful gradient. Proceeding.")

# ─── Verification at key positions ────────────────────────────────────────────
print("\nO2 verification at key positions:")
# pO2 at (50, 50) um — near origin
idx_x_50  = np.argmin(np.abs(x_nodes_um - 50.0))
idx_y_50  = np.argmin(np.abs(y_nodes_um - 50.0))
o2_near_origin = O2_GRID[idx_x_50, idx_y_50]
print(f"  pO2 at (50, 50) um = {o2_near_origin:.4f} mmHg  (expected: low, near 0)")

# pO2 at (4300, 0) — on x-axis at r_hyp
idx_4300 = np.argmin(np.abs(x_nodes_um - 4300.0))
idx_0    = 0   # closest to y=0 is y=50
o2_at_rhyp_x = O2_GRID[idx_4300, idx_0]
r_at_rhyp_x = np.sqrt((x_nodes_um[idx_4300]/1000.0)**2 + (y_nodes_um[idx_0]/1000.0)**2)
print(f"  pO2 at ({x_nodes_um[idx_4300]:.0f}, {y_nodes_um[idx_0]:.0f}) um = {o2_at_rhyp_x:.4f} mmHg  (r={r_at_rhyp_x:.3f}mm, expected ~5 mmHg)")

# pO2 at (0, 4300) — on y-axis at r_hyp (symmetry check)
idx_0_x  = 0
idx_4300_y = np.argmin(np.abs(y_nodes_um - 4300.0))
o2_at_rhyp_y = O2_GRID[idx_0_x, idx_4300_y]
r_at_rhyp_y = np.sqrt((x_nodes_um[idx_0_x]/1000.0)**2 + (y_nodes_um[idx_4300_y]/1000.0)**2)
print(f"  pO2 at ({x_nodes_um[idx_0_x]:.0f}, {y_nodes_um[idx_4300_y]:.0f}) um = {o2_at_rhyp_y:.4f} mmHg  (r={r_at_rhyp_y:.3f}mm, expected ~5 mmHg)")
print(f"  Symmetry diff: {abs(o2_at_rhyp_x - o2_at_rhyp_y):.6f} mmHg")

# pO2 at (7300, 7300) — outer corner
idx_7300_x = np.argmin(np.abs(x_nodes_um - 7300.0))
idx_7300_y = np.argmin(np.abs(y_nodes_um - 7300.0))
o2_outer = O2_GRID[idx_7300_x, idx_7300_y]
print(f"  pO2 at (7300, 7300) um = {o2_outer:.4f} mmHg  (expected: highest values)")

# pO2 at exact r=4300 um along x-axis (analytical check)
o2_exact_4300 = float(np.clip(O2_MAX - BETA * RHO_0 * np.exp(-(4.3**2) / (2.0 * SIGMA_MM**2)), 0, O2_MAX))
print(f"\nAnalytical O2 at r=4.3mm: {o2_exact_4300:.4f} mmHg  (target: ~5 mmHg)")

# ─── Export physicell_o2.csv ──────────────────────────────────────────────────
print("\nExporting physicell_o2.csv ...")
x_um_flat = PX_UM.ravel()
y_um_flat = PY_UM.ravel()
o2_flat   = O2_GRID.ravel()

csv_path = os.path.join(OUT, "physicell_o2.csv")
# Export 4-column format: x_micron, y_micron, z_micron, value
# z=0 for the central slice; this matches load_o2_from_csv expected format (x,y,z,o2)
z_um_flat = np.zeros_like(x_um_flat)
header = "x_micron,y_micron,z_micron,value"
np.savetxt(csv_path,
           np.column_stack([x_um_flat, y_um_flat, z_um_flat, o2_flat]),
           delimiter=",", header=header, comments="", fmt="%.6f")
print(f"  Saved: {csv_path}  ({len(o2_flat)} rows, 4-column format with z=0)")

# ─── Export metadata JSON ──────────────────────────────────────────────────────
meta = {
    "domain_description": "Full quarter-domain [0,7350]um x [0,7350]um",
    "spacing_um":         SPACING_UM,
    "n_nodes_per_axis":   N_X,
    "total_nodes":        N_X * N_Y,
    "x_range_um":         [float(x_nodes_um[0]), float(x_nodes_um[-1])],
    "y_range_um":         [float(y_nodes_um[0]), float(y_nodes_um[-1])],
    "formula":            "O2 = clip(O2_max - beta*rho_0*exp(-R^2/(2*sigma^2)), 0, O2_max)",
    "parameters": {
        "sigma_mm": SIGMA_MM,
        "beta":     BETA,
        "O2_max":   O2_MAX,
        "rho_0":    RHO_0,
    },
    "verification": {
        "pO2_at_50_50_um":     float(o2_near_origin),
        "pO2_at_4300_50_um":   float(o2_at_rhyp_x),
        "pO2_at_50_4300_um":   float(o2_at_rhyp_y),
        "symmetry_diff_mmhg":  float(abs(o2_at_rhyp_x - o2_at_rhyp_y)),
        "pO2_at_7300_7300_um": float(o2_outer),
    },
    "fields": {
        "o2_mmhg": {"min": float(O2_GRID.min()), "max": float(O2_GRID.max())}
    },
}
meta_path = os.path.join(OUT, "physicell_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Saved: {meta_path}")

# ─── Verification figure ───────────────────────────────────────────────────────
print("\nGenerating verification figure ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"PhysiCell Full-Domain Export — Verification\n"
    f"Grid: {N_X}x{N_Y} nodes at {SPACING_UM:.0f} um spacing",
    fontsize=12, fontweight="bold",
)

THETA = np.linspace(0.0, np.pi / 2.0, 500)

# Panel 1: O2 heatmap of the full quarter domain
ax = axes[0]
extent_um = [x_nodes_um[0] - SPACING_UM/2, x_nodes_um[-1] + SPACING_UM/2,
             y_nodes_um[0] - SPACING_UM/2, y_nodes_um[-1] + SPACING_UM/2]
im = ax.imshow(
    O2_GRID.T, origin="lower",
    extent=extent_um,
    cmap="RdYlGn", vmin=0.0, vmax=38.0,
    aspect="equal", interpolation="bilinear",
)
# Hypoxic boundary arc
ax.plot(R_HYP_UM * np.cos(THETA), R_HYP_UM * np.sin(THETA),
        color="lime", lw=2.0, ls=":", label=f"r_hyp = {R_HYP_MM} mm")
# Tumor boundary arc
ax.plot(R_TUMOR_UM * np.cos(THETA), R_TUMOR_UM * np.sin(THETA),
        color="white", lw=1.5, ls="--", label=f"r_tumor = {R_TUMOR_MM} mm")
fig.colorbar(im, ax=ax, label="O2 (mmHg)")
ax.set_title("Full Quarter-Domain O2 Field")
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.2)

# Panel 2: Radial O2 profile (1D)
ax = axes[1]
r_mm = np.linspace(0, 7.5, 500)
o2_r = np.clip(O2_MAX - BETA * RHO_0 * np.exp(-r_mm**2 / (2.0 * SIGMA_MM**2)),
               0.0, O2_MAX)
ax.plot(r_mm * 1000, o2_r, color="#1A5FA8", lw=2.5, label="O2(r)")
ax.axhline(5.0, color="red", ls="--", lw=1.8, label="5 mmHg (hypoxic threshold)")
ax.axhline(1.0, color="darkred", ls=":", lw=1.5, label="1 mmHg (necrotic)")
ax.axvline(4300.0, color="orange", ls="--", lw=1.5, label="r_hyp = 4300 um")
ax.axvline(7350.0, color="green", ls="--", lw=1.5, label="r_tumor = 7350 um")
ax.scatter([x_nodes_um[idx_4300]], [o2_at_rhyp_x], color="red", s=80, zorder=5,
           label=f"CSV value at ({x_nodes_um[idx_4300]:.0f}, {y_nodes_um[idx_0]:.0f}) um = {o2_at_rhyp_x:.2f} mmHg")
ax.set_xlabel("r (um)")
ax.set_ylabel("pO2 (mmHg)")
ax.set_title("Radial O2 Profile")
ax.legend(fontsize=7, loc="lower right")
ax.grid(True)
ax.set_xlim(0, 8000)

fig.tight_layout()
fig_path = os.path.join(OUT, "physicell_grid_verification.png")
fig.savefig(fig_path)
plt.close()
print(f"  Saved: {fig_path}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 3 Complete — PhysiCell Full-Domain Export Summary")
print("=" * 60)
print(f"  Domain         : [0, {DOMAIN_UM}] um x [0, {DOMAIN_UM}] um (quarter)")
print(f"  Spacing        : {SPACING_UM:.0f} um")
print(f"  Grid           : {N_X} x {N_Y} = {N_X * N_Y} nodes")
print(f"  x range        : {x_nodes_um[0]:.1f} ... {x_nodes_um[-1]:.1f} um")
print(f"  y range        : {y_nodes_um[0]:.1f} ... {y_nodes_um[-1]:.1f} um")
print(f"  O2 range       : {O2_GRID.min():.4f} ... {O2_GRID.max():.4f} mmHg")
print(f"  pO2 at (4300,50): {o2_at_rhyp_x:.4f} mmHg  (target ~5 mmHg, within 1 mmHg: {'PASS' if abs(o2_at_rhyp_x - 5.0) < 1.0 else 'FAIL'})")
print(f"  pO2 at (50,4300): {o2_at_rhyp_y:.4f} mmHg  (radial symmetry: {'PASS' if abs(o2_at_rhyp_x - o2_at_rhyp_y) < 0.01 else 'FAIL'})")
print(f"  pO2 at (7300,7300): {o2_outer:.4f} mmHg")
print(f"\nFiles written to {OUT}/")
print("  physicell_o2.csv")
print("  physicell_metadata.json")
print("  physicell_grid_verification.png")
print("=" * 60)
