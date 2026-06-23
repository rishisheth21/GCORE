#!/usr/bin/env python3
"""
S.M.A.R.C. GBM Pipeline — Phase 1: Oxygen Model
o2_model.py

Models steady-state oxygen distribution in glioblastoma using a Gaussian
tumor density and linear O2 consumption.  Generates Figures 1-4 and saves
data arrays + metrics JSON.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.ndimage import gaussian_filter
import json, os

# ─── Output directory ────────────────────────────────────────────────────────
OUT = os.path.expanduser("~/smarc_gbm/outputs")
os.makedirs(OUT, exist_ok=True)

# ─── Global style (applied once, never overridden per figure) ────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#111111",
    "axes.titleweight":  "bold",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        "#DDDDDD",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "savefig.facecolor": "white",
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# ─── Model parameters (NEVER change without explicit instruction) ─────────────
domain      = 10.0      # mm  — grid spans −domain to +domain
res         = 0.2       # mm/voxel
sigma       = 3.0       # mm
rho_0       = 1.0       # peak tumour density (dimensionless)
beta        = 92.0      # mmHg per density unit
O2_max      = 38.0      # mmHg
O2_crit     = 5.0       # mmHg  — hypoxia threshold
D_O2        = 1.7e-3    # mm²/s  — oxygen diffusion in brain tissue
random_seed = 42
np.random.seed(random_seed)

# ─── 3-D grid ────────────────────────────────────────────────────────────────
coords = np.arange(-domain, domain + res / 2, res)   # 101 points
N      = len(coords)
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")   # (N,N,N)

# ─── Core field equations ─────────────────────────────────────────────────────
R2  = X**2 + Y**2 + Z**2
rho = rho_0 * np.exp(-R2 / (2.0 * sigma**2))                           # Eq 1
O2  = np.clip(O2_max - beta * rho, 0.0, O2_max)                        # Eq 2

# Derived scalars
r_hyp    = sigma * np.sqrt(-2.0 * np.log((O2_max - O2_crit) /
                                          (beta * rho_0)))               # Eq 3
core_O2  = float(max(0.0, O2_max - beta * rho_0))
r_tumor  = sigma * np.sqrt(-2.0 * np.log(0.05))

# 1-D radial flux (Eq 4)
r_1d = np.linspace(0.0, domain, 600)
J_r  = (-D_O2 * beta * rho_0 * (r_1d / sigma**2)
        * np.exp(-r_1d**2 / (2.0 * sigma**2)))

# ─── Metrics ──────────────────────────────────────────────────────────────────
hyp_mask         = O2 < O2_crit
hypoxic_fraction = float(hyp_mask.mean())
voxel_vol        = res**3          # mm³
hypoxic_volume   = float(hyp_mask.sum() * voxel_vol)
total_volume     = float(N**3 * voxel_vol)

# Tumor-region hypoxic fraction (tumor defined as rho > 0.05)
tumor_mask             = rho > 0.05
tumor_voxels           = int(tumor_mask.sum())
tumor_hyp_mask         = hyp_mask & tumor_mask
tumor_hypoxic_fraction = float(tumor_hyp_mask.sum() / tumor_voxels) if tumor_voxels > 0 else 0.0
tumor_volume           = float(tumor_voxels * voxel_vol)

metrics = {
    "hypoxic_fraction":         hypoxic_fraction,
    "tumor_hypoxic_fraction":   tumor_hypoxic_fraction,
    "hypoxic_radius_mm":        float(r_hyp),
    "mean_O2":                  float(O2.mean()),
    "min_O2":                   float(O2.min()),
    "max_O2":                   float(O2.max()),
    "variance_O2":              float(O2.var()),
    "hypoxic_volume_mm3":       hypoxic_volume,
    "tumor_volume_mm3":         tumor_volume,
    "total_volume_mm3":         total_volume,
}

# ─── Central 2-D slices ───────────────────────────────────────────────────────
ci     = N // 2         # index of the z=0 / y=0 / x=0 plane
rho_xy = rho[:, :, ci]  # (N,N) — tumour density at z=0
O2_xy  = O2[:, :, ci]   # (N,N) — O2 at z=0
O2_xz  = O2[:, ci, :]   # (N,N) — O2 at y=0
O2_yz  = O2[ci, :, :]   # (N,N) — O2 at x=0

# 1-D radial profile along +x axis
x_1d      = coords[coords >= 0]
rho_1d    = rho_0 * np.exp(-x_1d**2 / (2.0 * sigma**2))
O2_1d     = np.clip(O2_max - beta * rho_1d, 0.0, O2_max)
rho_1d_n  = rho_1d / rho_0       # normalised 0–1
O2_1d_n   = O2_1d  / O2_max      # normalised 0–1

# Common colourmaps
CMAP_O2  = plt.cm.RdYlGn
CMAP_RHO = plt.cm.hot_r
EXTENT   = [-domain, domain, -domain, domain]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 3-D Sinkhole Surface
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 1…")
fig1 = plt.figure(figsize=(11, 8.5))
fig1.patch.set_facecolor("white")
ax1  = fig1.add_subplot(111, projection="3d")
ax1.set_facecolor("white")

# Downsampled surface (every other grid point → 51×51)
step = 2
xs   = coords[::step]
Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
Zg   = np.clip(O2_max - beta * rho_0 *
               np.exp(-(Xg**2 + Yg**2) / (2.0 * sigma**2)), 0.0, O2_max)

Zg_smooth = gaussian_filter(Zg, sigma=1.5)
surf = ax1.plot_surface(
    Xg, Yg, Zg_smooth,
    cmap=CMAP_O2, vmin=0.0, vmax=O2_max,
    alpha=0.90, linewidth=0, antialiased=True,
)

# Floor contour projection
Z_FLOOR = -4.0
ax1.contourf(Xg, Yg, Zg, zdir="z", offset=Z_FLOOR, levels=25,
             cmap=CMAP_O2, alpha=0.55, vmin=0.0, vmax=O2_max)

# Hypoxic boundary ring (dashed red, elevated to surface height at r_hyp)
theta  = np.linspace(0.0, 2.0 * np.pi, 360)
ring_x = r_hyp * np.cos(theta)
ring_y = r_hyp * np.sin(theta)
ring_z = np.full_like(theta,
         float(np.clip(O2_max - beta * rho_0 *
                       np.exp(-r_hyp**2 / (2.0 * sigma**2)),
                       0.0, O2_max)))
ax1.plot(ring_x, ring_y, ring_z, "r--", lw=2.5,
         label=f"Hypoxic boundary  r = {r_hyp:.2f} mm")

# Vertical drop line to floor
ax1.plot([0, 0], [0, 0], [ring_z[0], Z_FLOOR], "r:", lw=1.2, alpha=0.6)

# Colorbar
cb1 = fig1.colorbar(surf, ax=ax1, pad=0.08, shrink=0.55, aspect=18)
cb1.set_label("O₂ (mmHg)", fontsize=10)

# Stats text box
stats = (
    f"σ  = {sigma} mm\n"
    f"β  = {beta} mmHg/unit\n"
    f"Core O₂ (clipped)  = {core_O2:.1f} mmHg\n"
    f"Tumor radius  = {r_tumor:.2f} mm\n"
    f"r_hyp  = {r_hyp:.2f} mm\n"
    f"Tumor hypoxic fraction  = {tumor_hypoxic_fraction*100:.1f}%"
)
ax1.text2D(0.02, 0.98, stats, transform=ax1.transAxes,
           fontsize=8.5, va="top", linespacing=1.6,
           bbox=dict(boxstyle="round,pad=0.45", fc="white",
                     ec="#333333", alpha=0.92))

ax1.set_xlabel("x (mm)", labelpad=10)
ax1.set_ylabel("y (mm)", labelpad=10)
ax1.set_zlabel("O₂ (mmHg)", labelpad=10)
ax1.set_zlim(Z_FLOOR, O2_max + 1)
ax1.set_title("GBM Oxygen Field — 3D Sinkhole Surface", pad=18)
ax1.legend(loc="upper right", fontsize=8.5)

fname1 = os.path.join(OUT, "figure1_3d_sinkhole.png")
fig1.savefig(fname1)
print(f"Saved: {fname1}")
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 2×2 Analysis Panel
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2…")
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 11))
fig2.suptitle("GBM Oxygen Model — 2D Analysis Panel",
              fontsize=14, fontweight="bold", y=1.00)

# ── Top-left: tumour density ─────────────────────────────────────────────────
ax = axes2[0, 0]
im = ax.imshow(rho_xy.T, origin="lower", extent=EXTENT,
               cmap=CMAP_RHO, vmin=0.0, vmax=rho_0)
cb = fig2.colorbar(im, ax=ax, label="ρ (normalised density)")
circ = plt.Circle((0, 0), r_hyp, color="cyan", fill=False,
                  lw=2.0, ls="--", label=f"r_hyp = {r_hyp:.2f} mm")
ax.add_patch(circ)
ax.set_title("Tumour Density  ρ(x, y)  — z = 0")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect("equal")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True)

# ── Top-right: oxygen field ──────────────────────────────────────────────────
ax = axes2[0, 1]
im = ax.imshow(O2_xy.T, origin="lower", extent=EXTENT,
               cmap=CMAP_O2, vmin=0.0, vmax=O2_max)
cb = fig2.colorbar(im, ax=ax, label="O₂ (mmHg)")
cb.ax.axhline(y=O2_crit, color="red", lw=1.5, ls="--")
cs = ax.contour(coords, coords, O2_xy.T, levels=[O2_crit],
                colors=["red"], linewidths=[2.0], linestyles=["--"])
ax.clabel(cs, fmt="%.0f mmHg", fontsize=8, inline=True)
ax.set_title("Oxygen Field  O₂(x, y)  — z = 0")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect("equal")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.grid(True)

# ── Bottom-left: radial profiles ─────────────────────────────────────────────
ax = axes2[1, 0]
ax.plot(x_1d, rho_1d_n, color="#E05C2A", lw=2.3, label="ρ (normalised)")
ax.plot(x_1d, O2_1d_n,  color="#2B8C3E", lw=2.3, label="O₂ (normalised)")
ax.fill_between(x_1d, 0.0, 1.05,
                where=(x_1d <= r_hyp),
                color="#FFB3BA", alpha=0.40, label="Hypoxic zone")
ax.axvline(r_hyp, color="red", ls="--", lw=2.0,
           label=f"r_hyp = {r_hyp:.2f} mm")
ax.set_xlim(0.0, domain)
ax.set_ylim(0.0, 1.08)
ax.set_xlabel("Distance from centre (mm)")
ax.set_ylabel("Normalised value")
ax.set_title("Radial Profiles — ρ and O₂")
ax.legend(fontsize=8, loc="center right")
ax.grid(True)

# ── Bottom-right: oxygen flux ────────────────────────────────────────────────
ax = axes2[1, 1]
ax.plot(r_1d, J_r, color="#1A5FA8", lw=2.3, label="J(r)  — O₂ flux")
j_max_idx = int(np.argmin(J_r))
ax.axvline(r_1d[j_max_idx], color="darkorange", ls="--", lw=2.0,
           label=f"Max efflux  r = {r_1d[j_max_idx]:.1f} mm")
ax.axvline(r_hyp, color="red", ls="--", lw=2.0,
           label=f"Hypoxic boundary  r = {r_hyp:.2f} mm")
ax.axhline(0.0, color="#555555", lw=0.9, ls="-")
ax.set_xlabel("Distance from centre (mm)")
ax.set_ylabel("J(r)  (mmHg·mm/s)")
ax.set_title("Oxygen Flux  J(r)  vs Radius")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True)

fig2.tight_layout(rect=[0, 0, 1, 0.98])
fname2 = os.path.join(OUT, "figure2_2d_analysis.png")
fig2.savefig(fname2)
print(f"Saved: {fname2}")
plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Statistics Panel
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 3…")
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5.5))
fig3.suptitle("GBM Oxygen Model — Statistics",
              fontsize=14, fontweight="bold")

O2_flat = O2.ravel()

# ── Left: O2 histogram ───────────────────────────────────────────────────────
ax = axes3[0]
n_bins = 80
counts, bin_edges, patches = ax.hist(
    O2_flat, bins=n_bins, edgecolor="white", linewidth=0.3)
# Colour bars: red below O2_crit, green above
for patch, left in zip(patches, bin_edges[:-1]):
    patch.set_facecolor("#C0392B" if left < O2_crit else "#2B8C3E")
ax.axvline(O2_crit, color="red", ls="--", lw=2.2,
           label=f"Hypoxia threshold  {O2_crit} mmHg")
# Hypoxic fraction labels — top-left, two clean lines, no boxes
ax.text(0.03, 0.97,
        f"Domain: {hypoxic_fraction * 100:.1f}%",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold", color="#C0392B")
ax.text(0.03, 0.89,
        f"Tumor: {tumor_hypoxic_fraction * 100:.1f}%",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold", color="#7D3C98")
ax.set_xlabel("O₂ (mmHg)")
ax.set_ylabel("Voxel count")
ax.set_title("O₂ Distribution — Full 3D Field")
ax.legend(fontsize=8)
ax.grid(True)

# ── Middle: bar chart at 5 key radii ─────────────────────────────────────────
ax = axes3[1]
key_r   = [0.0, 1.0, sigma, 2.0 * sigma, domain]
key_lbl = [
    "Core\n(r = 0)",
    "r = 1 mm",
    f"r = σ\n({sigma:.0f} mm)",
    f"r = 2σ\n({2*sigma:.0f} mm)",
    f"Periphery\n({domain:.0f} mm)",
]
key_O2 = [
    float(np.clip(O2_max - beta * rho_0 * np.exp(-r**2 / (2.0 * sigma**2)),
                  0.0, O2_max))
    for r in key_r
]
bar_norm   = mcolors.Normalize(vmin=0.0, vmax=O2_max)
bar_colors = [CMAP_O2(bar_norm(v)) for v in key_O2]
bars = ax.bar(key_lbl, key_O2, color=bar_colors,
              edgecolor="#333333", linewidth=0.9, width=0.6)
for bar, val in zip(bars, key_O2):
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.3,
            f"{val:.1f} mmHg", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.axhline(O2_crit, color="red", ls="--", lw=1.8,
           label=f"Hypoxia threshold  ({O2_crit} mmHg)")
ax.set_ylabel("O₂ (mmHg)")
ax.set_title("O₂ at Key Radial Distances")
ax.set_ylim(0.0, O2_max * 1.15)
ax.legend(fontsize=8)
ax.grid(True, axis="y")

# Add a colorbar for bar chart context
sm_bar = plt.cm.ScalarMappable(cmap=CMAP_O2, norm=bar_norm)
sm_bar.set_array([])
cb3m = fig3.colorbar(sm_bar, ax=ax, pad=0.02, shrink=0.85)
cb3m.set_label("O₂ (mmHg)", fontsize=9)

# ── Right: pie chart — tumor volume only ──────────────────────────────────────
ax = axes3[2]
hyp_pct  = tumor_hypoxic_fraction * 100.0
norm_pct = 100.0 - hyp_pct
wedges, texts, autotexts = ax.pie(
    [hyp_pct, norm_pct],
    labels=["Hypoxic\n(< 5 mmHg)", "Normoxic\n(≥ 5 mmHg)"],
    autopct="%1.1f%%",
    colors=["#C0392B", "#2B8C3E"],
    startangle=90,
    explode=(0.07, 0.0),
    wedgeprops=dict(edgecolor="white", linewidth=2.0),
    textprops=dict(fontsize=10),
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
    at.set_color("white")
ax.set_title("Hypoxic vs Normoxic — Tumor Volume")

fig3.tight_layout(rect=[0, 0, 1, 0.95])
fname3 = os.path.join(OUT, "figure3_statistics.png")
fig3.savefig(fname3)
print(f"Saved: {fname3}")
plt.close(fig3)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Three Orthogonal Cross-Sections
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 4…")
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5.2))
fig4.suptitle("GBM Oxygen Field — Orthogonal Cross-Sections",
              fontsize=14, fontweight="bold")

contour_levels = [5, 10, 20]          # mmHg — labelled on every slice
contour_colors = ["red", "orange", "gold"]
contour_lw     = [2.0, 1.6, 1.3]

slices_info = [
    (O2_xy,  coords, coords, "XY Plane  (z = 0)",  "x (mm)", "y (mm)"),
    (O2_xz,  coords, coords, "XZ Plane  (y = 0)",  "x (mm)", "z (mm)"),
    (O2_yz,  coords, coords, "YZ Plane  (x = 0)",  "y (mm)", "z (mm)"),
]

for ax, (data, cx, cy, title, xlabel, ylabel) in zip(axes4, slices_info):
    im = ax.imshow(data.T, origin="lower",
                   extent=[-domain, domain, -domain, domain],
                   cmap=CMAP_O2, vmin=0.0, vmax=O2_max)
    cs = ax.contour(cx, cy, data.T,
                    levels=contour_levels,
                    colors=contour_colors,
                    linewidths=contour_lw)
    ax.clabel(cs, fmt="%.0f mmHg", fontsize=7.5, inline=True)
    fig4.colorbar(im, ax=ax, label="O₂ (mmHg)", shrink=0.88)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

# Shared legend for contour levels
legend_patches = [
    mpatches.Patch(color=c, label=f"{lv} mmHg")
    for c, lv in zip(contour_colors, contour_levels)
]
fig4.legend(handles=legend_patches, title="O₂ contours",
            loc="lower center", ncol=3, fontsize=9,
            bbox_to_anchor=(0.5, -0.04))

fig4.tight_layout(rect=[0, 0.03, 1, 0.96])
fname4 = os.path.join(OUT, "figure4_orthogonal_slices.png")
fig4.savefig(fname4)
print(f"Saved: {fname4}")
plt.close(fig4)

# ══════════════════════════════════════════════════════════════════════════════
# Save arrays and metrics
# ══════════════════════════════════════════════════════════════════════════════
p = os.path.join(OUT, "oxygen_3D.npy")
np.save(p, O2)
print(f"Saved: {p}")

p = os.path.join(OUT, "tumor_density_3D.npy")
np.save(p, rho)
print(f"Saved: {p}")

p = os.path.join(OUT, "coords_1D.npy")
np.save(p, coords)
print(f"Saved: {p}")

p = os.path.join(OUT, "o2_metrics.json")
with open(p, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: {p}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  O2 MODEL — METRICS SUMMARY")
print("═" * 55)
for k, v in metrics.items():
    print(f"  {k:<30}  {v:.6g}")
print("═" * 55)
print(f"  Hypoxic fraction (full domain):   {hypoxic_fraction * 100:.3f}%")
print(f"  Hypoxic fraction (tumor, ρ>0.05): {tumor_hypoxic_fraction * 100:.3f}%")
print("═" * 55)
print("All done.")
