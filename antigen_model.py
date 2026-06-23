#!/usr/bin/env python3
"""
S.M.A.R.C. GBM Pipeline — Phase 2: Antigen Model (v2)
antigen_model.py

Literature-based molecular response model with distinct parameters for
HIF-1α, GD2, and CAIX across the GBM oxygen landscape.
  HIF-1α: sharp switch at 5 mmHg  (k=2.5)
  GD2   : moderate switch at 7 mmHg (k=1.5, forward sigmoid)
  CAIX  : very broad gradient at 20 mmHg (k=0.3)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import zoom as ndimage_zoom
import json, os

# ─── Output directory ─────────────────────────────────────────────────────────
OUT = os.path.expanduser("~/smarc_gbm/outputs")
os.makedirs(OUT, exist_ok=True)

# ─── Global style ─────────────────────────────────────────────────────────────
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

# ─── Load Phase 1 data ────────────────────────────────────────────────────────
O2     = np.load(os.path.join(OUT, "oxygen_3D.npy"))
rho    = np.load(os.path.join(OUT, "tumor_density_3D.npy"))
coords = np.load(os.path.join(OUT, "coords_1D.npy"))
N      = len(coords)
domain = float(coords[-1])   # 10.0 mm

# ─── Parameters (literature-based, each molecule distinct) ────────────────────
O2_half_hif  = 5.0;  k_hif  = 2.5   # sharp switch at hypoxic threshold
O2_half_gd2  = 7.0;  k_gd2  = 1.5   # moderate, slightly shifted
O2_half_caix = 20.0; k_caix = 0.3   # very broad, ultra-gentle gradient

sigma   = 3.0
rho_0   = 1.0
O2_max  = 38.0
beta    = 92.0
r_tumor = sigma * np.sqrt(-2.0 * np.log(0.05))   # ~7.34 mm effective boundary

EXTENT = [-domain, domain, -domain, domain]

# Colors
C_HIF  = "#E84545"
C_GD2  = "#9B1FBE"
C_CAIX = "#1A6FBF"
C_O2   = "#2E8B57"

CMAP_HIF  = plt.cm.YlOrRd
CMAP_GD2  = plt.cm.RdPu
CMAP_CAIX = plt.cm.PuBu
CMAP_O2   = plt.cm.RdYlGn

# ─── 3-D field equations ──────────────────────────────────────────────────────
HIF  = 1.0 / (1.0 + np.exp(+k_hif  * (O2 - O2_half_hif)))
GD2  = 1.0 / (1.0 + np.exp(-k_gd2  * (O2 - O2_half_gd2)))
CAIX = 1.0 / (1.0 + np.exp(+k_caix * (O2 - O2_half_caix)))

# ─── Central slice (z = 0) ────────────────────────────────────────────────────
ci      = N // 2
HIF_xy  = HIF[:, :, ci]
GD2_xy  = GD2[:, :, ci]
CAIX_xy = CAIX[:, :, ci]
O2_xy   = O2[:, :, ci]
rho_xy  = rho[:, :, ci]

KILL = GD2_xy.copy()
GAP  = np.clip(CAIX_xy - GD2_xy, 0.0, None)

# ─── 3-D radius ───────────────────────────────────────────────────────────────
X3, Y3, Z3 = np.meshgrid(coords, coords, coords, indexing="ij")
R3 = np.sqrt(X3**2 + Y3**2 + Z3**2)

# ─── Tumor mask ───────────────────────────────────────────────────────────────
tumor_mask = rho > 0.05

# ─── 1-D radial profiles (analytical, along +x axis) ─────────────────────────
x_1d    = coords[coords >= 0]
O2_1d   = np.clip(O2_max - beta * rho_0 * np.exp(-x_1d**2 / (2.0 * sigma**2)),
                  0.0, O2_max)
HIF_1d  = 1.0 / (1.0 + np.exp(+k_hif  * (O2_1d - O2_half_hif)))
GD2_1d  = 1.0 / (1.0 + np.exp(-k_gd2  * (O2_1d - O2_half_gd2)))
CAIX_1d = 1.0 / (1.0 + np.exp(+k_caix * (O2_1d - O2_half_caix)))

# Transition radii (where each molecule = 0.5)
r_hif_trans  = float(x_1d[np.argmin(np.abs(HIF_1d  - 0.5))])
r_gd2_trans  = float(x_1d[np.argmin(np.abs(GD2_1d  - 0.5))])
r_caix_trans = float(x_1d[np.argmin(np.abs(CAIX_1d - 0.5))])

# ─── Metrics ──────────────────────────────────────────────────────────────────
hif_core  = float(HIF[ci, ci, ci])
gd2_core  = float(GD2[ci, ci, ci])
caix_core = float(CAIX[ci, ci, ci])

GD2_act  = GD2  > 0.5
CAIX_act = CAIX > 0.5

gd2_blind_fraction       = float((GD2[tumor_mask]  < 0.5).mean())
caix_targetable_fraction = float((CAIX[tumor_mask] > 0.5).mean())

t = int(tumor_mask.sum())
gd2_only_fraction  = float(( GD2_act & ~CAIX_act & tumor_mask).sum() / t)
caix_only_fraction = float((~GD2_act &  CAIX_act & tumor_mask).sum() / t)
dual_fraction      = float(( GD2_act &  CAIX_act & tumor_mask).sum() / t)
neither_fraction   = float((~GD2_act & ~CAIX_act & tumor_mask).sum() / t)

# CAIX rescue fraction of the GD2 blind zone
blind_3d        = tumor_mask & (GD2 < 0.5)
rescue_3d       = blind_3d & (CAIX > 0.5)
caix_rescue_pct = 100.0 * float(rescue_3d.sum()) / max(float(blind_3d.sum()), 1.0)

antigen_metrics = {
    "o2_half_hif":  O2_half_hif,  "k_hif":  k_hif,
    "o2_half_gd2":  O2_half_gd2,  "k_gd2":  k_gd2,
    "o2_half_caix": O2_half_caix, "k_caix": k_caix,
    "hif_core":  hif_core,
    "gd2_core":  gd2_core,
    "caix_core": caix_core,
    "gd2_blind_fraction":        gd2_blind_fraction,
    "caix_targetable_fraction":  caix_targetable_fraction,
    "gd2_only_fraction":         gd2_only_fraction,
    "caix_only_fraction":        caix_only_fraction,
    "dual_coverage_fraction":    dual_fraction,
    "neither_fraction":          neither_fraction,
    "gd2_transition_radius_mm":  r_gd2_trans,
    "caix_transition_radius_mm": r_caix_trans,
}

# ─── Helper: save figure ──────────────────────────────────────────────────────
def save_fig(fig, name, transparent=False):
    path = os.path.join(OUT, name)
    fig.savefig(path, transparent=transparent)
    print(f"Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Transfer functions
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 1 — transfer functions…")
fig1, ax1 = plt.subplots(figsize=(11, 6.5))

O2_line = np.linspace(0.0, O2_max, 500)
HIF_line  = 1.0 / (1.0 + np.exp(+k_hif  * (O2_line - O2_half_hif)))
GD2_line  = 1.0 / (1.0 + np.exp(-k_gd2  * (O2_line - O2_half_gd2)))
CAIX_line = 1.0 / (1.0 + np.exp(+k_caix * (O2_line - O2_half_caix)))

# Background shading
ax1.axvspan(0.0,  5.0, alpha=0.10, color="red",    label="Severe hypoxia (<5 mmHg)")
ax1.axvspan(5.0, 10.0, alpha=0.08, color="orange",  label="Moderate hypoxia (5–10 mmHg)")

# Molecule curves
ax1.plot(O2_line, HIF_line,  color=C_HIF,  lw=2.5,
         label=f"HIF-1α  (O₂_half={O2_half_hif}, k={k_hif})")
ax1.plot(O2_line, GD2_line,  color=C_GD2,  lw=2.5,
         label=f"GD2  (O₂_half={O2_half_gd2}, k={k_gd2})")
ax1.plot(O2_line, CAIX_line, color=C_CAIX, lw=2.5,
         label=f"CAIX  (O₂_half={O2_half_caix}, k={k_caix})")

# O2_half verticals
for xv, color in [(O2_half_hif, C_HIF), (O2_half_gd2, C_GD2),
                  (O2_half_caix, C_CAIX)]:
    ax1.axvline(xv, color=color, lw=0.8, alpha=0.5, ls="--")

ax1.axhline(0.5, color="#888888", lw=1.2, ls="--", label="50% expression")

ax1.set_xlim(0.0, O2_max)
ax1.set_ylim(-0.03, 1.08)
ax1.set_xlabel("O₂ (mmHg)")
ax1.set_ylabel("Normalised expression")
ax1.set_title("Molecular Response to Oxygen — Transfer Functions")
ax1.legend(loc="center right", fontsize=9.5, framealpha=0.95)
ax1.grid(True)

fig1.tight_layout()
save_fig(fig1, "antigen_fig1_transfer_functions.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 2×2 contour maps
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2 — 2×2 contour maps…")
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))
fig2.suptitle("GBM Molecular Landscape — Central Slice (z=0)",
              fontsize=14, fontweight="bold")

LEVELS    = 50
CROP      = 8.0
THETA_RNG = np.linspace(0.0, 2.0 * np.pi, 300)
ANN_BOX   = dict(boxstyle="round", fc="black", alpha=0.5)

# ── Top-left: O2 ──────────────────────────────────────────────────────────────
ax = axes2[0, 0]
cf = ax.contourf(coords, coords, O2_xy.T, levels=LEVELS,
                 cmap=CMAP_O2, vmin=0.0, vmax=O2_max)
cs = ax.contour(coords, coords, O2_xy.T, levels=[2.0, 5.0, 10.0, 20.0],
                colors="white", linewidths=1.4, linestyles="--")
ax.clabel(cs, fmt="%.0f mmHg", fontsize=8, inline=True)
cb = fig2.colorbar(cf, ax=ax, label="O₂ (mmHg)", shrink=0.88)
cb.set_ticks([0, 5, 10, 20, 38])
ax.set_xlim(-CROP, CROP); ax.set_ylim(-CROP, CROP)
ax.set_title("Oxygen Field O₂(x,y)")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
ax.text(0.03, 0.03, f"r_hyp = {r_hif_trans:.1f} mm  (O\u2082 = 5 mmHg)",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8, color="white", bbox=ANN_BOX)

# ── Top-right: HIF-1α ────────────────────────────────────────────────────────
ax = axes2[0, 1]
cf = ax.contourf(coords, coords, HIF_xy.T, levels=LEVELS,
                 cmap=CMAP_HIF, vmin=0.0, vmax=1.0)
cs_hif_thin = ax.contour(coords, coords, HIF_xy.T, levels=[0.25, 0.75],
                          colors="white", linewidths=1.3, linestyles="--")
ax.clabel(cs_hif_thin, fmt="%.2f", fontsize=8, inline=True)
cs_hif_thick = ax.contour(coords, coords, HIF_xy.T, levels=[0.5],
                           colors="white", linewidths=2.5, linestyles="--")
ax.clabel(cs_hif_thick, fmt="%.2f", fontsize=8, inline=True)
cb = fig2.colorbar(cf, ax=ax, label="HIF-1\u03b1 (0\u20131)", shrink=0.88)
cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(-CROP, CROP); ax.set_ylim(-CROP, CROP)
ax.set_title("HIF-1\u03b1 Activation")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
ax.text(0.03, 0.03, f"50% boundary: r = {r_hif_trans:.1f} mm",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8, color="white", bbox=ANN_BOX)

# ── Bottom-left: GD2 ─────────────────────────────────────────────────────────
ax_gd2 = axes2[1, 0]
cf = ax_gd2.contourf(coords, coords, GD2_xy.T, levels=LEVELS,
                     cmap=plt.cm.PuRd, vmin=0.0, vmax=1.0)
cs_gd2_thin = ax_gd2.contour(coords, coords, GD2_xy.T, levels=[0.25, 0.75],
                               colors="white", linewidths=1.3, linestyles="--")
ax_gd2.clabel(cs_gd2_thin, fmt="%.2f", fontsize=8, inline=True)
cs_gd2_thick = ax_gd2.contour(coords, coords, GD2_xy.T, levels=[0.5],
                                colors="white", linewidths=2.5, linestyles="--")
ax_gd2.clabel(cs_gd2_thick, fmt="%.2f", fontsize=8, inline=True)
cb = fig2.colorbar(cf, ax=ax_gd2, label="GD2 (0\u20131)", shrink=0.88)
cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
# Black border circle at blind boundary
ax_gd2.plot(4.3 * np.cos(THETA_RNG), 4.3 * np.sin(THETA_RNG),
            "k-", lw=1.5, alpha=0.7)
ax_gd2.set_xlim(-CROP, CROP); ax_gd2.set_ylim(-CROP, CROP)
ax_gd2.set_title("GD2 Antigen \u2014 CAR-T Target")
ax_gd2.set_xlabel("x (mm)"); ax_gd2.set_ylabel("y (mm)")
ax_gd2.set_aspect("equal"); ax_gd2.grid(True, alpha=0.25)
ax_gd2.text(0.03, 0.03, f"Blind boundary: r = {r_gd2_trans:.1f} mm",
            transform=ax_gd2.transAxes, ha="left", va="bottom",
            fontsize=8, color="white", bbox=ANN_BOX)

# Arrow annotation pointing to blind boundary ring (replaces broken zoom inset)
_tip_x = 4.4 * np.cos(np.pi / 4)   # ≈ 3.11 mm
_tip_y = 4.4 * np.sin(np.pi / 4)   # ≈ 3.11 mm
ax_gd2.annotate(
    "CAR-T blind zone\n(r < 4.4 mm)",
    xy=(_tip_x, _tip_y),
    xytext=(6.0, 6.0),
    fontsize=9, color="white", ha="center",
    arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
)

# Dashed rectangle on main GD2 panel marking the zoom region (upper-left quadrant)
from matplotlib.patches import Rectangle as _Rect
ax_gd2.add_patch(_Rect((-5.5, 1.0), 4.0, 4.0, linewidth=1.0,
                        edgecolor='white', facecolor='none',
                        linestyle='dashed', alpha=0.8))

# ── Bottom-right: CAIX ────────────────────────────────────────────────────────
ax = axes2[1, 1]
cf = ax.contourf(coords, coords, CAIX_xy.T, levels=LEVELS,
                 cmap=CMAP_CAIX, vmin=0.0, vmax=1.0)
cs_caix_thin = ax.contour(coords, coords, CAIX_xy.T, levels=[0.25, 0.75],
                            colors="white", linewidths=1.3, linestyles="--")
ax.clabel(cs_caix_thin, fmt="%.2f", fontsize=8, inline=True)
cs_caix_thick = ax.contour(coords, coords, CAIX_xy.T, levels=[0.5],
                             colors="white", linewidths=2.5, linestyles="--")
ax.clabel(cs_caix_thick, fmt="%.2f", fontsize=8, inline=True)
cb = fig2.colorbar(cf, ax=ax, label="CAIX (0\u20131)", shrink=0.88)
cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
# Black border circle at CAIX active boundary
ax.plot(5.5 * np.cos(THETA_RNG), 5.5 * np.sin(THETA_RNG),
        "k-", lw=1.5, alpha=0.7)
ax.set_xlim(-CROP, CROP); ax.set_ylim(-CROP, CROP)
ax.set_title("CAIX \u2014 Hypoxia Therapeutic Target")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
ax.text(0.03, 0.03, f"Active boundary: r = ~{r_caix_trans:.1f} mm",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8, color="white", bbox=ANN_BOX)

fig2.tight_layout(pad=3.0)

# ── INSET on GD2 panel — transition zone zoom ─────────────────────────────────
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

X_slice   = coords
Y_slice   = coords
gd2_slice = GD2_xy.T

ax_ins = inset_axes(ax_gd2,
                    width="35%", height="35%",
                    loc="upper right",
                    borderpad=0.8)

ax_ins.contourf(X_slice, Y_slice, gd2_slice,
                levels=50, cmap="PuRd")

ax_ins.set_xlim(2.0, 6.5)
ax_ins.set_ylim(2.0, 6.5)

theta = np.linspace(0, 2 * np.pi, 300)
ax_ins.plot(4.4 * np.cos(theta), 4.4 * np.sin(theta),
            "white", lw=1.8, linestyle="dashed", alpha=0.9)

ax_ins.contour(X_slice, Y_slice, gd2_slice,
               levels=[0.5], colors="white", linewidths=1.2)

ax_ins.set_title("Transition zone", fontsize=7,
                 fontweight="bold", pad=2, color="#111111")
ax_ins.set_xticks([3, 4.4, 6])
ax_ins.set_yticks([3, 4.4, 6])
ax_ins.tick_params(labelsize=5, colors="#333333")
for spine in ax_ins.spines.values():
    spine.set_edgecolor("#333333")
    spine.set_linewidth(1.2)

from matplotlib.patches import Rectangle
rect = Rectangle((2.0, 2.0), 4.5, 4.5,
                 linewidth=1.2, edgecolor="white",
                 facecolor="none", linestyle="dashed", alpha=0.7)
ax_gd2.add_patch(rect)

mark_inset(ax_gd2, ax_ins, loc1=2, loc2=4,
           fc="none", ec="white", lw=0.8, alpha=0.6)
# ── END INSET ─────────────────────────────────────────────────────────────────

save_fig(fig2, "antigen_fig2_contour_maps.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Cascade maps with inter-panel arrows
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 3 — cascade maps…")
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5))
fig3.subplots_adjust(left=0.04, right=0.98, top=0.62,
                     bottom=0.12, wspace=0.40)
fig3.suptitle("Hypoxia Cascade: O₂ → HIF-1α → GD2 / CAIX",
              fontsize=13, fontweight="bold", y=0.98)

cascade_panels = [
    (O2_xy,   CMAP_O2,   0.0, O2_max, "O₂ Field",       "O₂ (mmHg)"),
    (HIF_xy,  CMAP_HIF,  0.0, 1.0,   "HIF-1α",          "HIF-1α (0–1)"),
    (GD2_xy,  CMAP_GD2,  0.0, 1.0,   "GD2 — Blind Zone","GD2 (0–1)"),
    (CAIX_xy, CMAP_CAIX, 0.0, 1.0,   "CAIX — Target",   "CAIX (0–1)"),
]

for ax, (field, cmap, vmin, vmax, title, cblabel) in zip(axes3, cascade_panels):
    cf = ax.contourf(coords, coords, field.T, levels=40,
                     cmap=cmap, vmin=vmin, vmax=vmax)
    fig3.colorbar(cf, ax=ax, label=cblabel, shrink=0.88, pad=0.02)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-8.0, 8.0)
    ax.grid(True, alpha=0.25)

# Black boundary circles on GD2 (index 2) and CAIX (index 3)
axes3[2].plot(4.3 * np.cos(THETA_RNG), 4.3 * np.sin(THETA_RNG),
              "k-", lw=1.5, alpha=0.7)
axes3[3].plot(5.5 * np.cos(THETA_RNG), 5.5 * np.sin(THETA_RNG),
              "k-", lw=1.5, alpha=0.7)

# Inter-panel arrows at figure level
pos = [ax.get_position() for ax in axes3]

def add_cascade_arrow(fig, x0, x1, y, label, color):
    arrow = FancyArrowPatch(
        (x0 + 0.006, y), (x1 - 0.006, y),
        transform=fig.transFigure,
        arrowstyle="->", mutation_scale=20,
        color=color, lw=2.2, clip_on=False,
    )
    fig.add_artist(arrow)
    fig.text((x0 + x1) / 2.0, y + 0.038, label,
             transform=fig.transFigure,
             ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=color)

add_cascade_arrow(fig3, pos[0].x1, pos[1].x0, y=0.76,
                  label="HIF-1α\nactivation", color=C_HIF)
add_cascade_arrow(fig3, pos[1].x1, pos[2].x0, y=0.72,
                  label="GD2\nsuppression", color=C_GD2)
add_cascade_arrow(fig3, pos[1].x1, pos[3].x0, y=0.84,
                  label="CAIX induction", color=C_CAIX)

save_fig(fig3, "antigen_fig3_cascade_maps.png", transparent=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — 2×2 Therapeutic implications
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 4 — killability maps…")
fig4, axes4 = plt.subplots(2, 2, figsize=(13, 11))
fig4.suptitle("Therapeutic Landscape — CAR-T Kill Probability & Coverage",
              fontsize=14, fontweight="bold")

# ── Top-left: GD2 kill map ────────────────────────────────────────────────────
ax = axes4[0, 0]
cf = ax.contourf(coords, coords, KILL.T, levels=50,
                 cmap=CMAP_GD2, vmin=0.0, vmax=1.0)
cs = ax.contour(coords, coords, KILL.T, levels=[0.5],
                colors="white", linewidths=2.0, linestyles="--")
ax.clabel(cs, fmt={0.5: "50% kill threshold"}, fontsize=8, inline=True)
fig4.colorbar(cf, ax=ax, label="Kill probability (0–1)", shrink=0.88)
ax.set_title("Standard CAR-T Kill Probability (GD2)")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

# ── Top-right: CAIX therapeutic map ──────────────────────────────────────────
ax = axes4[0, 1]
cf = ax.contourf(coords, coords, CAIX_xy.T, levels=50,
                 cmap=CMAP_CAIX, vmin=0.0, vmax=1.0)
cs = ax.contour(coords, coords, CAIX_xy.T, levels=[0.5],
                colors="white", linewidths=2.0, linestyles="--")
ax.clabel(cs, fmt="CAIX=0.5", fontsize=8, inline=True)
fig4.colorbar(cf, ax=ax, label="CAIX expression (0–1)", shrink=0.88)
ax.set_title("CAIX-Targeting Kill Probability")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

# ── Bottom-left: Therapeutic gap GAP = CAIX - GD2 ───────────────────────────
ax = axes4[1, 0]
gap_max = float(GAP.max())
cf = ax.contourf(coords, coords, GAP.T, levels=50,
                 cmap="RdYlGn", vmin=0.0, vmax=gap_max)
cs = ax.contour(coords, coords, GAP.T, levels=[0.0],
                colors="black", linewidths=1.5, linestyles="--")
ax.clabel(cs, fmt="GAP=0", fontsize=8, inline=True)
fig4.colorbar(cf, ax=ax, label="Therapeutic gain (CAIX − GD2, ≥0)", shrink=0.88)
ax.set_title("Therapeutic Gain: CAIX vs GD2  (positive = CAIX better)")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
ax.set_aspect("equal"); ax.grid(True, alpha=0.25)

# ── Bottom-right: Dual coverage RGB map ──────────────────────────────────────
ax = axes4[1, 1]
gd2_on  = GD2_xy  > 0.5
caix_on = CAIX_xy > 0.5

coverage_rgb = np.ones((N, N, 3))
coverage_rgb[ gd2_on & ~caix_on] = [0.45, 0.00, 0.65]
coverage_rgb[~gd2_on &  caix_on] = [0.10, 0.35, 0.75]
coverage_rgb[ gd2_on &  caix_on] = [0.00, 0.65, 0.55]
coverage_rgb[~gd2_on & ~caix_on] = [0.85, 0.85, 0.85]

ax.imshow(coverage_rgb.transpose(1, 0, 2), origin="lower", extent=EXTENT,
          aspect="equal", interpolation="nearest")
ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)

legend_patches = [
    mpatches.Patch(color=[0.45, 0.00, 0.65], label="GD2 only (CAR-T active)"),
    mpatches.Patch(color=[0.10, 0.35, 0.75], label="CAIX only (rescue zone)"),
    mpatches.Patch(color=[0.00, 0.65, 0.55], label="Both active"),
    mpatches.Patch(color=[0.85, 0.85, 0.85], label="Neither (dead zone)"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right",
          framealpha=0.95)
ax.set_title("Therapy Coverage Map — GD2 vs CAIX vs Dual")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.grid(True, alpha=0.25)

fig4.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig4, "antigen_fig4_killability.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Radial profiles + spatial gradients
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 5 — radial profiles…")
fig5, (ax5l, ax5r) = plt.subplots(1, 2, figsize=(16, 6))
fig5.suptitle("Radial Molecular Analysis", fontsize=14, fontweight="bold")

ax5l.plot(x_1d, HIF_1d,  color=C_HIF,  lw=2.3, label="HIF-1α")
ax5l.plot(x_1d, GD2_1d,  color=C_GD2,  lw=2.3, label="GD2")
ax5l.plot(x_1d, CAIX_1d, color=C_CAIX, lw=2.3, label="CAIX")

# HIF transition — thin unlabeled dashed
ax5l.axvline(r_hif_trans, color=C_HIF, lw=0.9, ls="--", alpha=0.55)

# GD2 blind boundary — labeled
ax5l.axvline(r_gd2_trans, color="#9B1FBE", lw=1.2, ls="--", alpha=0.7)
ax5l.text(r_gd2_trans + 0.15, 0.05, "GD2 blind boundary",
          color="#9B1FBE", fontsize=8, rotation=90, va="bottom", ha="left")

# CAIX active boundary — labeled
ax5l.axvline(r_caix_trans, color="#1A6FBF", lw=1.2, ls="--", alpha=0.7)
ax5l.text(r_caix_trans + 0.15, 0.05, "CAIX active boundary",
          color="#1A6FBF", fontsize=8, rotation=90, va="bottom", ha="left")

ax5l_r = ax5l.twinx()
ax5l_r.plot(x_1d, O2_1d, color=C_O2, lw=2.3, label="O₂ (mmHg)", zorder=1)
ax5l_r.set_ylabel("O₂ (mmHg)", color=C_O2, fontsize=11)
ax5l_r.tick_params(axis="y", colors=C_O2)
ax5l_r.set_ylim(0.0, 42.0)

ax5l.set_xlim(0.0, domain)
ax5l.set_ylim(0.0, 1.08)
ax5l.set_xlabel("Radius from centre (mm)")
ax5l.set_ylabel("Normalised expression (0–1)")
ax5l.set_title("Radial Molecular Profiles")
ax5l.grid(True)

lines_l, labels_l = ax5l.get_legend_handles_labels()
lines_r, labels_r = ax5l_r.get_legend_handles_labels()
ax5l.legend(lines_l + lines_r, labels_l + labels_r,
            loc="lower right", fontsize=8, framealpha=0.95)

dHIF_dr  = np.gradient(HIF_1d,  x_1d)
dGD2_dr  = np.gradient(GD2_1d,  x_1d)
dCAIX_dr = np.gradient(CAIX_1d, x_1d)

ax5r.plot(x_1d, dHIF_dr,  color=C_HIF,  lw=2.3, label="dHIF/dr")
ax5r.plot(x_1d, dGD2_dr,  color=C_GD2,  lw=2.3, label="dGD2/dr")
ax5r.plot(x_1d, dCAIX_dr, color=C_CAIX, lw=2.3, label="dCAIX/dr")
ax5r.axhline(0.0, color="#555555", lw=0.8, ls="-")

for rv, color in [(r_hif_trans, C_HIF), (r_gd2_trans, C_GD2),
                  (r_caix_trans, C_CAIX)]:
    ax5r.axvline(rv, color=color, lw=0.9, ls="--", alpha=0.55)

ax5r.set_xlim(0.0, domain)
ax5r.set_xlabel("Radius from centre (mm)")
ax5r.set_ylabel("Rate of change per mm")
ax5r.set_title("Spatial Gradient — Where Transitions Occur")
ax5r.legend(fontsize=9, loc="lower right")
ax5r.grid(True)

fig5.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig5, "antigen_fig5_radial_profiles.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Zone analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 6 — zone analysis…")
fig6, (ax6l, ax6r) = plt.subplots(1, 2, figsize=(15, 6))
fig6.suptitle("Therapy Coverage Analysis by Tumor Zone",
              fontsize=14, fontweight="bold")

zone_labels = ["Core\n(r<2mm)", "Mid-tumor\n(2–5mm)", f"Edge\n(5mm–r_t)"]
zone_masks_3d = [
    tumor_mask & (R3 < 2.0),
    tumor_mask & (R3 >= 2.0) & (R3 < 5.0),
    tumor_mask & (R3 >= 5.0) & (R3 < r_tumor),
]

hif_means  = [float(HIF[m].mean())  if m.any() else 0.0 for m in zone_masks_3d]
gd2_means  = [float(GD2[m].mean())  if m.any() else 0.0 for m in zone_masks_3d]
caix_means = [float(CAIX[m].mean()) if m.any() else 0.0 for m in zone_masks_3d]
kill_means = [float(GD2[m].mean())  if m.any() else 0.0 for m in zone_masks_3d]

x_pos = np.arange(len(zone_labels))
w = 0.20
b_h = ax6l.bar(x_pos - 1.5*w, hif_means,  w, color=C_HIF,    label="HIF-1α",
               edgecolor="#333", lw=0.8)
b_g = ax6l.bar(x_pos - 0.5*w, gd2_means,  w, color=C_GD2,    label="GD2",
               edgecolor="#333", lw=0.8)
b_c = ax6l.bar(x_pos + 0.5*w, caix_means, w, color=C_CAIX,   label="CAIX",
               edgecolor="#333", lw=0.8)
b_k = ax6l.bar(x_pos + 1.5*w, kill_means, w, color="#CC4400", label="Kill prob.",
               edgecolor="#333", lw=0.8)

for bars in (b_h, b_g, b_c, b_k):
    for bar in bars:
        ax6l.text(bar.get_x() + bar.get_width() / 2.0,
                  bar.get_height() + 0.012,
                  f"{bar.get_height():.1f}",
                  ha="center", va="bottom", fontsize=7.5)

ax6l.axhline(0.5, color="red", ls="--", lw=1.5, label="Threshold (0.5)")
ax6l.set_xticks(x_pos); ax6l.set_xticklabels(zone_labels, fontsize=9)
ax6l.set_ylim(0.0, 1.25)
ax6l.set_ylabel("Mean expression (0–1)")
ax6l.set_title("Mean Molecular Expression by Tumor Zone")
ax6l.legend(fontsize=8, loc="upper right"); ax6l.grid(True, axis="y")

r_flat    = R3[tumor_mask]
g_flat    = GD2_act[tumor_mask]
c_flat    = CAIX_act[tumor_mask]
sort_idx  = np.argsort(r_flat)
r_s       = r_flat[sort_idx]
g_s       = g_flat[sort_idx]
c_s       = c_flat[sort_idx]
total_t   = len(r_s)

gd2_only_s  = ( g_s & ~c_s).astype(float)
caix_only_s = (~g_s &  c_s).astype(float)
both_s      = ( g_s &  c_s).astype(float)
neither_s   = (~g_s & ~c_s).astype(float)

cum_go = np.cumsum(gd2_only_s)  / total_t
cum_co = np.cumsum(caix_only_s) / total_t
cum_b  = np.cumsum(both_s)      / total_t
cum_n  = np.cumsum(neither_s)   / total_t

r_plot  = np.linspace(0.0, r_tumor, 300)
go_plot = np.interp(r_plot, r_s, cum_go)
co_plot = np.interp(r_plot, r_s, cum_co)
b_plot  = np.interp(r_plot, r_s, cum_b)
n_plot  = np.interp(r_plot, r_s, cum_n)

ax6r.stackplot(r_plot,
               go_plot, co_plot, b_plot, n_plot,
               labels=["GD2 only (CAR-T active)",
                       "CAIX rescue (CAIX>0.5, GD2<0.5)",
                       "Dual coverage (both>0.5)",
                       "Neither (therapy gap)"],
               colors=["#9B1FBE", "#1A6FBF", "#20B2AA", "#CCCCCC"],
               alpha=0.85)

ax6r.set_xlim(0.0, r_tumor)
ax6r.set_ylim(0.0, 1.0)
ax6r.set_xlabel("Radius from centre (mm)")
ax6r.set_ylabel("Cumulative fraction of tumor volume")
ax6r.set_title("Cumulative Therapy Coverage vs Radius")
ax6r.legend(fontsize=8, loc="upper left"); ax6r.grid(True, alpha=0.5)

fig6.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig6, "antigen_fig6_zone_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Statistics 1×3
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 7 — statistics…")
fig7, axes7 = plt.subplots(1, 3, figsize=(16, 5.5))
fig7.suptitle("GBM Antigen Model — Distribution Statistics",
              fontsize=14, fontweight="bold")

GD2_tumor  = GD2[tumor_mask]
CAIX_tumor = CAIX[tumor_mask]
HIF_tumor  = HIF[tumor_mask]

ax = axes7[0]
n_bins = 60
counts_g, edges_g, patches_g = ax.hist(GD2_tumor, bins=n_bins, edgecolor="none")
norm_g = plt.Normalize(0.0, 1.0)
for patch, left in zip(patches_g, edges_g[:-1]):
    patch.set_facecolor(CMAP_GD2(norm_g(left + (edges_g[1] - edges_g[0]) / 2.0)))
ax.axvspan(0.0, 0.5, alpha=0.12, color="#9B1FBE")
ax.axvspan(0.5, 1.0, alpha=0.07, color="#FF88CC")
ax.axvline(0.5, color="#9B1FBE", ls="--", lw=2.0, label="GD2 = 0.5")
ax.text(0.03, 0.97, f"CAR-T blind: {gd2_blind_fraction*100:.1f}%",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold", color="#9B1FBE")
ax.text(0.03, 0.88, f"CAR-T active: {(1-gd2_blind_fraction)*100:.1f}%",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold", color="#CC4488")
ax.set_xlabel("GD2 value"); ax.set_ylabel("Voxel count")
ax.set_xlim(0.0, 1.0)
ax.set_title("GD2 Distribution in Tumor")
ax.legend(fontsize=8); ax.grid(True)

ax = axes7[1]
counts_c, edges_c, patches_c = ax.hist(CAIX_tumor, bins=n_bins, edgecolor="none")
norm_c = plt.Normalize(0.0, 1.0)
for patch, left in zip(patches_c, edges_c[:-1]):
    patch.set_facecolor(CMAP_CAIX(norm_c(left + (edges_c[1] - edges_c[0]) / 2.0)))
ax.axvspan(0.5, 1.0, alpha=0.15, color="#1A6FBF")
ax.axvspan(0.0, 0.5, alpha=0.07, color="#CCCCCC")
ax.axvline(0.5, color=C_CAIX, ls="--", lw=2.0, label="CAIX = 0.5")
ax.text(0.97, 0.97, f"Targetable: {caix_targetable_fraction*100:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, fontweight="bold", color=C_CAIX)
ax.text(0.03, 0.97, f"Not targeted: {(1-caix_targetable_fraction)*100:.1f}%",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold", color="#888888")
ax.set_xlabel("CAIX value"); ax.set_ylabel("Voxel count")
ax.set_xlim(0.0, 1.0)
ax.set_title("CAIX Distribution in Tumor")
ax.legend(fontsize=8); ax.grid(True)

ax = axes7[2]
key_r   = [0.0, 1.0, sigma, 2.0 * sigma, domain]
key_lbl = ["Core\n(r=0)", "r=1mm",
           f"r=σ\n({sigma:.0f}mm)", f"r=2σ\n({2*sigma:.0f}mm)",
           f"Periph.\n({domain:.0f}mm)"]
key_O2_v = [float(np.clip(O2_max - beta * rho_0 *
                           np.exp(-r**2 / (2.0 * sigma**2)), 0.0, O2_max))
            for r in key_r]
key_HIF  = [float(1.0 / (1.0 + np.exp(+k_hif  * (v - O2_half_hif)))) for v in key_O2_v]
key_GD2  = [float(1.0 / (1.0 + np.exp(-k_gd2  * (v - O2_half_gd2)))) for v in key_O2_v]
key_CAIX = [float(1.0 / (1.0 + np.exp(+k_caix * (v - O2_half_caix)))) for v in key_O2_v]

x_pos7 = np.arange(len(key_lbl))
w7     = 0.26
bh = ax.bar(x_pos7 - w7, key_HIF,  w7, color=C_HIF,  label="HIF-1α",
            edgecolor="#333", lw=0.8)
bg = ax.bar(x_pos7,      key_GD2,  w7, color=C_GD2,  label="GD2",
            edgecolor="#333", lw=0.8)
bc = ax.bar(x_pos7 + w7, key_CAIX, w7, color=C_CAIX, label="CAIX",
            edgecolor="#333", lw=0.8)
for bars in (bh, bg, bc):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5)
ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Threshold (0.5)")
ax.set_xticks(x_pos7); ax.set_xticklabels(key_lbl, fontsize=8.5)
ax.set_ylim(0.0, 1.25)
ax.set_ylabel("Expression (0–1)")
ax.set_title("HIF / GD2 / CAIX at Key Radial Distances")
ax.legend(fontsize=8); ax.grid(True, axis="y")

fig7.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig7, "antigen_fig7_statistics.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Predicted therapeutic outcome (two-panel spatial map)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 8 — outcome prediction…")

# Build 2D category maps for central slice
outside_2d = rho_xy < 0.05

# LEFT panel categories: 0=outside, 1=blind/resistant, 2=kills
cat_left = np.zeros((N, N), dtype=int)
cat_left[~outside_2d & (GD2_xy < 0.5)]  = 1   # red  — blind zone
cat_left[~outside_2d & (GD2_xy >= 0.5)] = 2   # green — kills

# RIGHT panel categories: 0=outside, 1=neither(red), 2=GD2-outer(green), 3=CAIX(blue)
cat_right = np.zeros((N, N), dtype=int)
_in = ~outside_2d
cat_right[_in & (CAIX_xy > 0.5)]                          = 3   # blue  — CAIX kills
cat_right[_in & (CAIX_xy <= 0.5) & (GD2_xy > 0.5)]       = 2   # green — GD2 outer ring
cat_right[_in & (CAIX_xy <= 0.5) & (GD2_xy <= 0.5)]      = 1   # red   — neither

cmap_left  = ListedColormap(["#CCCCCC", "#CC3333", "#44AA44"])
cmap_right = ListedColormap(["#CCCCCC", "#CC3333", "#44AA44", "#1A6FBF"])

# Upsample for smoother circles (order=0 = nearest, preserves discrete categories)
_ZF = 3
cat_left_up  = ndimage_zoom(cat_left,  _ZF, order=0)
cat_right_up = ndimage_zoom(cat_right, _ZF, order=0)

fig8, (ax8l, ax8r) = plt.subplots(1, 2, figsize=(14, 7))
fig8.suptitle("Predicted Therapeutic Outcome — Antigen Distribution Model",
              fontsize=14, fontweight="bold")

# ── Left panel ────────────────────────────────────────────────────────────────
im = ax8l.imshow(cat_left_up.T, origin="lower", extent=EXTENT,
                 cmap=cmap_left, vmin=0, vmax=2,
                 aspect="equal", interpolation="nearest")
ax8l.set_xlim(-8, 8); ax8l.set_ylim(-8, 8)
ax8l.set_title("GD2 CAR-T — Standard Therapy")
ax8l.set_xlabel("x (mm)"); ax8l.set_ylabel("y (mm)")
ax8l.grid(True, alpha=0.25)

legend_left = [
    mpatches.Patch(color="#CC3333", label="Therapy-resistant core"),
    mpatches.Patch(color="#44AA44", label="CAR-T kills here"),
    mpatches.Patch(color="#CCCCCC", label="Outside tumor"),
]
ax8l.legend(handles=legend_left, fontsize=9, loc="lower right", framealpha=0.95)

# Arrow annotation pointing to resistant core
ax8l.annotate(
    "Core escapes → relapse",
    xy=(0.0, 0.0),
    xytext=(3.5, 5.5),
    fontsize=9, color="#CC3333", fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color="#CC3333", lw=1.5),
)

# Stats text box below panel
blind_pct = gd2_blind_fraction * 100.0
ax8l.text(0.5, -0.12,
          f"GD2-blind volume: {blind_pct:.1f}% of tumor",
          transform=ax8l.transAxes, ha="center", va="top",
          fontsize=10, fontweight="bold", color="#CC3333",
          bbox=dict(boxstyle="round", fc="#FFF0F0", ec="#CC3333", alpha=0.9))

# ── Right panel ───────────────────────────────────────────────────────────────
ax8r.imshow(cat_right_up.T, origin="lower", extent=EXTENT,
            cmap=cmap_right, vmin=0, vmax=3,
            aspect="equal", interpolation="nearest")
ax8r.set_xlim(-8, 8); ax8r.set_ylim(-8, 8)
ax8r.set_title("CAIX CAR-T — Rescue Therapy")
ax8r.set_xlabel("x (mm)"); ax8r.set_ylabel("y (mm)")
ax8r.grid(True, alpha=0.25)

legend_right = [
    mpatches.Patch(color="#1A6FBF", label="CAIX active — core eliminated"),
    mpatches.Patch(color="#44AA44", label="GD2 active — outer ring"),
    mpatches.Patch(color="#CC3333", label="Neither active"),
    mpatches.Patch(color="#CCCCCC", label="Outside tumor"),
]
ax8r.legend(handles=legend_right, fontsize=9, loc="lower right", framealpha=0.95)

ax8r.annotate(
    "Core eliminated",
    xy=(0.0, 0.0),
    xytext=(3.5, 5.5),
    fontsize=9, color="#1A6FBF", fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="->", color="#1A6FBF", lw=1.5),
)

caix_pct = caix_targetable_fraction * 100.0
ax8r.text(0.5, -0.12,
          f"CAIX covers: {caix_pct:.1f}% of tumor",
          transform=ax8r.transAxes, ha="center", va="top",
          fontsize=10, fontweight="bold", color="#1A6FBF",
          bbox=dict(boxstyle="round", fc="#F0F4FF", ec="#1A6FBF", alpha=0.9))

fig8.tight_layout(rect=[0, 0.06, 1, 0.96])
save_fig(fig8, "antigen_fig8_outcome_prediction.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Summary statistics (3-panel)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 9 — summary stats…")
fig9, (ax9l, ax9m, ax9r) = plt.subplots(1, 3, figsize=(15, 5))
fig9.suptitle("S.M.A.R.C. GBM — Model Summary",
              fontsize=14, fontweight="bold")

# ── Left: Horizontal bar chart — therapy coverage breakdown ───────────────────
bar_labels  = ["GD2 only\n(CAR-T kills)",
               "CAIX rescue\nzone",
               "Dual\ncoverage",
               "Neither\n(gap)"]
bar_values  = [gd2_only_fraction  * 100.0,
               caix_only_fraction * 100.0,
               dual_fraction      * 100.0,
               neither_fraction   * 100.0]
bar_colors  = ["#9B1FBE", "#1A6FBF", "#009999", "#AAAAAA"]

y_pos = np.arange(len(bar_labels))
bars9 = ax9l.barh(y_pos, bar_values, color=bar_colors,
                  edgecolor="#333333", linewidth=0.8, height=0.55)

for bar, val in zip(bars9, bar_values):
    ax9l.text(val + 0.5, bar.get_y() + bar.get_height() / 2.0,
              f"{val:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold")

ax9l.set_yticks(y_pos)
ax9l.set_yticklabels(bar_labels, fontsize=9)
ax9l.set_xlim(0, 85)
ax9l.set_xlabel("% of tumor volume")
ax9l.set_title("Therapy Coverage Breakdown")
ax9l.axvline(0, color="#333333", lw=0.8)
ax9l.grid(True, axis="x", alpha=0.4)
ax9l.invert_yaxis()

# ── Middle: Parameter / metrics table ────────────────────────────────────────
ax9m.axis("off")
ax9m.set_title("Model Parameters & Key Metrics")

table_data = [
    ["Tumor radius",    "7.35 mm"],
    ["Hypoxic radius",  "4.3 mm"],
    ["GD2 blind zone",  f"r < {r_gd2_trans:.1f} mm"],
    ["CAIX active zone",f"r < {r_caix_trans:.1f} mm"],
    ["CAIX rescues",    f"{caix_rescue_pct:.0f}% of blind zone"],
    ["HIF-1α O\u2082_half", "5.0 mmHg"],
    ["GD2 O\u2082_half",    "7.0 mmHg"],
    ["CAIX O\u2082_half",   "20.0 mmHg"],
]

# Alternating row colors
row_colors = []
for i in range(len(table_data)):
    c = "#F5F0FF" if i % 2 == 0 else "#FFFFFF"
    row_colors.append([c, c])

tbl = ax9m.table(
    cellText=table_data,
    colLabels=["Metric", "Value"],
    cellLoc="left",
    loc="center",
    cellColours=row_colors,
    colColours=["#9B1FBE", "#9B1FBE"],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.15, 1.6)

# Style header cells
for j in range(2):
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# ── Right: Therapeutic window line plot ──────────────────────────────────────
O2_line_r  = np.linspace(0.0, O2_max, 500)
GD2_line_r  = 1.0 / (1.0 + np.exp(-k_gd2  * (O2_line_r - O2_half_gd2)))
CAIX_line_r = 1.0 / (1.0 + np.exp(+k_caix * (O2_line_r - O2_half_caix)))

# Shade between the two O2_half values: GD2=0.5 (~7 mmHg) to CAIX=0.5 (~20 mmHg)
gd2_half_o2  = O2_line_r[np.argmin(np.abs(GD2_line_r  - 0.5))]
caix_half_o2 = O2_line_r[np.argmin(np.abs(CAIX_line_r - 0.5))]
window_mask  = (O2_line_r >= gd2_half_o2) & (O2_line_r <= caix_half_o2)
ax9r.fill_between(O2_line_r, 0.0, 1.0, where=window_mask,
                  color="#009999", alpha=0.2, label="CAIX rescue window")

ax9r.plot(O2_line_r, GD2_line_r,  color=C_GD2,  lw=2.3, label="GD2")
ax9r.plot(O2_line_r, CAIX_line_r, color=C_CAIX, lw=2.3, label="CAIX")
ax9r.axhline(0.5, color="#888888", lw=1.0, ls="--", alpha=0.7)

# Vertical dashed lines at each O2_half
ax9r.axvline(O2_half_gd2,  color=C_GD2,  lw=1.0, ls="--", alpha=0.6,
             label=f"GD2 O\u2082_half = {O2_half_gd2} mmHg")
ax9r.axvline(O2_half_caix, color=C_CAIX, lw=1.0, ls="--", alpha=0.6,
             label=f"CAIX O\u2082_half = {O2_half_caix} mmHg")

# Rescue window label placed inside the shaded band
ax9r.text(13.0, 0.7, "CAIX rescue\nwindow",
          ha="center", va="center", fontsize=8.5,
          color="#007777", fontweight="bold")

ax9r.set_xlim(0.0, O2_max)
ax9r.set_ylim(0.0, 1.08)
ax9r.set_xlabel("O\u2082 (mmHg)")
ax9r.set_ylabel("Normalised expression")
ax9r.set_title("Therapeutic Window —\nCAIX Rescues GD2 Failure")
ax9r.legend(fontsize=8, loc="center right", framealpha=0.95)
ax9r.grid(True, alpha=0.4)

fig9.tight_layout(rect=[0, 0, 1, 0.94])
save_fig(fig9, "antigen_fig9_summary_stats.png")


# ══════════════════════════════════════════════════════════════════════════════
# Save arrays and metrics
# ══════════════════════════════════════════════════════════════════════════════
for arr, name in [(HIF, "hif_3D.npy"), (GD2, "gd2_3D.npy"), (CAIX, "caix_3D.npy")]:
    p = os.path.join(OUT, name)
    np.save(p, arr)
    print(f"Saved: {p}")

p = os.path.join(OUT, "antigen_metrics.json")
with open(p, "w") as f:
    json.dump(antigen_metrics, f, indent=2)
print(f"Saved: {p}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  ANTIGEN MODEL — METRICS SUMMARY")
print("═" * 60)
print(f"  GD2 blind fraction (tumor)         {gd2_blind_fraction*100:.2f}%")
print(f"  CAIX targetable fraction           {caix_targetable_fraction*100:.2f}%")
print(f"  GD2 only fraction (of tumor)       {gd2_only_fraction*100:.2f}%")
print(f"  CAIX only fraction (of tumor)      {caix_only_fraction*100:.2f}%")
print(f"  Dual coverage fraction             {dual_fraction*100:.2f}%")
print(f"  Therapy dead zone fraction         {neither_fraction*100:.2f}%")
print(f"  CAIX rescues of GD2 blind zone     {caix_rescue_pct:.1f}%")
print(f"  HIF transition radius              {r_hif_trans:.2f} mm")
print(f"  GD2 transition radius (blind edge) {r_gd2_trans:.2f} mm")
print(f"  CAIX transition radius (window)    {r_caix_trans:.2f} mm")
print("═" * 60)
print("All done.")
