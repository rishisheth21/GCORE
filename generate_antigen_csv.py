#!/usr/bin/env python3
"""
S.M.A.R.C. GBM Pipeline — Antigen CSV Generator
generate_antigen_csv.py

Reads physicell_o2.csv from outputs/, applies piecewise linear antigen
formulas matching gbm_smarc.cpp tumor_cell_rule, writes physicell_antigens.csv.

GD2:
  pO2 < 5.0:  GD2 = max(0, pO2/5.0) * 0.2          (0.0 to 0.2 in hypoxic)
  pO2 >= 5.0: GD2 = 0.8 + 0.2*(pO2-5.0)/33.0       (0.8 to 1.0 in normoxic)

CAIX:
  pO2 < 5.0:  CAIX = 0.8 + 0.2*(5.0-pO2)/5.0       (0.8 to 1.0 in hypoxic)
  pO2 >= 5.0: CAIX = max(0, 0.2*(1-(pO2-5.0)/33.0)) (0.2 to 0.0 in normoxic)

HIF1a:
  pO2 < 5.0:  HIF1a = 1.0 - pO2/5.0                 (1.0 to 0.0 in hypoxic)
  pO2 >= 5.0: HIF1a = 0.0
"""

import numpy as np
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.expanduser("~/smarc_gbm")
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ─── Load O2 CSV ──────────────────────────────────────────────────────────────
o2_csv = os.path.join(OUT, "physicell_o2.csv")
print(f"Loading O2 CSV: {o2_csv}")
data = np.loadtxt(o2_csv, delimiter=",", skiprows=1)
x_um = data[:, 0]
y_um = data[:, 1]
o2   = data[:, 2]
print(f"  Loaded {len(o2)} voxels")
print(f"  x range: {x_um.min():.1f} .. {x_um.max():.1f} um")
print(f"  y range: {y_um.min():.1f} .. {y_um.max():.1f} um")
print(f"  pO2 range: {o2.min():.4f} .. {o2.max():.4f} mmHg")

# ─── Piecewise antigen formulas ────────────────────────────────────────────────
def compute_gd2(pO2):
    """GD2 piecewise linear: low in hypoxic, high in normoxic."""
    gd2 = np.where(
        pO2 < 5.0,
        np.maximum(0.0, pO2 / 5.0) * 0.2,
        0.8 + 0.2 * (pO2 - 5.0) / 33.0
    )
    return np.clip(gd2, 0.0, 1.0)

def compute_caix(pO2):
    """CAIX piecewise linear: high in hypoxic, low in normoxic."""
    caix = np.where(
        pO2 < 5.0,
        0.8 + 0.2 * (5.0 - pO2) / 5.0,
        np.maximum(0.0, 0.2 * (1.0 - (pO2 - 5.0) / 33.0))
    )
    return np.clip(caix, 0.0, 1.0)

def compute_hif1a(pO2):
    """HIF1a: active under hypoxia, zero normoxic."""
    hif1a = np.where(
        pO2 < 5.0,
        1.0 - pO2 / 5.0,
        0.0
    )
    return np.clip(hif1a, 0.0, 1.0)

gd2   = compute_gd2(o2)
caix  = compute_caix(o2)
hif1a = compute_hif1a(o2)

print(f"\nAntigen value ranges:")
print(f"  GD2:   {gd2.min():.4f} .. {gd2.max():.4f}")
print(f"  CAIX:  {caix.min():.4f} .. {caix.max():.4f}")
print(f"  HIF1a: {hif1a.min():.4f} .. {hif1a.max():.4f}")

# ─── Verification checks ──────────────────────────────────────────────────────
r_um = np.sqrt(x_um**2 + y_um**2)
r_hyp_um = 4300.0

inside_mask  = r_um < r_hyp_um
outside_mask = r_um >= r_hyp_um

mean_gd2_in   = float(gd2[inside_mask].mean())   if inside_mask.any()  else 0.0
mean_gd2_out  = float(gd2[outside_mask].mean())  if outside_mask.any() else 0.0
mean_caix_in  = float(caix[inside_mask].mean())  if inside_mask.any()  else 0.0
mean_caix_out = float(caix[outside_mask].mean()) if outside_mask.any() else 0.0

print(f"\nVerification checks (r_hyp = {r_hyp_um:.0f} um):")
print(f"  Mean GD2  inside  r<{r_hyp_um:.0f}um: {mean_gd2_in:.4f}  (target: < 0.25)")
print(f"  Mean GD2  outside r>={r_hyp_um:.0f}um: {mean_gd2_out:.4f}  (target: > 0.80)")
print(f"  Mean CAIX inside  r<{r_hyp_um:.0f}um: {mean_caix_in:.4f}  (target: > 0.50)")
print(f"  Mean CAIX outside r>={r_hyp_um:.0f}um: {mean_caix_out:.4f}  (target: < 0.20)")

checks_passed = True
if mean_gd2_in >= 0.25:
    print("  FAIL: Mean GD2 inside is too high!")
    checks_passed = False
else:
    print("  PASS: Mean GD2 inside is low (hypoxic zone)")

if mean_gd2_out <= 0.80:
    print("  FAIL: Mean GD2 outside is too low!")
    checks_passed = False
else:
    print("  PASS: Mean GD2 outside is high (normoxic zone)")

if mean_caix_in <= 0.50:
    print("  FAIL: Mean CAIX inside is too low!")
    checks_passed = False
else:
    print("  PASS: Mean CAIX inside is high (hypoxic zone)")

if mean_caix_out >= 0.20:
    print("  FAIL: Mean CAIX outside is too high!")
    checks_passed = False
else:
    print("  PASS: Mean CAIX outside is low (normoxic zone)")

# Monotonicity check: sort by pO2 and verify GD2 increases, CAIX decreases
sorted_idx = np.argsort(o2)
o2_sorted   = o2[sorted_idx]
gd2_sorted  = gd2[sorted_idx]
caix_sorted = caix[sorted_idx]

# Check monotonicity in bulk (allow tiny numerical noise from clip)
gd2_mono  = bool(np.all(np.diff(gd2_sorted)  >= -1e-10))
caix_mono = bool(np.all(np.diff(caix_sorted) <= 1e-10))
print(f"\nMonotonicity checks:")
print(f"  GD2  monotonically non-decreasing with pO2: {'PASS' if gd2_mono else 'FAIL'}")
print(f"  CAIX monotonically non-increasing with pO2: {'PASS' if caix_mono else 'FAIL'}")
if not gd2_mono or not caix_mono:
    checks_passed = False

if checks_passed:
    print("\n  ALL CHECKS PASSED")
else:
    print("\n  WARNING: Some checks failed — review antigen formulas")

# ─── Write physicell_antigens.csv ─────────────────────────────────────────────
antigen_csv = os.path.join(OUT, "physicell_antigens.csv")
header = "x_micron,y_micron,GD2,CAIX,HIF1a"
out_data = np.column_stack([x_um, y_um, gd2, caix, hif1a])
np.savetxt(antigen_csv, out_data,
           delimiter=",", header=header, comments="", fmt="%.6f")
print(f"\nSaved: {antigen_csv}  ({len(gd2)} rows)")

print("\n" + "=" * 60)
print("Antigen CSV generation complete")
print("=" * 60)
print(f"  Mean GD2  in/out = {mean_gd2_in:.4f} / {mean_gd2_out:.4f}")
print(f"  Mean CAIX in/out = {mean_caix_in:.4f} / {mean_caix_out:.4f}")
print("=" * 60)
