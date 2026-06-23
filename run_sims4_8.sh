#!/bin/bash
# run_sims4_8.sh — Run PhysiCell Sims 4-8 sequentially with analytics
# Assumes Sim 3 is already running; this script handles 4-8.
set -e

PHYS=/home/rshef/PhysiCell
SIM_ROOT=/home/rshef/gbm_simulations
LOG=$SIM_ROOT/run_log.txt
ANALYZE="python3 $SIM_ROOT/analyze_treatment_sim.py"

run_sim() {
    local NUM=$1
    local CFG=$2
    local SIM_NAME=$3
    local LABEL=$4

    echo "" >> $LOG
    echo "=== Sim $NUM: $LABEL ===" >> $LOG
    date >> $LOG

    # Clean output dir
    rm -rf $SIM_ROOT/$SIM_NAME/physicell_output/*
    mkdir -p $SIM_ROOT/$SIM_NAME/physicell_output

    # Run simulation
    echo "[$(date '+%H:%M:%S')] Starting $SIM_NAME..." | tee -a $LOG
    cd $PHYS && ./project config/$CFG > $SIM_ROOT/$SIM_NAME/run.log 2>&1
    EXIT_CODE=$?
    echo "SIM${NUM}_EXIT=$EXIT_CODE" >> $LOG

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: Sim $NUM failed (exit $EXIT_CODE)" | tee -a $LOG
        tail -20 $SIM_ROOT/$SIM_NAME/run.log >> $LOG
    else
        echo "[$(date '+%H:%M:%S')] Sim $NUM done. Running analytics..." | tee -a $LOG
        $ANALYZE "$SIM_NAME" "$LABEL" >> $LOG 2>&1
        echo "[$(date '+%H:%M:%S')] Analytics done: $SIM_NAME" | tee -a $LOG
    fi
}

echo "[$(date '+%H:%M:%S')] Starting Sims 4-8 pipeline..." | tee -a $LOG
run_sim 4 gbm_sim4_caix.xml      caix_targeting_cart    "CAIX-Targeting CAR-T (500 cells)"
run_sim 5 gbm_sim5_dual.xml      dual_target_cart       "Dual-Target CAR-T (GD2+CAIX, 500 cells)"
run_sim 6 gbm_sim6_sequential.xml sequential_cart       "Sequential CAR-T (GD2 Day1 + CAIX Day7)"
run_sim 7 gbm_sim7_pulsed.xml    pulsed_cart            "Pulsed CAR-T (3x167 cells)"
run_sim 8 gbm_sim8_early.xml     early_cart             "Early Intervention CAR-T (Day 0.25)"

# ── Grand summary ─────────────────────────────────────────────────────────────
echo "" >> $LOG
echo "=== FINAL SUMMARY ===" >> $LOG
date >> $LOG

python3 - >> $LOG 2>&1 << 'PYEOF'
import os, csv, glob

SIM_ROOT = os.path.expanduser("~/gbm_simulations")
sims = [
    ("baseline_cart_treatment", "Baseline CAR-T (500 GD2)"),
    ("highdose_cart",           "High-Dose CAR-T (2000)"),
    ("caix_targeting_cart",     "CAIX-Targeting (500)"),
    ("dual_target_cart",        "Dual-Target (500)"),
    ("sequential_cart",         "Sequential"),
    ("pulsed_cart",             "Pulsed"),
    ("early_cart",              "Early (Day 0.25)"),
]

all_rows = []
for sim_name, label in sims:
    cmp_csv = os.path.join(SIM_ROOT, sim_name, "comparison", "efficacy_metrics.csv")
    if not os.path.exists(cmp_csv):
        continue
    with open(cmp_csv) as f:
        rows = list(csv.DictReader(f))
        if rows:
            all_rows.append(rows[0])

if all_rows:
    header = list(all_rows[0].keys())
    print("\nEfficacy Summary Table:")
    print(f"{'Sim':<35} {'Final Tumor':>12} {'Efficacy%':>10} {'vs Baseline%':>14}")
    print("-"*75)
    for r in all_rows:
        print(f"{r.get('label','?'):<35} {r.get('final_tumor','?'):>12} "
              f"{r.get('efficacy_vs_untreated_pct','?'):>10} "
              f"{r.get('efficacy_vs_baseline_pct','?'):>14}")
PYEOF

echo "" >> $LOG
echo "=== ALL SIMS COMPLETE ===" >> $LOG
date >> $LOG
echo "[$(date '+%H:%M:%S')] All sims 4-8 complete."
