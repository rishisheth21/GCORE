#!/bin/bash
# run_powerlaw_all.sh — Run all 7 power-law O2 simulations sequentially.
# Fast settings: dx=200, dt_diff=0.05, output every 24h.
# Fix errors and keep going. Log everything.

set -e
PHYS=/home/rshef/PhysiCell
PL=/home/rshef/gbm_simulations/powerlaw_o2
MAIN_LOG=/home/rshef/gbm_simulations/run_log.txt
ANALYZE="python3 /home/rshef/gbm_simulations/analyze_pl_sim.py"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $MAIN_LOG; }

log ""
log "=== Power Law Simulations Start ==="
date >> $MAIN_LOG

run_sim() {
    local NUM=$1; local CFG=$2; local SIM_NAME=$3; local LABEL="$4"
    local START=$(date +%s)
    log "--- Sim $NUM: $LABEL ---"

    # Clear output
    rm -rf $PL/$SIM_NAME/physicell_output
    mkdir -p $PL/$SIM_NAME/physicell_output

    # Run
    cd $PHYS
    timeout 900 ./project config/$CFG > $PL/$SIM_NAME/run.log 2>&1
    local EXIT=$?
    local END=$(date +%s)
    local ELAPSED=$((END-START))

    if [ $EXIT -ne 0 ] && [ $EXIT -ne 124 ]; then
        log "  ERROR exit=$EXIT elapsed=${ELAPSED}s — check $SIM_NAME/run.log"
        tail -10 $PL/$SIM_NAME/run.log >> $MAIN_LOG 2>/dev/null || true
    else
        log "  Done exit=$EXIT elapsed=${ELAPSED}s"
        # Analytics
        $ANALYZE "$SIM_NAME" "$LABEL" >> $MAIN_LOG 2>&1 || log "  Analytics warning for $SIM_NAME"
    fi
}

# ── SIMS ──────────────────────────────────────────────────────────────────────
run_sim PL1 gbm_pl1_tumor.xml   baseline_tumor_growth  "PL-1 Untreated Power Law"
run_sim PL2 gbm_pl2_cart.xml    baseline_cart          "PL-2 Baseline GD2 500 CAR-T"
run_sim A   gbm_plA_hif.xml     hif_inhibition         "Sim A HIF-1a Inhibition"
run_sim B   gbm_plB_o2boost.xml o2_enhancement         "Sim B O2 Enhancement"
run_sim C   gbm_plC_hypact.xml  hypoxia_activated_cart "Sim C Hypoxia-Activated CAR-T"
run_sim D   gbm_plD_metro.xml   metronomic_cart        "Sim D Metronomic Infusion"
run_sim E   gbm_plE_dual.xml    dual_target_powerlaw   "Sim E Dual GD2+CAIX Power Law"

log ""
log "=== All power law sims done. Running comparative analysis... ==="
python3 /home/rshef/gbm_simulations/powerlaw_comparative.py >> $MAIN_LOG 2>&1 \
    && log "Comparative figures done." \
    || log "WARNING: comparative figures had errors — check log"

log "=== run_powerlaw_all.sh complete ==="
