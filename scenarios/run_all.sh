#!/bin/bash
# Uruchamia wszystkie scenariusze (S1-S6) dla wybranych rodzin modeli.
#
# Użycie:
#   bash run_all.sh                         # wszystkie rodziny, wszystkie scenariusze
#   bash run_all.sh llama                   # tylko Llama
#   bash run_all.sh qwen S3 S5              # Qwen, tylko S3 i S5
#   SKIP_EVAL=1 bash run_all.sh             # bez ewaluacji między krokami (szybsze)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="${BASE}/logs"
mkdir -p "${LOGS_DIR}"

# ── Argumenty ────────────────────────────────
FAMILY_FILTER="${1:-all}"      # all | qwen | llama | phi
shift || true
SCENARIO_FILTER="${@:-S1 S2 S3 S4 S5 S6}"

FAMILIES=("qwen" "llama" "phi")
SCENARIOS=("S1" "S2" "S3" "S4" "S5" "S6")

SKIP_EVAL_ARG=""
[ "${SKIP_EVAL:-0}" = "1" ] && SKIP_EVAL_ARG="--skip-eval"

# ── Wybór rodzin ──────────────────────────────
if [ "${FAMILY_FILTER}" != "all" ]; then
    FAMILIES=("${FAMILY_FILTER}")
fi

# ── Wybór scenariuszy ─────────────────────────
if [ -n "${SCENARIO_FILTER}" ]; then
    SCENARIOS=(${SCENARIO_FILTER})
fi

TOTAL=$(( ${#FAMILIES[@]} * ${#SCENARIOS[@]} ))
DONE=0
START_TIME=$(date +%s)

echo "======================================================"
echo "  LLM Compression — Uruchamianie wszystkich scenariuszy"
echo "======================================================"
echo "  Rodziny    : ${FAMILIES[*]}"
echo "  Scenariusze: ${SCENARIOS[*]}"
echo "  Łącznie    : ${TOTAL} przebiegów"
echo "  Logowanie  : ${LOGS_DIR}/"
echo "======================================================"
echo ""

for family in "${FAMILIES[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        DONE=$(( DONE + 1 ))
        LOG_FILE="${LOGS_DIR}/${family}_${scenario}.log"

        ELAPSED=$(( $(date +%s) - START_TIME ))
        AVG=$(( ELAPSED / DONE ))
        ETA=$(( AVG * (TOTAL - DONE) ))

        echo ""
        echo "──────────────────────────────────────────────────"
        echo "  [${DONE}/${TOTAL}] ${family} / ${scenario}"
        echo "  ETA: ~$(( ETA / 3600 ))h$(( (ETA % 3600) / 60 ))m"
        echo "  Log: ${LOG_FILE}"
        echo "──────────────────────────────────────────────────"

        python "${BASE}/scenarios/run_scenario.py" \
            --scenario "${scenario}" \
            --family "${family}" \
            ${SKIP_EVAL_ARG} \
            2>&1 | tee "${LOG_FILE}"

        echo "✅ ${family} / ${scenario} zakończony"
    done
done

# ── Zbierz wszystkie wyniki ───────────────────
echo ""
echo ">>> Zbieranie wszystkich wyników do summary..."
python "${BASE}/scripts/07_collect.py" --print

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
echo ""
echo "======================================================"
echo "  ✅ WSZYSTKIE SCENARIUSZE ZAKOŃCZONE"
echo "  Łączny czas: $(( TOTAL_TIME / 3600 ))h$(( (TOTAL_TIME % 3600) / 60 ))m"
echo "  Wyniki: ${BASE}/results/summary.csv"
echo "======================================================"
