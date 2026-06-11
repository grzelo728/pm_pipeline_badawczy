#!/bin/bash
# PILOT: Qwen2.5-7B, Scenariusz S3 (Pruning → KD → Kwantyzacja)
# To jest PIERWSZY eksperyment do uruchomienia — waliduje cały pipeline.
#
# Szacowany czas na L40S: ~5-7h, koszt: ~$3.50-5.00
#
# Użycie:
#   bash pilot_qwen_s3.sh
#   bash pilot_qwen_s3.sh --skip-eval      # bez ewaluacji między krokami (~2h)

set -e

SKIP_EVAL=0
[ "${1}" = "--skip-eval" ] && SKIP_EVAL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$SCRIPT_DIR")"
STUDENT="${BASE}/models/Qwen_Qwen2.5-7B"
# Teacher: idealnie Qwen2.5-14B, ale jako fallback używamy samego studenta (self-distillation)
# Self-distillation jest uproszczeniem, ale waliduje pipeline KD.
TEACHER="${BASE}/models/Qwen_Qwen2.5-7B"
# Jeśli pobrałeś Qwen2.5-14B, odkomentuj poniższe:
# TEACHER="${BASE}/models/Qwen_Qwen2.5-14B"

RUN_NAME="pilot_qwen_s3"
MODELS="${BASE}/models/${RUN_NAME}"

echo "======================================================="
echo "  PILOT: Qwen2.5-7B — Scenariusz S3 (P → KD → Q)"
echo "======================================================="
echo "  Student : ${STUDENT}"
echo "  Teacher : ${TEACHER}"
echo "  Prefix  : ${MODELS}"
echo "======================================================="
echo ""

# Sprawdź czy model bazowy istnieje
if [ ! -d "${STUDENT}" ]; then
    echo "❌ Brak modelu bazowego: ${STUDENT}"
    echo "   Uruchom najpierw: python scripts/02_download.py --family qwen --students"
    exit 1
fi

eval_step() {
    local model_path="$1"
    local step_name="$2"
    if [ "${SKIP_EVAL}" = "0" ]; then
        echo ">>> Ewaluacja: ${step_name}"
        bash "${BASE}/scripts/06_evaluate.sh" "${model_path}" "${step_name}"
    else
        echo ">>> [pominięto ewaluację] ${step_name}"
    fi
}

# ── KROK 0: Baseline ─────────────────────────────────────
echo ">>> KROK 0: Ewaluacja baseline"
eval_step "${STUDENT}" "${RUN_NAME}_step0_baseline"

# ── KROK 1: Pruning ───────────────────────────────────────
echo ""
echo ">>> KROK 1: Pruning (Wanda, sparsity=0.5)"
PRUNED="${MODELS}_pruned"
python "${BASE}/scripts/03_pruning.py" \
    --input "${STUDENT}" \
    --output "${PRUNED}"

eval_step "${PRUNED}" "${RUN_NAME}_step1_pruned"

# ── KROK 2: Knowledge Distillation ───────────────────────
echo ""
echo ">>> KROK 2: Knowledge Distillation"
echo "    Teacher: ${TEACHER}"
echo "    Student: ${PRUNED} (FP16, po pruningu)"
KD_OUT="${MODELS}_pruned_kd"
python "${BASE}/scripts/05_kd.py" \
    --teacher "${TEACHER}" \
    --student "${PRUNED}" \
    --output  "${KD_OUT}" \
    --samples 1000 \
    --epochs  1

eval_step "${KD_OUT}" "${RUN_NAME}_step2_kd"

# ── KROK 3: Kwantyzacja GPTQ ──────────────────────────────
echo ""
echo ">>> KROK 3: Kwantyzacja GPTQ (INT4)"
QUANT_OUT="${MODELS}_pruned_kd_quant"
python "${BASE}/scripts/04_quantize.py" \
    --input  "${KD_OUT}" \
    --output "${QUANT_OUT}"

eval_step "${QUANT_OUT}" "${RUN_NAME}_step3_quant"

# ── Podsumowanie wyników ──────────────────────────────────
echo ""
echo ">>> Zbieranie wyników..."
python "${BASE}/scripts/07_collect.py" \
    --filter "${RUN_NAME}" \
    --print

echo ""
echo "======================================================="
echo "  ✅ PILOT S3 ZAKOŃCZONY"
echo ""
echo "  Modele:"
echo "    Baseline : ${STUDENT}"
echo "    Pruned   : ${PRUNED}"
echo "    KD       : ${KD_OUT}"
echo "    Final    : ${QUANT_OUT}"
echo ""
echo "  Wyniki: ${BASE}/results/${RUN_NAME}_*/"
echo "  CSV   : ${BASE}/results/summary.csv"
echo "======================================================="
