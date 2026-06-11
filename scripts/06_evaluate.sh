#!/bin/bash
# Ewaluacja modelu benchmarkami lm-evaluation-harness.
# Każdy benchmark ma stałą liczbę próbek (seed=42 → identyczne próbki dla wszystkich modeli).
#
# Użycie:
#   bash 06_evaluate.sh <ścieżka_modelu> <nazwa_kroku>
#   bash 06_evaluate.sh /workspace/llm_compression/models/qwen_pruned qwen_S3_step1_pruned
#
# Opcjonalne zmienne środowiskowe:
#   EVAL_BATCH=4    # zmniejsz przy OOM (domyślnie 8)

set -e

MODEL_PATH="${1:?Podaj ścieżkę modelu jako argument 1}"
STEP_NAME="${2:?Podaj nazwę kroku jako argument 2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${BASE}/results/${STEP_NAME}"
EVAL_BATCH="${EVAL_BATCH:-8}"
EVAL_SEED=42

echo "============================================"
echo "EWALUACJA: ${STEP_NAME}"
echo "  Model : ${MODEL_PATH}"
echo "  Wyniki: ${RESULTS_DIR}"
echo "  Batch : ${EVAL_BATCH}"
echo "============================================"

mkdir -p "${RESULTS_DIR}"

# ── Metadane modelu ──────────────────────────────────────
META_FILE="${RESULTS_DIR}/model_meta.json"
MODEL_SIZE_BYTES=$(du -sb "${MODEL_PATH}" | cut -f1)
MODEL_SIZE_GB=$(du -sh "${MODEL_PATH}" | cut -f1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
EVAL_START=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "${META_FILE}" <<EOF
{
  "step_name": "${STEP_NAME}",
  "model_path": "${MODEL_PATH}",
  "model_size_bytes": ${MODEL_SIZE_BYTES},
  "model_size_gb": "${MODEL_SIZE_GB}",
  "eval_start": "${EVAL_START}",
  "gpu": "${GPU_NAME}",
  "gpu_vram": "${GPU_VRAM}",
  "batch_size": ${EVAL_BATCH},
  "seed": ${EVAL_SEED}
}
EOF
echo "   Rozmiar modelu: ${MODEL_SIZE_GB}"
echo "   GPU: ${GPU_NAME} (${GPU_VRAM})"

# ── Model GPTQ ───────────────────────────────────────────
# transformers 5.9 + gptqmodel wykrywają kwantyzację automatycznie
if [ -f "${MODEL_PATH}/quantize_config.json" ]; then
    echo "   Wykryto model GPTQ (auto-load przez transformers)"
fi

MODEL_ARGS="pretrained=${MODEL_PATH},trust_remote_code=True"

# ── Funkcja uruchamiająca pojedynczy benchmark ───────────
run_task() {
    local TASK=$1
    local LIMIT=$2
    local LIMIT_ARG=""
    [ -n "${LIMIT}" ] && LIMIT_ARG="--limit ${LIMIT}"

    echo ""
    echo ">>> ${TASK} (próbki: ${LIMIT:-wszystkie})"
    TASK_START=$(date +%s)

    python -m lm_eval \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASK}" \
        --device cuda:0 \
        --batch_size "${EVAL_BATCH}" \
        --seed "${EVAL_SEED}" \
        --output_path "${RESULTS_DIR}/${TASK}" \
        --log_samples \
        ${LIMIT_ARG}

    TASK_END=$(date +%s)
    TASK_SECS=$((TASK_END - TASK_START))
    echo "   Czas: ${TASK_SECS}s"
    echo "{\"task\": \"${TASK}\", \"elapsed_seconds\": ${TASK_SECS}}" \
        >> "${RESULTS_DIR}/timing.jsonl"
}

# ── Benchmarki ───────────────────────────────────────────
# run_task "gpqa"            gated
run_task "arc_challenge"    500
run_task "gsm8k"            50   # generatywny — limit 50 próbek (~15 min na model)
run_task "winogrande"       500
run_task "truthfulqa_mc2"   500
run_task "hellaswag"        500
# run_task "mmlu_pro"         50   # CoT, 14 subtasków×50=700 req, 2h + wyniki 0% przy małym limicie
run_task "ifeval"           "541"  # wszystkie pytania

# ── Podsumowanie czasowe ─────────────────────────────────
EVAL_END=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"eval_end\": \"${EVAL_END}\"}" >> "${RESULTS_DIR}/timing.jsonl"

echo ""
echo "✅ Ewaluacja zakończona → ${RESULTS_DIR}"
echo "   Metadane: ${META_FILE}"
echo "   Czasy: ${RESULTS_DIR}/timing.jsonl"
echo "   Następny krok: python scripts/07_collect.py --filter ${STEP_NAME}"
