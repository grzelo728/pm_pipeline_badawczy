"""
Globalna konfiguracja projektu LLM Compression.
Praca magisterska: wpływ kolejności metod kompresji na jakość modelu.
Środowisko: RunPod NVIDIA L40S (48GB VRAM)
"""

import os

# ─────────────────────────────────────────────
# ŚCIEŻKI
# ─────────────────────────────────────────────
# Katalog projektu = lokalizacja tego pliku (przenośne, niezależne od maszyny).
# Można nadpisać zmienną środowiskową LLMC_BASE_DIR.
BASE_DIR     = os.environ.get("LLMC_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = f"{BASE_DIR}/models"
RESULTS_DIR  = f"{BASE_DIR}/results"
LOGS_DIR     = f"{BASE_DIR}/logs"
# Zewnętrzne repozytoria narzędzi (klonowane przez 01_setup.sh) — nadpisywalne przez env.
WANDA_DIR    = os.environ.get("WANDA_DIR", os.path.join(os.path.dirname(BASE_DIR), "wanda"))
LMEVAL_DIR   = os.environ.get("LMEVAL_DIR", os.path.join(os.path.dirname(BASE_DIR), "lm-eval"))

# ─────────────────────────────────────────────
# MODELE (student → teacher)
# ─────────────────────────────────────────────
MODELS = {
    "qwen": {
        "student": "Qwen/Qwen2.5-7B",
        "teacher": "Qwen/Qwen2.5-14B",
        # Fallback gdy 14B za duży:
        # "teacher": "Qwen/Qwen2.5-7B-Instruct",
    },
    "llama": {
        "student": "meta-llama/Llama-3.2-3B",
        "teacher": "meta-llama/Llama-3.1-8B",
    },
    "phi": {
        "student": "microsoft/Phi-3.5-mini-instruct",
        "teacher": "microsoft/Phi-3-medium-4k-instruct",
        # Fallback gdy 14B za ciasny:
        # "teacher": "microsoft/Phi-3-small-8k-instruct",
    },
}

# ─────────────────────────────────────────────
# 6 SCENARIUSZY (permutacje P, Q, KD)
# ─────────────────────────────────────────────
# P  = Pruning (Wanda, sparsity=0.5)
# Q  = Quantization (GPTQ INT4)
# KD = Knowledge Distillation (logit-based)

SCENARIOS = {
    "S1": ["P", "Q", "KD"],
    "S2": ["Q", "P", "KD"],
    "S3": ["P", "KD", "Q"],   # ← pilot
    "S4": ["Q", "KD", "P"],
    "S5": ["KD", "P", "Q"],
    "S6": ["KD", "Q", "P"],
}

# ─────────────────────────────────────────────
# HIPERPARAMETRY
# ─────────────────────────────────────────────

# Pruning
PRUNING_SPARSITY     = 0.5
PRUNING_NSAMPLES     = 128
PRUNING_TYPE         = "unstructured"
PRUNING_METHOD       = "wanda"

# Quantization
QUANT_BITS           = 4
QUANT_GROUP_SIZE     = 128
QUANT_NSAMPLES       = 128
QUANT_DESC_ACT       = False

# Knowledge Distillation
KD_TEMPERATURE       = 2.0
KD_ALPHA             = 0.5      # waga KL vs CE: alpha*KL + (1-alpha)*CE
KD_LR                = 2e-5
KD_EPOCHS            = 1
KD_SAMPLES           = 1000
KD_BATCH_SIZE        = 2
KD_GRAD_ACCUM        = 8        # efektywny batch = 2 * 8 = 16
KD_MAX_SEQ_LEN       = 512

# ─────────────────────────────────────────────
# EWALUACJA
# ─────────────────────────────────────────────
EVAL_TASKS = [
    "gpqa",
    "arc_challenge",
    "gsm8k",
    "winogrande",
    "truthfulqa_mc2",
    "hellaswag",
    "mmlu_pro",
    "ifeval",
]
EVAL_BATCH_SIZE = 8
EVAL_SEED       = 42

# Stała liczba próbek na benchmark (seed=42 → te same próbki za każdym razem)
# None = cały zbiór
EVAL_LIMITS = {
    "gpqa":           None,   # 448 pytań — bierzemy wszystkie
    "arc_challenge":  500,
    "gsm8k":          500,
    "winogrande":     500,
    "truthfulqa_mc2": 500,
    "hellaswag":      500,
    "mmlu_pro":       500,
    "ifeval":         None,   # 541 pytań — bierzemy wszystkie
}

# ─────────────────────────────────────────────
# TOKENY / CREDENTIALE
# ─────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # ustaw: export HF_TOKEN=hf_...

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def model_safe_name(hub_name: str) -> str:
    """'Qwen/Qwen2.5-7B' → 'Qwen_Qwen2.5-7B'"""
    return hub_name.replace("/", "_")


def model_local_path(hub_name: str) -> str:
    return f"{MODELS_DIR}/{model_safe_name(hub_name)}"


def result_path(model_family: str, scenario: str, step: str) -> str:
    return f"{RESULTS_DIR}/{model_family}_{scenario}_{step}"
