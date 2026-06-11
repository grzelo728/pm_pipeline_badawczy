#!/bin/bash
# Instalacja środowiska na świeżym Podzie RunPod (PyTorch 2.8.0 template)
# Uruchom JAKO PIERWSZE po stworzeniu Poda.
# Użycie: bash 01_setup.sh [HF_TOKEN]

set -e

HF_TOKEN="${1:-${HF_TOKEN:-}}"

echo "======================================="
echo "  LLM Compression — Setup środowiska"
echo "======================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'brak nvidia-smi')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo '?')"
echo ""

# ── Katalogi ────────────────────────────────
mkdir -p /workspace/llm_compression/{scripts,models,results,logs,scenarios}

# ── Biblioteki Python ────────────────────────
echo ">>> Instalacja bibliotek Python..."
# torch musi być pierwszy — auto-gptq buduje rozszerzenie CUDA i wymaga torch
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.46.3 accelerate datasets tokenizers sentencepiece protobuf
pip install -q gptqmodel optimum
pip install -q torch-pruning
echo "✓ biblioteki Python"

# ── Wanda (pruning) ─────────────────────────
if [ ! -d "/workspace/wanda" ]; then
    echo ">>> Klonowanie Wanda..."
    git clone https://github.com/locuslab/wanda.git /workspace/wanda
    echo "✓ Wanda"
else
    echo "✓ Wanda już istnieje"
fi

# ── lm-evaluation-harness ───────────────────
if [ ! -d "/workspace/lm-eval" ]; then
    echo ">>> Klonowanie lm-evaluation-harness..."
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness /workspace/lm-eval
    cd /workspace/lm-eval
    pip install -q -e ".[math,ifeval,sentencepiece]"
    echo "✓ lm-eval"
else
    echo "✓ lm-eval już istnieje"
fi

# ── HuggingFace login ────────────────────────
if [ -n "$HF_TOKEN" ]; then
    echo ">>> Logowanie do HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
    echo "✓ HF login"
else
    echo "⚠ HF_TOKEN nie podany — Llama (gated) nie będzie dostępna!"
    echo "  Uruchom: huggingface-cli login --token hf_TWOJ_TOKEN"
fi

# ── Weryfikacja ──────────────────────────────
echo ""
echo ">>> Weryfikacja środowiska..."
python - <<'EOF'
import torch, transformers, gptqmodel, datasets
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA dostępny: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU          : {p.name}")
    print(f"VRAM         : {p.total_memory/1e9:.1f} GB")
print(f"Transformers : {transformers.__version__}")
print(f"gptqmodel    : {gptqmodel.__version__}")
print(f"datasets     : {datasets.__version__}")
EOF

echo ""
echo "✅ Setup zakończony! Następny krok: python scripts/02_download.py"
