# LLM Compression Pipeline

Pipeline do badania wpływu **kolejności** metod kompresji dużych modeli językowych
na jakość modelu finalnego:

- **P** — pruning (magnitude, unstructured, 50% sparsity)
- **Q** — kwantyzacja (GPTQ INT4, group_size=128)
- **KD** — destylacja wiedzy (logit-based, adaptery LoRA)

Scenariusze kolejności: **S3** (P→KD→Q), **S5** (KD→P→Q), **S6** (KD→Q→P).
Modele: Qwen2.5-7B, Llama-3.2-3B, Phi-3.5-mini.

## Struktura

```
config.py                  parametry: modele, hiperparametry, ścieżki
requirements.txt           zależności (pip)
scripts/
  01_setup.sh              instalacja środowiska
  02_download.py           pobranie modeli z HuggingFace
  03_pruning.py            magnitude pruning
  03b_prune_gptq.py        pruning po kwantyzacji (dekwantyzacja, scenariusz S6)
  04_quantize.py           kwantyzacja GPTQ INT4
  05_kd.py                 destylacja wiedzy (LoRA)
  06_evaluate.sh           ewaluacja (lm-evaluation-harness)
  07_collect.py            agregacja wyników do CSV/JSON
scenarios/
  run_scenario.py          orkiestracja pojedynczego scenariusza (S1-S6)
  run_all.sh               wszystkie rodzine i scenariusze
  pilot_qwen_s3.sh         przykład: S3 dla Qwen
```

## Uruchomienie (maszyna z GPU)

```bash
bash scripts/01_setup.sh
python scenarios/run_scenario.py --scenario S3 --family qwen
python scripts/07_collect.py --filter qwen_S3
```

## Ewaluacja

Sześć benchmarków: ARC-Challenge, GSM8K, WinoGrande, TruthfulQA, HellaSwag, IFEval.
Dekodowanie deterministyczne (temperatura 0, seed 42).

## Środowisko

PyTorch + HuggingFace Transformers, `gptqmodel`, `peft`, `lm-evaluation-harness`.
Pojedyncze GPU (NVIDIA A100 / L40S, 48 GB VRAM).
