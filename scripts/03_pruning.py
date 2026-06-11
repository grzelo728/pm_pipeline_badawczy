"""
Pruning modelu — magnitude pruning (unstructured, sparsity=0.5).

Użycie:
    python 03_pruning.py --input models/Qwen_Qwen2.5-7B --output models/qwen_pruned
    python 03_pruning.py --input models/Qwen_Qwen2.5-7B --output models/qwen_pruned --sparsity 0.3

Metoda: magnitude pruning — zeruje wagi o najmniejszej wartości bezwzględnej.
Działa na każdej architekturze (Qwen, Llama, Phi, itp.).
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PRUNING_SPARSITY


def run_pruning(
    input_path: str,
    output_path: str,
    sparsity: float = PRUNING_SPARSITY,
) -> None:
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())

    print(f"{'='*50}")
    print(f"PRUNING: MAGNITUDE (unstructured)")
    print(f"  Wejście : {input_path}")
    print(f"  Wyjście : {output_path}")
    print(f"  Sparsity: {sparsity} ({int(sparsity*100)}% wag zerowanych)")
    print(f"{'='*50}")

    print(">>> Ładowanie modelu na CPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        input_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(">>> Model załadowany.", flush=True)

    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    total_layers = len(layers)
    print(f">>> Znaleziono {total_layers} warstw Linear do pruningu.", flush=True)
    print(f"{'─'*50}", flush=True)

    total, pruned = 0, 0
    for i, (name, module) in enumerate(layers):
        w = module.weight.data
        flat = w.abs().float().flatten()
        k = max(1, int(sparsity * flat.numel()))
        threshold = flat.kthvalue(k).values
        mask = w.abs() < threshold
        module.weight.data[mask] = 0
        layer_pruned = mask.sum().item()
        layer_total = w.numel()
        pruned += layer_pruned
        total += layer_total

        # Log co 10 warstw i na końcu
        if i % 10 == 0 or i == total_layers - 1:
            print(
                f"  [{i+1:3d}/{total_layers}] {name:<50s} "
                f"sparsity={layer_pruned/layer_total:.1%}",
                flush=True
            )

    print(f"{'─'*50}", flush=True)
    print(f">>> Sparsity globalna: {pruned/total:.2%} ({pruned:,}/{total:,} wag wyzerowanych)", flush=True)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(">>> Zapisywanie modelu...", flush=True)
    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"\n✅ Pruning zakończony → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Magnitude Pruning")
    parser.add_argument("--input", required=True, help="Ścieżka do modelu wejściowego")
    parser.add_argument("--output", required=True, help="Ścieżka do modelu wyjściowego")
    parser.add_argument("--sparsity", type=float, default=PRUNING_SPARSITY, help="Współczynnik rzadkości (0-1)")
    args = parser.parse_args()

    run_pruning(
        input_path=args.input,
        output_path=args.output,
        sparsity=args.sparsity,
    )
