"""
Kwantyzacja GPTQ do INT4.

Użycie:
    python 04_quantize.py --input models/qwen_pruned --output models/qwen_pruned_quant
    python 04_quantize.py --input models/qwen_pruned --output models/qwen_pruned_quant --bits 8

UWAGA: Po kwantyzacji GPTQ wagi są zamrożone (INT4 nie ma gradientów).
       KD musi być wykonane PRZED kwantyzacją (scenariusze S3, S5, S6)
       lub student musi być załadowany z FP16 checkpointu (S1, S2, S4).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QUANT_BITS, QUANT_GROUP_SIZE, QUANT_NSAMPLES, QUANT_DESC_ACT

from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig


def get_calibration_data(tokenizer, nsamples: int, max_length: int = 512) -> list[dict]:
    """Pobiera dane kalibracyjne z C4 (allenai/c4)."""
    print(f"   Pobieranie {nsamples} próbek kalibracyjnych z C4...")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []

    for i, sample in enumerate(dataset):
        if i >= nsamples:
            break
        enc = tokenizer(
            sample["text"],
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        samples.append({
            "input_ids": enc.input_ids[0],
            "attention_mask": enc.attention_mask[0],
        })

    print(f"   Zebrano {len(samples)} próbek.")
    return samples


def run_quantization(
    input_path: str,
    output_path: str,
    bits: int = QUANT_BITS,
    group_size: int = QUANT_GROUP_SIZE,
    nsamples: int = QUANT_NSAMPLES,
    desc_act: bool = QUANT_DESC_ACT,
) -> None:
    print(f"{'='*50}")
    print(f"KWANTYZACJA GPTQ (INT{bits})")
    print(f"  Wejście   : {input_path}")
    print(f"  Wyjście   : {output_path}")
    print(f"  Bity      : {bits}")
    print(f"  Group size: {group_size}")
    print(f"  Samples   : {nsamples}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantize_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
    )

    print(">>> Ładowanie modelu do kwantyzacji...")
    model = GPTQModel.from_pretrained(
        input_path,
        quantize_config=quantize_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    calib_data = get_calibration_data(tokenizer, nsamples=nsamples)

    print(f">>> Kwantyzuję... (może zająć 30-60 min dla 7B, ~20 min dla 3B)")
    model.quantize(calib_data)

    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    del model
    torch.cuda.empty_cache()

    print(f"\n✅ Kwantyzacja zakończona → {output_path}")
    print("   Uwaga: ten model ma zamrożone wagi (INT4). KD po tej operacji NIE zadziała.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kwantyzacja GPTQ")
    parser.add_argument("--input", required=True, help="Ścieżka do modelu wejściowego")
    parser.add_argument("--output", required=True, help="Ścieżka do modelu wyjściowego")
    parser.add_argument("--bits", type=int, default=QUANT_BITS, choices=[2, 3, 4, 8], help="Liczba bitów kwantyzacji")
    parser.add_argument("--group-size", type=int, default=QUANT_GROUP_SIZE)
    parser.add_argument("--nsamples", type=int, default=QUANT_NSAMPLES)
    args = parser.parse_args()

    run_quantization(
        input_path=args.input,
        output_path=args.output,
        bits=args.bits,
        group_size=args.group_size,
        nsamples=args.nsamples,
    )
