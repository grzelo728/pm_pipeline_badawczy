"""
Pobieranie modeli bazowych (studentów i teacherów) z HuggingFace.
Modele zapisywane są w FP16 na CPU (NIE GGUF).

Użycie:
    python 02_download.py                    # pobierz wszystkie modele
    python 02_download.py --family llama     # tylko modele Llama
    python 02_download.py --teachers         # tylko teacherzy (duże modele)
    python 02_download.py --dry-run          # wypisz listę bez pobierania
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS, MODELS_DIR, model_safe_name

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(hub_name: str, force: bool = False) -> str:
    safe = model_safe_name(hub_name)
    save_path = f"{MODELS_DIR}/{safe}"

    if os.path.exists(save_path) and not force:
        print(f"⏭  {hub_name} — już pobrane ({save_path})")
        return save_path

    print(f"⬇  Pobieranie: {hub_name}")
    print(f"   Cel: {save_path}")

    tokenizer = AutoTokenizer.from_pretrained(hub_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hub_name,
        torch_dtype=torch.float16,
        device_map="cpu",          # na CPU — oszczędzamy VRAM
        trust_remote_code=True,
    )

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    del model

    size_gb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(save_path)
        for f in files
    ) / 1e9

    print(f"✅ {hub_name} → {save_path} ({size_gb:.1f} GB)")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Pobieranie modeli LLM")
    parser.add_argument("--family", choices=list(MODELS.keys()), help="Pobierz tylko wybraną rodzinę")
    parser.add_argument("--teachers", action="store_true", help="Pobierz tylko teacherów")
    parser.add_argument("--students", action="store_true", help="Pobierz tylko studentów")
    parser.add_argument("--force", action="store_true", help="Pobierz ponownie nawet jeśli istnieje")
    parser.add_argument("--dry-run", action="store_true", help="Wypisz listę bez pobierania")
    args = parser.parse_args()

    families = [args.family] if args.family else list(MODELS.keys())

    to_download = []
    for fam in families:
        cfg = MODELS[fam]
        if not args.teachers:
            to_download.append(("student", fam, cfg["student"]))
        if not args.students:
            to_download.append(("teacher", fam, cfg["teacher"]))

    print(f"Modele do pobrania: {len(to_download)}")
    for role, fam, name in to_download:
        print(f"  [{fam}/{role}] {name}")

    if args.dry_run:
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    for role, fam, name in to_download:
        try:
            download_model(name, force=args.force)
        except Exception as e:
            print(f"❌ Błąd podczas pobierania {name}: {e}")
            print("   Kontynuuję z pozostałymi modelami...")

    print("\n✅ Pobieranie zakończone.")
    print(f"   Modele w: {MODELS_DIR}")


if __name__ == "__main__":
    main()
