"""
Uruchamia dowolny scenariusz S1-S6 dla wybranej rodziny modeli.

Scenariusze (permutacje P, Q, KD):
    S1: P  → Q  → KD
    S2: Q  → P  → KD
    S3: P  → KD → Q    ← zalecany pilot
    S4: Q  → KD → P
    S5: KD → P  → Q
    S6: KD → Q  → P

Optymalizacja — współdzielone checkpointy pierwszego kroku:
    Każda metoda pojawia się jako pierwsza dokładnie 2 razy:
      P pierwszy : S1, S3  → {family}_step1_P  (liczony raz, reużywany)
      Q pierwszy : S2, S4  → {family}_step1_Q
      KD pierwszy: S5, S6  → {family}_step1_KD
    Jeśli checkpoint już istnieje na dysku, krok jest pomijany (~3h oszczędności).

Użycie:
    python run_scenario.py --scenario S3 --family qwen
    python run_scenario.py --scenario S5 --family llama --skip-eval
    python run_scenario.py --scenario S1 --family phi --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BASE_DIR, MODELS_DIR, RESULTS_DIR,
    MODELS, SCENARIOS,
    model_safe_name, model_local_path,
)


def checkpoint_exists(path: str) -> bool:
    """Sprawdza czy checkpoint modelu już istnieje (ma config.json)."""
    return (Path(path) / "config.json").exists()


def run_cmd(cmd: str, dry_run: bool = False) -> None:
    print(f"\n$ {cmd}\n")
    if dry_run:
        return
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"Polecenie zakończone błędem (kod {result.returncode}): {cmd}")
    print(f"   (czas: {elapsed/60:.1f} min)")


def evaluate(model_path: str, step_name: str, dry_run: bool = False) -> None:
    # Pomiń ewaluację jeśli wyniki już istnieją
    result_dir = Path(RESULTS_DIR) / step_name
    if result_dir.exists() and any(result_dir.rglob("*.json")):
        print(f"   ⏭  Ewaluacja już istnieje: {step_name} — pomijam")
        return
    run_cmd(f"bash {BASE_DIR}/scripts/06_evaluate.sh {model_path} {step_name}", dry_run)


def do_pruning(input_path: str, output_path: str, dry_run: bool = False) -> str:
    if checkpoint_exists(output_path):
        print(f"   ⏭  Checkpoint już istnieje: {output_path} — pomijam pruning")
        return output_path
    run_cmd(
        f"python {BASE_DIR}/scripts/03_pruning.py"
        f" --input {input_path}"
        f" --output {output_path}",
        dry_run,
    )
    return output_path


def do_quantize(input_path: str, output_path: str, dry_run: bool = False) -> str:
    if checkpoint_exists(output_path):
        print(f"   ⏭  Checkpoint już istnieje: {output_path} — pomijam kwantyzację")
        return output_path
    run_cmd(
        f"python {BASE_DIR}/scripts/04_quantize.py"
        f" --input {input_path}"
        f" --output {output_path}",
        dry_run,
    )
    return output_path


def do_kd(teacher_path: str, student_path: str, output_path: str, dry_run: bool = False) -> str:
    if checkpoint_exists(output_path):
        print(f"   ⏭  Checkpoint już istnieje: {output_path} — pomijam KD")
        return output_path
    run_cmd(
        f"python {BASE_DIR}/scripts/05_kd.py"
        f" --teacher {teacher_path}"
        f" --student {student_path}"
        f" --output {output_path}",
        dry_run,
    )
    return output_path


def _do_kd_step(
    current_path: str,
    next_path: str,
    teacher_path: str,
    fp16_before_quant_path: str | None,
    need_fp16_before_q: bool,
    dry_run: bool,
) -> tuple[str, str]:
    """Wykonuje krok KD. Zwraca (current_path, step_suffix)."""
    student_for_kd = fp16_before_quant_path if (need_fp16_before_q and fp16_before_quant_path) else current_path
    if student_for_kd != current_path:
        print(f"   ℹ KD używa FP16 studenta: {student_for_kd}")

    kd_out = do_kd(teacher_path, student_for_kd, next_path, dry_run)

    if need_fp16_before_q:
        print("   ℹ Ponowna kwantyzacja po KD...")
        return do_quantize(kd_out, f"{next_path}_requant", dry_run), "kd_requant"
    return kd_out, "kd"


def _execute_step(
    step: str,
    step_index: int,
    current_path: str,
    next_path: str,
    teacher_path: str,
    fp16_before_quant_path: str | None,
    need_fp16_before_q: bool,
    q_idx: int,
    family: str,
    scenario_id: str,
    dry_run: bool,
) -> tuple[str, str | None, str]:
    """Wykonuje jeden krok kompresji (P/Q/KD). Zwraca (current_path, fp16_path, step_name)."""
    step_label = step.lower()

    if step == "P":
        if need_fp16_before_q and q_idx > step_index:
            fp16_before_quant_path = current_path
        current_path = do_pruning(current_path, next_path, dry_run)

    elif step == "Q":
        if need_fp16_before_q:
            fp16_before_quant_path = current_path
            print(f"   ℹ FP16 checkpoint dla KD: {fp16_before_quant_path}")
        current_path = do_quantize(current_path, next_path, dry_run)

    elif step == "KD":
        current_path, step_label = _do_kd_step(
            current_path, next_path, teacher_path,
            fp16_before_quant_path, need_fp16_before_q, dry_run,
        )

    step_name = f"{family}_{scenario_id}_step{step_index+1}_{step_label}"
    return current_path, fp16_before_quant_path, step_name


def run_scenario(
    scenario_id: str,
    family: str,
    skip_eval: bool = False,
    dry_run: bool = False,
) -> None:
    steps = SCENARIOS[scenario_id]
    cfg = MODELS[family]
    student_base = model_local_path(cfg["student"])
    teacher_base = model_local_path(cfg["teacher"])

    # Współdzielone checkpointy pierwszego kroku (reużywane między scenariuszami)
    # Ścieżka niezależna od scenariusza — np. models/qwen_shared_step1_P
    shared_step1 = {
        "P":  f"{MODELS_DIR}/{family}_shared_step1_P",
        "Q":  f"{MODELS_DIR}/{family}_shared_step1_Q",
        "KD": f"{MODELS_DIR}/{family}_shared_step1_KD",
    }

    # Pozostałe kroki są unikalne dla scenariusza
    scenario_prefix = f"{MODELS_DIR}/{family}_{scenario_id}"

    print(f"\n{'#'*60}")
    print(f"# SCENARIUSZ {scenario_id}: {' → '.join(steps)}")
    print(f"# Rodzina  : {family}")
    print(f"# Student  : {cfg['student']}")
    print(f"# Teacher  : {cfg['teacher']}")
    print(f"{'#'*60}\n")

    total_start = time.time()

    # ── Baseline ──────────────────────────────────────────
    print(">>> KROK 0: Baseline")
    if not skip_eval:
        evaluate(student_base, f"{family}_baseline", dry_run)

    # KD po GPTQ wymaga FP16 checkpointu przed kwantyzacją
    q_idx  = steps.index("Q")  if "Q"  in steps else -1
    kd_idx = steps.index("KD") if "KD" in steps else -1
    need_fp16_before_q = (kd_idx > q_idx and q_idx >= 0)

    current_path = student_base
    fp16_before_quant_path = None

    for i, step in enumerate(steps):
        next_path  = shared_step1[step] if i == 0 else f"{scenario_prefix}_step{i+1}_{step.lower()}"
        label      = "[współdzielony]" if i == 0 else ""
        print(f"\n>>> KROK {i+1}/{len(steps)}: {step} {label}")

        current_path, fp16_before_quant_path, step_name = _execute_step(
            step=step,
            step_index=i,
            current_path=current_path,
            next_path=next_path,
            teacher_path=teacher_base,
            fp16_before_quant_path=fp16_before_quant_path,
            need_fp16_before_q=need_fp16_before_q,
            q_idx=q_idx,
            family=family,
            scenario_id=scenario_id,
            dry_run=dry_run,
        )

        if not skip_eval:
            evaluate(current_path, step_name, dry_run)

    # ── Zbierz wyniki ──────────────────────────────────────
    print("\n>>> Zbieranie wyników...")
    run_cmd(
        f"python {BASE_DIR}/scripts/07_collect.py"
        f" --filter {family}_{scenario_id}"
        f" --print",
        dry_run,
    )

    total_min = (time.time() - total_start) / 60
    print(f"\n✅ Scenariusz {scenario_id} ({family}) zakończony w {total_min:.0f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchamia scenariusz kompresji S1-S6")
    parser.add_argument("--scenario",  required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument("--family",    required=True, choices=list(MODELS.keys()))
    parser.add_argument("--skip-eval", action="store_true", help="Pomiń ewaluację między krokami")
    parser.add_argument("--dry-run",   action="store_true", help="Wypisz komendy bez uruchamiania")
    args = parser.parse_args()

    run_scenario(
        scenario_id=args.scenario,
        family=args.family,
        skip_eval=args.skip_eval,
        dry_run=args.dry_run,
    )
