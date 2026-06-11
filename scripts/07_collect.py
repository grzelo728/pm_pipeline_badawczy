"""
Zbieranie wyników ewaluacji do CSV i JSON.

Skanuje katalog results/ w poszukiwaniu plików JSON z lm-eval,
parsuje wyniki i agreguje je do:
  - results/summary.csv    ← tabela do Excela/pandas
  - results/summary.json   ← pełne dane

Użycie:
    python 07_collect.py                    # zbierz wszystkie wyniki
    python 07_collect.py --filter qwen      # tylko wyniki dla Qwen
    python 07_collect.py --print            # wypisz tabelę na terminal
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, EVAL_TASKS

# Kluczowe metryki dla każdego taska
TASK_METRIC = {
    "gpqa":            "acc,none",
    "arc_challenge":   "acc_norm,none",
    "gsm8k":           "exact_match,strict-match",
    "winogrande":      "acc,none",
    "truthfulqa_mc2":  "acc,none",
    "hellaswag":       "acc_norm,none",
    "mmlu_pro":        "acc,none",
    "ifeval":          "prompt_level_strict_acc,none",
}


def parse_lmeval_result(result_dir: Path) -> dict | None:
    """Parsuje wynik lm-eval z katalogu (szuka pliku results_*.json)."""
    json_files = list(result_dir.glob("results_*.json"))
    if not json_files:
        # Starszy format — plik results.json bezpośrednio
        json_files = list(result_dir.glob("*.json"))

    if not json_files:
        return None

    result_file = sorted(json_files)[-1]  # najnowszy

    try:
        with open(result_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    results = data.get("results", {})
    parsed = {"_file": str(result_file)}

    for task_key, metric_key in TASK_METRIC.items():
        # lm-eval może przechowywać task pod różnymi nazwami
        task_data = results.get(task_key) or results.get(f"{task_key}_0shot") or {}
        value = task_data.get(metric_key)
        if value is None:
            # Spróbuj pierwszej dostępnej metryki
            value = next(iter(task_data.values()), None)
        parsed[task_key] = round(value * 100, 2) if value is not None else None

    return parsed


def parse_step_name(name: str) -> dict:
    """
    Parsuje nazwę katalogu wyników, np.:
      'qwen_S3_step1_pruned' → {family: 'qwen', scenario: 'S3', step: 'step1_pruned'}
    """
    parts = name.split("_", 2)
    return {
        "family":   parts[0] if len(parts) > 0 else name,
        "scenario": parts[1] if len(parts) > 1 else "",
        "step":     parts[2] if len(parts) > 2 else "",
        "run_name": name,
    }


def collect_results(results_dir: str, filter_str: str = "") -> list[dict]:
    """Zbiera wszystkie wyniki z podkatalogu results/."""
    base = Path(results_dir)
    rows = []

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        if filter_str and filter_str.lower() not in subdir.name.lower():
            continue
        if subdir.name in ("summary",):
            continue

        parsed = parse_lmeval_result(subdir)
        if parsed is None:
            print(f"  ⚠ Brak pliku wyników w: {subdir.name}")
            continue

        meta = parse_step_name(subdir.name)
        row = {**meta, **parsed}
        rows.append(row)
        print(f"  ✓ {subdir.name}")

    return rows


def save_csv(rows: list[dict], output_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    """Drukuje czytelną tabelę wyników na terminal."""
    tasks = list(TASK_METRIC.keys())
    header = f"{'Run':<35} " + " ".join(f"{t[:8]:>8}" for t in tasks)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for row in rows:
        name = row.get("run_name", "?")[:35]
        scores = " ".join(
            f"{row.get(t, '-'):>8.1f}" if row.get(t) is not None else f"{'—':>8}"
            for t in tasks
        )
        print(f"{name:<35} {scores}")

    print("=" * len(header))

    # Średnia po taskach
    avg_row = {}
    for t in tasks:
        vals = [r[t] for r in rows if r.get(t) is not None]
        avg_row[t] = sum(vals) / len(vals) if vals else None

    avg_str = " ".join(
        f"{avg_row[t]:>8.1f}" if avg_row[t] is not None else f"{'—':>8}"
        for t in tasks
    )
    print(f"{'AVG':<35} {avg_str}")


def main():
    parser = argparse.ArgumentParser(description="Zbieranie wyników ewaluacji")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--filter", default="", help="Filtruj po nazwie (np. 'qwen', 'S3')")
    parser.add_argument("--print", action="store_true", dest="print_table", help="Wypisz tabelę")
    args = parser.parse_args()

    print(f"Skanowanie: {args.results_dir}")
    rows = collect_results(args.results_dir, filter_str=args.filter)

    if not rows:
        print("Brak wyników do zebrania.")
        return

    summary_csv  = str(Path(args.results_dir) / "summary.csv")
    summary_json = str(Path(args.results_dir) / "summary.json")

    save_csv(rows, summary_csv)
    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Zapisano {len(rows)} wierszy:")
    print(f"   CSV : {summary_csv}")
    print(f"   JSON: {summary_json}")

    if args.print_table:
        print_table(rows)


if __name__ == "__main__":
    main()
