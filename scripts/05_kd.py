"""
Logit-based Knowledge Distillation (KD).
Strata = alpha * KL(soft) + (1-alpha) * CE(hard)

Użycie:
    python 05_kd.py \
        --teacher models/meta-llama_Llama-3.1-8B \
        --student models/llama_pruned \
        --output models/llama_pruned_kd

    python 05_kd.py \
        --teacher models/Qwen_Qwen2.5-7B \
        --student models/qwen_pruned \
        --output models/qwen_pruned_kd \
        --samples 2000 --epochs 2

WAŻNE:
  - Teacher i student MUSZĄ mieć ten sam tokenizer (ta sama rodzina modeli).
  - GPTQ modele mają zamrożone wagi — KD na nich NIE zadziała.
    Dla scenariuszy S1, S2, S4 (KD po GPTQ): załaduj FP16 wersję,
    wykonaj KD, a następnie ponownie kwantyzuj.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    KD_TEMPERATURE, KD_ALPHA, KD_LR, KD_EPOCHS,
    KD_SAMPLES, KD_BATCH_SIZE, KD_GRAD_ACCUM, KD_MAX_SEQ_LEN,
    LOGS_DIR,
)


class AlpacaDataset(Dataset):
    """Alpaca instruction-following dataset do treningu KD."""

    def __init__(self, data, tokenizer, max_len: int = KD_MAX_SEQ_LEN):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        text = f"### Instruction:\n{d['instruction']}"
        if d.get("input"):
            text += f"\n### Input:\n{d['input']}"
        text += f"\n### Response:\n{d['output']}"

        enc = self.tok(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def run_kd(
    teacher_path: str,
    student_path: str,
    output_path: str,
    T: float = KD_TEMPERATURE,
    alpha: float = KD_ALPHA,
    lr: float = KD_LR,
    epochs: int = KD_EPOCHS,
    samples: int = KD_SAMPLES,
    batch: int = KD_BATCH_SIZE,
    grad_accum: int = KD_GRAD_ACCUM,
    max_seq_len: int = KD_MAX_SEQ_LEN,
) -> None:
    print(f"{'='*50}")
    print(f"KNOWLEDGE DISTILLATION")
    print(f"  Teacher : {teacher_path}")
    print(f"  Student : {student_path}")
    print(f"  Wyjście : {output_path}")
    print(f"  T={T}, alpha={alpha}, lr={lr}")
    print(f"  epochs={epochs}, samples={samples}, batch={batch}")
    print(f"{'='*50}")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Teacher (zamrożony) ────────────────────────────────
    print(">>> Ładowanie teachera (INT8 — oszczędność VRAM)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e9
    print(f"   Teacher: {teacher_params:.2f}B params | device: {next(teacher.parameters()).device}")

    # ── Student (do trenowania z LoRA) ────────────────────
    print(">>> Ładowanie studenta...")
    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Auto-detekcja warstw attention do LoRA (różne rodziny mają różne nazwy):
    #   Qwen/Llama: q_proj, k_proj, v_proj, o_proj
    #   Phi-3.5:    qkv_proj (fused), o_proj
    module_names = {n.split(".")[-1] for n, _ in student.named_modules()}
    if "qkv_proj" in module_names:
        targets = ["qkv_proj", "o_proj"]
    else:
        targets = [m for m in ["q_proj", "k_proj", "v_proj", "o_proj"] if m in module_names]
    print(f"   LoRA target_modules (auto): {targets}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )
    student = get_peft_model(student, lora_config)
    student.gradient_checkpointing_enable()
    student.train()

    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "Student nie ma trainable params!\n"
            "Prawdopodobnie załadowałeś model GPTQ (wagi zamrożone w INT4).\n"
            "Rozwiązania:\n"
            "  A) KD musi być PRZED kwantyzacją (scenariusze S3, S5, S6)\n"
            "  B) Załaduj FP16 checkpoint, zrób KD, następnie ponownie kwantyzuj"
        )

    student_params = sum(p.numel() for p in student.parameters()) / 1e9
    trainable_params = sum(p.numel() for p in trainable) / 1e9
    print(f"   Student: {student_params:.2f}B params | trainable (LoRA): {trainable_params*1000:.1f}M")

    # ── Dane treningowe ────────────────────────────────────
    print(f">>> Pobieranie {samples} próbek z Alpaca...")
    raw = load_dataset("tatsu-lab/alpaca", split="train")
    raw = raw.shuffle(seed=42).select(range(min(samples, len(raw))))
    loader = DataLoader(
        AlpacaDataset(raw, tokenizer, max_len=max_seq_len),
        batch_size=batch,
        shuffle=True,
        drop_last=True,
    )
    print(f"   Batchy na epokę: {len(loader)} (eff. batch={batch * grad_accum})")

    # ── Optymalizator ──────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    # ── Trening ────────────────────────────────────────────
    log_file = Path(LOGS_DIR) / f"kd_{Path(output_path).name}.log"
    best_loss = float("inf")

    with open(log_file, "w") as log:
        log.write("epoch,step,kl_loss,ce_loss,total_loss\n")

        for epoch in range(epochs):
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch_data in enumerate(loader):
                ids  = batch_data["input_ids"].to(student.device)
                mask = batch_data["attention_mask"].to(student.device)

                # Forward teacher (no grad)
                with torch.no_grad():
                    t_logits = teacher(input_ids=ids, attention_mask=mask).logits.float().cpu()

                # Forward student
                s_out = student(input_ids=ids, attention_mask=mask, labels=ids)
                s_logits = s_out.logits.float()
                ce_loss  = s_out.loss

                # KL divergence (soft targets)
                kl_loss = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits.to(s_logits.device) / T, dim=-1),
                    reduction="batchmean",
                ) * (T ** 2)
                del t_logits

                loss = (alpha * kl_loss + (1 - alpha) * ce_loss) / grad_accum
                loss.backward()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                step_loss = loss.item() * grad_accum
                total_loss += step_loss

                if step % 50 == 0:
                    msg = (f"  E{epoch+1}/{epochs} S{step+1}/{len(loader)} "
                           f"kl={kl_loss.item():.4f} ce={ce_loss.item():.4f} "
                           f"total={step_loss:.4f}")
                    print(msg)
                    log.write(f"{epoch+1},{step+1},{kl_loss.item():.4f},"
                              f"{ce_loss.item():.4f},{step_loss:.4f}\n")

            avg_loss = total_loss / len(loader)
            print(f"\n  >>> Epoch {epoch+1} — avg_loss={avg_loss:.4f}\n")
            best_loss = min(best_loss, avg_loss)

    # ── Zapis modelu ───────────────────────────────────────
    print(">>> Scalanie LoRA z wagami i zapisywanie...")
    student = student.merge_and_unload()
    student.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    del teacher, student
    torch.cuda.empty_cache()

    print(f"\n✅ KD zakończone → {output_path}")
    print(f"   Best loss: {best_loss:.4f} | Log: {log_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Knowledge Distillation")
    p.add_argument("--teacher", required=True, help="Ścieżka do modelu teachera")
    p.add_argument("--student", required=True, help="Ścieżka do modelu studenta")
    p.add_argument("--output",  required=True, help="Ścieżka do wyjściowego modelu")
    p.add_argument("--samples", type=int,   default=KD_SAMPLES)
    p.add_argument("--epochs",  type=int,   default=KD_EPOCHS)
    p.add_argument("--temp",    type=float, default=KD_TEMPERATURE, help="Temperatura KD")
    p.add_argument("--alpha",   type=float, default=KD_ALPHA,       help="Waga KL vs CE (0-1)")
    p.add_argument("--lr",      type=float, default=KD_LR)
    a = p.parse_args()

    run_kd(
        teacher_path=a.teacher,
        student_path=a.student,
        output_path=a.output,
        T=a.temp,
        alpha=a.alpha,
        lr=a.lr,
        epochs=a.epochs,
        samples=a.samples,
    )
