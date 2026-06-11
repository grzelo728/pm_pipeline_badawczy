"""
Pruning modelu po kwantyzacji GPTQ (scenariusz S6: KD->Q->P).
Magnitude pruning nie operuje na spakowanych wagach INT4 (MarlinLinear),
więc najpierw dekwantyzujemy wagi metodą macierzy jednostkowej
(y = I @ W^T = W^T, niezależne od formatu pakowania kernela),
a następnie zerujemy 50% najmniejszych co do |w| — per warstwa.
Błąd kwantyzacji INT4 zostaje wpalony w odzyskane wagi FP16.

Obsługa fused layers (Phi-3): gptqmodel rozdziela fused `gate_up_proj`
na `gate_proj`+`up_proj` (i ew. `qkv_proj` na `q/k/v_proj`). Naczynie
(natywny model) ma wersje fused, więc składamy komponenty z powrotem
(konkatenacja wzdłuż wymiaru wyjścia, w kolejności jak w architekturze).

Użycie:
  python 03b_prune_gptq.py \
    --quant  models/phi_S6_step2_q \
    --vessel models/phi_shared_step1_KD \
    --output models/phi_S6_step3_p
"""
import argparse
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel

SPARSITY = 0.5  # nadpisywane przez --sparsity (0 = czysta dekwantyzacja+fuzja, bez pruningu)


def prune_(W, sparsity):  # per-warstwa magnitude (sparsity=0 -> bez zerowania)
    k = int(W.numel() * sparsity)
    if k > 0:
        thr = W.abs().flatten().kthvalue(k).values
        W = W * (W.abs() > thr)
    return W


def run(quant_path, vessel_path, output_path, sparsity=SPARSITY):
    print(">>> Ładowanie naczynia FP16 (struktura):", vessel_path)
    vessel = AutoModelForCausalLM.from_pretrained(
        vessel_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(vessel_path, trust_remote_code=True)
    vessel_lin = {n: m for n, m in vessel.named_modules() if isinstance(m, nn.Linear)}

    print(">>> Ładowanie modelu INT4 (GPTQ):", quant_path)
    qm = GPTQModel.load(quant_path, device="cuda")
    qmodel = qm.model

    def is_q(m):
        return hasattr(m, "qweight") and hasattr(m, "scales")

    # ── PASS 1: dekwantyzacja + pruning wszystkich warstw kwantyzowanych ──
    pruned = {}  # nazwa -> W (cpu, fp16)
    for name, ql in qmodel.named_modules():
        if not is_q(ql):
            continue
        I = torch.eye(ql.in_features, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            y = ql(I)
        if getattr(ql, "bias", None) is not None:
            y = y - ql.bias
        W = prune_(y.t().contiguous().float(), sparsity)
        pruned[name] = W.half().cpu()
        del I, y, W; torch.cuda.empty_cache()

    # ── PASS 2: zapis do naczynia (bezpośrednio lub składając fused) ──
    tot = zer = cnt = 0
    used = set()
    for vname, vmod in vessel_lin.items():
        leaf = vname.rsplit(".", 1)[-1]
        base = vname.rsplit(".", 1)[0] if "." in vname else ""

        if vname in pruned:                      # bezpośrednie dopasowanie
            W = pruned[vname]; used.add(vname)
        else:
            # czy to warstwa fused, której komponenty gptqmodel rozdzielił?
            order = None
            if leaf == "gate_up_proj":
                order = ["gate_proj", "up_proj"]
            elif leaf == "qkv_proj":
                order = ["q_proj", "k_proj", "v_proj"]
            comps = [f"{base}.{c}" for c in order] if order else []
            if comps and all(c in pruned for c in comps):
                W = torch.cat([pruned[c] for c in comps], dim=0)  # konkatenacja po wyjściu
                for c in comps: used.add(c)
            else:
                # niekwantyzowany (np. lm_head) — przytnij z własnych wag naczynia
                W = prune_(vmod.weight.data.float(), sparsity).half()

        if W.shape != vmod.weight.shape:
            print(f"  ⚠ niezgodny kształt {vname}: {tuple(W.shape)} vs {tuple(vmod.weight.shape)} — pomijam")
            continue
        vmod.weight.data = W
        tot += W.numel(); zer += (W == 0).sum().item(); cnt += 1

    leftover = [k for k in pruned if k not in used]
    if leftover:
        print(f"  ⚠ {len(leftover)} warstw kwantyzowanych nie trafiło do naczynia (np.: {leftover[:3]})")

    print(f">>> Gotowe: {cnt} warstw zapisanych, sparsity globalna={zer/tot:.4f}")
    print(">>> Zapisywanie:", output_path)
    vessel.save_pretrained(output_path)
    tok.save_pretrained(output_path)
    print("✅ Zapisano:", output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quant", required=True, help="Model po GPTQ (INT4)")
    p.add_argument("--vessel", required=True, help="Model FP16 o tej samej architekturze (naczynie)")
    p.add_argument("--output", required=True, help="Ścieżka wyjściowa")
    p.add_argument("--sparsity", type=float, default=SPARSITY,
                   help="0.5 = pruning (S6); 0 = sama dekwantyzacja+fuzja (eval GPTQ Phi)")
    a = p.parse_args()
    run(a.quant, a.vessel, a.output, a.sparsity)
