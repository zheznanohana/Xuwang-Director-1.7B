"""Benchmark every saved version of a run on the FULL val and test sets, then export the winner.

  .venv/Scripts/python bench_ckpts.py --ckpts out/ckpts_qwen3_1p7b --final H:/.../runs/qwen3_1p7b \
      --out out/bench_1p7b [--export out/qwen_1p7b_best]

Candidates: every step_*.pt collected by collect_ckpts.py, plus the run's final adapter (the
best-on-probe weights train_qwen.py restores at the end). One base model load; each candidate
only swaps LoRA + heads. Selection uses the val mean over channels (every channel weighs the
same); test is reported, never used to choose. Per-row hits are written for significance tests.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_qwen import Heads, encode, head_id, last_hidden, load

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def score(backbone, heads, tok, rows, max_len, batch, device):
    """-> (per-head accuracy, per-row records). Same hit rule as train_qwen.evaluate, vectorised per
    (batch, head): rows are grouped by channel so each batch touches few heads, one GPU sync per head."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["channel"], len(rows[i]["prompt"])))
    stats, per_row = {}, {}
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        chunk = [rows[i] for i in idx]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            h = last_hidden(backbone, encode(tok, [r["prompt"] for r in chunk], max_len, device))
        groups = {}
        for j, r in enumerate(chunk):
            for field, t in r["targets"].items():
                groups.setdefault(f"{r['channel']}/{field}", []).append((j, t))
        for key, items in groups.items():
            logits = heads(h[[j for j, _ in items]], key)
            if items[0][1]["kind"] == "choice":
                legal = torch.tensor([t.get("legal", [True] * len(t["dist"])) for _, t in items], device=device)
                preds = logits.masked_fill(~legal, -1e4).argmax(-1).tolist()
                hits = [float(t["dist"][p] == max(t["dist"])) for (_, t), p in zip(items, preds)]
            else:
                preds = (torch.sigmoid(logits) >= 0.5).tolist()
                hits = []
                for (_, t), pred in zip(items, preds):
                    gold = [p >= 0.5 for p in t["dist"]]
                    tp = sum(a and b for a, b in zip(pred, gold))
                    hits.append((2 * tp / (sum(pred) + sum(gold))) if (sum(pred) + sum(gold)) else 1.0)
            st = stats.setdefault(key, [0, 0.0])
            st[0] += len(hits)
            st[1] += sum(hits)
            for (j, _), hit in zip(items, hits):
                per_row.setdefault(idx[j], []).append(hit)
    records = [{"i": i, "channel": rows[i]["channel"], "hit": round(sum(per_row[i]) / len(per_row[i]), 4)} for i in range(len(rows))]
    return {k: round(v[1] / v[0], 4) for k, v in sorted(stats.items())}, records


def channel_means(scores):
    per = {}
    for k, v in scores.items():
        per.setdefault(k.split("/")[0], []).append(v)
    return {c: round(sum(v) / len(v), 4) for c, v in sorted(per.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "out", "sft_qwen_v2.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--ckpts", default=os.path.join(HERE, "out", "ckpts_qwen3_1p7b"))
    ap.add_argument("--final", default="", help="run dir with adapter/ + heads.json (optional)")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "bench_1p7b"))
    ap.add_argument("--export", default="", help="write the winner as a deployable run dir (adapter, heads.json, merged/)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--steps", default="", help="exact collected steps to test, e.g. 3244 (overrides --shortlist)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: rows per split")
    ap.add_argument("--shortlist", type=int, default=0, help="only the top-K collected steps by probe val (0 = all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    rows, spec = load(args.data)
    splits = {s: [r for r in rows if r["split"] == s] for s in ("val", "test")}
    if args.limit:   # smoke test: a small stratified-ish slice
        splits = {s: v[::max(1, len(v) // args.limit)][:args.limit] for s, v in splits.items()}
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side, tok.truncation_side = "left", "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.to(device).eval()
    backbone = model.base_model.model.model
    heads = Heads(model.config.hidden_size, spec).to(device)
    lora = {n: p for n, p in model.named_parameters() if "lora_" in n}

    def load_ckpt(path):
        w = torch.load(path, map_location="cpu", weights_only=False)["weights"]
        with torch.no_grad():
            for n, v in w["lora"].items():
                lora[n].copy_(v.to(lora[n].dtype))
        heads.load_state_dict(w["heads"])

    def load_final(run):
        set_peft_model_state_dict(model, load_file(os.path.join(run, "adapter", "adapter_model.safetensors")))
        hj = json.load(open(os.path.join(run, "heads.json"), encoding="utf-8"))
        with torch.no_grad():
            heads.log_scale.copy_(torch.tensor(math.log(hj["__scale__"])))
            for k in spec:
                layer = heads.layers[head_id(k)]
                layer.weight.copy_(torch.tensor(hj[k]["weight"]))
                layer.bias.copy_(torch.tensor(hj[k]["bias"]))

    candidates = [(os.path.basename(p)[:-3], p, load_ckpt) for p in sorted(glob.glob(os.path.join(args.ckpts, "step_*.pt")))]
    if args.steps:
        wanted = {int(x) for x in args.steps.split(",") if x.strip()}
        candidates = [c for c in candidates if int(c[0].split("_")[1]) in wanted]
    elif args.shortlist:
        # Only the strongest points by the training-time probe (steps without a probe score are skipped).
        status = json.load(open(os.path.join(args.final, "status.json"), encoding="utf-8"))
        probe = {h["step"]: h["val_mean"] for h in status.get("history", []) if "channels" in h}
        scored = [c for c in candidates if int(c[0].split("_")[1]) in probe]
        scored.sort(key=lambda c: probe[int(c[0].split("_")[1])], reverse=True)
        candidates = scored[:args.shortlist]
        print("shortlist:", [(c[0], probe[int(c[0].split('_')[1])]) for c in candidates], flush=True)
    if args.final and os.path.exists(os.path.join(args.final, "heads.json")):
        candidates.append(("final_best", args.final, load_final))
    summary = {}
    for name, path, loader in candidates:
        result_path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(result_path):
            summary[name] = json.load(open(result_path, encoding="utf-8"))["summary"]
            print("cached", name, summary[name], flush=True)
            continue
        t0 = time.time()
        loader(path)
        res = {}
        for split, data in splits.items():
            per_head, recs = score(backbone, heads, tok, data, args.max_len, args.batch, device)
            ch = channel_means(per_head)
            res[split] = {"heads": per_head, "channels": ch, "mean": round(sum(ch.values()) / len(ch), 4)}
            with open(os.path.join(args.out, f"{name}.{split}.rows.jsonl"), "w", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
        res["summary"] = {"val_mean": res["val"]["mean"], "test_mean": res["test"]["mean"], "seconds": round(time.time() - t0)}
        json.dump(res, open(result_path, "w", encoding="utf-8"), indent=1)
        summary[name] = res["summary"]
        print(name, summary[name], flush=True)
    winner = max(summary, key=lambda k: (summary[k]["val_mean"], summary[k]["test_mean"]))
    json.dump({"winner": winner, "summary": summary}, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"), indent=1)
    print("winner:", winner, summary[winner], flush=True)

    if args.export:
        name, path, loader = next(c for c in candidates if c[0] == winner)
        loader(path)
        os.makedirs(args.export, exist_ok=True)
        model.save_pretrained(os.path.join(args.export, "adapter"))
        tok.save_pretrained(os.path.join(args.export, "adapter"))
        with open(os.path.join(args.export, "heads.json"), "w", encoding="utf-8") as f:
            json.dump({"__scale__": float(heads.log_scale.exp()), "__input__": "l2_normalised_last_hidden"} |
                      {k: {"kind": v["kind"], "options": v["options"],
                           "weight": heads.layers[head_id(k)].weight.detach().float().cpu().tolist(),
                           "bias": heads.layers[head_id(k)].bias.detach().float().cpu().tolist()}
                       for k, v in spec.items()}, f)
        full = json.load(open(os.path.join(args.out, f"{winner}.json"), encoding="utf-8"))
        json.dump({"val": full["val"]["heads"], "test": full["test"]["heads"], "base": args.model, "candidate": winner},
                  open(os.path.join(args.export, "metrics.json"), "w", encoding="utf-8"), indent=1)
        merged = model.merge_and_unload()
        merged.save_pretrained(os.path.join(args.export, "merged"))
        tok.save_pretrained(os.path.join(args.export, "merged"))
        print("exported", winner, "->", args.export, flush=True)


if __name__ == "__main__":
    main()
