"""Track B: Qwen3 Base + LoRA + per-field heads on the last-token hidden state (multi-task).

  .venv/Scripts/python train_qwen.py --data out/sft_v1.jsonl --out out/qwen_v1 [--merge]

One forward pass per decision: the prompt ends with "<decide>", the final hidden state feeds
one linear head per (channel, field). Choice fields train with soft cross-entropy restricted to
legal options; multi-label fields with BCE on the teacher's per-tag frequencies. No answer text is generated. Legal masks restrict individual fields; joint
constraints still require host validation.
Outputs: LoRA adapter, heads.json (plain weights, loadable anywhere), metrics.json, optional
merged checkpoint for llama.cpp GGUF conversion (embedding mode, pooling=last).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def head_id(key):
    return key.replace("/", "__").replace("@", "_").replace(".", "_")


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    spec = {}
    for r in rows:
        for field, t in r["targets"].items():
            spec.setdefault(f"{r['channel']}/{field}", {"kind": t["kind"], "options": t["options"]})
    return rows, spec


class Heads(nn.Module):
    """Heads read the L2-normalised hidden state times a learned scale, so the runtime gives the
    same answer whether or not the inference backend (llama.cpp embeddings) normalises vectors."""

    def __init__(self, hidden, spec):
        super().__init__()
        self.spec = spec
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))
        self.layers = nn.ModuleDict({head_id(k): nn.Linear(hidden, len(v["options"])) for k, v in spec.items()})

    def forward(self, h, key):
        return self.layers[head_id(key)](F.normalize(h, dim=-1) * self.log_scale.exp())


def field_losses(h, rows, heads, n_fields):
    groups = {}
    for i, r in enumerate(rows):
        for field, t in r["targets"].items():
            groups.setdefault(f"{r['channel']}/{field}", []).append((i, t, 1.0 / n_fields[r["channel"]]))
    total = h.new_zeros(())
    for key, items in groups.items():
        idx = torch.tensor([i for i, _, _ in items], device=h.device)
        w = torch.tensor([wt for _, _, wt in items], device=h.device)
        logits = heads(h[idx], key)
        target = torch.tensor([t["dist"] for _, t, _ in items], device=h.device, dtype=torch.float32)
        if heads.spec[key]["kind"] == "choice":
            legal = torch.tensor([t.get("legal", [True] * len(t["dist"])) for _, t, _ in items], device=h.device)
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
            logp = F.log_softmax(logits.masked_fill(~legal, -1e4), -1)
            loss = -(target * logp).sum(-1)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target.clamp(0, 1), reduction="none").mean(-1)
        total = total + (loss * w).sum()
    return total / len(rows)


def encode(tok, prompts, max_len, device):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in enc.items()}


def last_hidden(backbone, enc):
    out = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return out.last_hidden_state[:, -1, :].float()   # left padding -> last position is the real "<decide>"


@torch.no_grad()
def evaluate(backbone, heads, tok, rows, args, device, amp_dtype=torch.bfloat16):
    backbone.eval()
    stats = {}
    for i in range(0, len(rows), args.eval_batch):
        chunk = rows[i:i + args.eval_batch]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
            h = last_hidden(backbone, encode(tok, [r["prompt"] for r in chunk], args.max_len, device))
        for j, r in enumerate(chunk):
            for field, t in r["targets"].items():
                key = f"{r['channel']}/{field}"
                logits = heads(h[j:j + 1], key)[0]
                s = stats.setdefault(key, {"n": 0, "hit": 0.0})
                s["n"] += 1
                if t["kind"] == "choice":
                    legal = torch.tensor(t.get("legal", [True] * len(t["dist"])), device=device)
                    pred = int(logits.masked_fill(~legal, -1e4).argmax())
                    s["hit"] += float(t["dist"][pred] == max(t["dist"]))
                else:
                    pred = (torch.sigmoid(logits) >= 0.5).cpu().tolist()
                    gold = [p >= 0.5 for p in t["dist"]]
                    tp = sum(a and b for a, b in zip(pred, gold))
                    s["hit"] += (2 * tp / (sum(pred) + sum(gold))) if (sum(pred) + sum(gold)) else 1.0
    backbone.train()
    return {k: round(v["hit"] / v["n"], 4) for k, v in sorted(stats.items())}


EVAL_VERSION = 2   # v2: stratified probe + mean over channels (v1 took val[:n], i.e. one channel only)


def stratified(rows, n, seed=1):
    """Up to n rows, round-robin over channels, so the periodic probe sees every channel."""
    by_channel = {}
    for r in rows:
        by_channel.setdefault(r["channel"], []).append(r)
    rng = random.Random(seed)
    queues = []
    for c in sorted(by_channel):
        q = list(by_channel[c])
        rng.shuffle(q)
        queues.append(q)
    out = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop())
    return out


def channel_means(scores):
    per = {}
    for k, v in scores.items():
        per.setdefault(k.split("/")[0], []).append(v)
    return {c: round(sum(v) / len(v), 4) for c, v in sorted(per.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/sft_v1.jsonl")
    ap.add_argument("--out", default="out/qwen_v1")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=2e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--max-eval", type=int, default=0, help="cap val/test rows for the final evaluation (0 = all)")
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--eval-rows", type=int, default=800, help="val rows used for the periodic best-checkpoint check")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--status-file", default="", help="JSON progress file rewritten at every log/eval (remote monitoring)")
    args = ap.parse_args()
    if os.name == "nt":   # keep Windows from sleeping while this process trains (released automatically on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 where the GPU supports it (A100/L4/RTX30+). On T4: fp32 master weights + fp16 autocast with
    # loss scaling (GradScaler) so tensor cores are used without fp16 overflow.
    # Native bf16 only (compute capability >= 8.0). torch's is_bf16_supported() also reports
    # emulated bf16 as supported (e.g. on T4), which is very slow.
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 and os.environ.get("FORCE_FP16") != "1"
    use_fp16 = device == "cuda" and not use_bf16
    # Frozen base weights: bf16, or fp16 on T4 (halves memory); trainable LoRA/heads are kept in fp32 below.
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if use_fp16 and args.batch > 4:
        # T4 (15GB): cap the micro-batch, keep the effective batch the same.
        args.accum *= args.batch // 4
        args.batch = 4
        print(f"T4 mode: micro-batch 4 x accum {args.accum}", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    rows, spec = load(args.data)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    val_probe = stratified(val, args.eval_rows)
    n_fields = {}
    for r in rows:
        n_fields[r["channel"]] = len(r["targets"])
    print(f"rows train={len(train)} val={len(val)} test={len(test)} heads={len(spec)} device={device} "
          f"gpu={torch.cuda.get_device_name(0) if device == 'cuda' else '-'} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    tok.truncation_side = "left"   # keep the tail: the decision marker must survive truncation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.to(device)
    if use_fp16:
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()   # fp32 master copy for GradScaler (fp16 grads cannot be unscaled)
    backbone = model.base_model.model.model   # Qwen3Model with LoRA injected; skips the vocab-sized LM head
    heads = Heads(model.config.hidden_size, spec).to(device)
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
                             {"params": heads.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    steps = math.ceil(len(train) * args.epochs / (args.batch * args.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, steps // 20)) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    # ---- checkpoint / resume (Colab may disconnect, laptops may power off) and best-on-val tracking
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ckpt_path = os.path.join(args.out, "ckpt.pt")

    def snapshot():
        return {"lora": {n: p.detach().float().cpu().clone() for n, p in trainable},
                "heads": {k: v.detach().float().cpu().clone() for k, v in heads.state_dict().items()}}

    def restore(state):
        lookup = dict(trainable)
        with torch.no_grad():
            for n, v in state["lora"].items():
                lookup[n].copy_(v.to(lookup[n].dtype))
        heads.load_state_dict(state["heads"])

    step, micro = 0, 0
    best = {"score": -1.0, "step": 0, "state": None}
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        restore(ck["weights"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = int(ck["step"])
        if ck.get("eval_version") == EVAL_VERSION:
            best = ck.get("best", best)
        else:
            print("checkpoint used an older validation probe: best-on-val tracking restarts", flush=True)
        print(f"resumed from step {step} (best val {best['score']:.4f} @ {best['step']})", flush=True)

    def save_ckpt():
        tmp = ckpt_path + ".tmp"
        torch.save({"weights": snapshot(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best, "eval_version": EVAL_VERSION}, tmp)
        os.replace(tmp, ckpt_path)

    status = {"model": args.model, "steps": steps, "train_rows": len(train), "state": "training", "history": [],
              "loss_curve": [], "eval_version": EVAL_VERSION}
    if args.resume and args.status_file and os.path.exists(args.status_file):
        try:
            prev = json.load(open(args.status_file, encoding="utf-8"))
            status["loss_curve"] = [x for x in prev.get("loss_curve", []) if x[0] <= step]
            if prev.get("eval_version") == EVAL_VERSION:
                status["history"] = [h for h in prev.get("history", []) if h["step"] <= step]
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def write_status(**kw):
        if not args.status_file:
            return
        status.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"), best_step=best["step"], best_val=round(best["score"], 4))
        tmp = args.status_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=1)
        os.replace(tmp, args.status_file)

    t0 = time.time()
    write_status(step=step, state="started")
    order = []
    bad_steps = 0
    while step < steps:
        if not order:
            order = random.sample(train, len(train))
        chunk, order = order[:args.batch], order[args.batch:]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
            h = last_hidden(backbone, encode(tok, [r["prompt"] for r in chunk], args.max_len, device))
        loss = field_losses(h, chunk, heads, n_fields) / args.accum
        if not torch.isfinite(loss):
            # Never let a NaN/inf batch touch the weights; abort if it keeps happening.
            bad_steps += 1
            opt.zero_grad(set_to_none=True)
            micro = 0
            print(f"non-finite loss skipped ({bad_steps})", flush=True)
            if bad_steps >= 20:
                write_status(state="failed", error="20 consecutive non-finite losses")
                raise SystemExit("aborting: 20 consecutive non-finite losses")
            continue
        bad_steps = 0
        scaler.scale(loss).backward()
        micro += 1
        if micro % args.accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(heads.parameters()) + [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == steps:
                mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0
                print(f"step {step}/{steps} loss {loss.item() * args.accum:.4f} {time.time() - t0:.0f}s peak_mem {mem:.2f}GB", flush=True)
                status["loss_curve"].append([step, round(loss.item() * args.accum, 4)])
                write_status(step=step, loss=round(loss.item() * args.accum, 4), elapsed_s=round(time.time() - t0),
                             peak_mem_gb=round(mem, 2), state="training")
            if step % args.eval_every == 0 or step == steps:
                val_scores = evaluate(backbone, heads, tok, val_probe, args, device, amp_dtype)
                per_channel = channel_means(val_scores)
                mean = sum(per_channel.values()) / max(1, len(per_channel))   # every channel weighs the same
                print(f"val mean={mean:.4f}", json.dumps(per_channel), flush=True)
                if mean > best["score"]:
                    best = {"score": mean, "step": step, "state": snapshot()}
                    print(f"best so far @ step {step}", flush=True)
                status["history"].append({"step": step, "val_mean": round(mean, 4), "channels": per_channel})
                write_status(step=step, last_val=per_channel, state="training")
            if step % args.save_every == 0 or step == steps:
                save_ckpt()

    if best["state"] is not None:
        restore(best["state"])   # ship the best-on-val weights, not simply the last step
        print(f"restored best weights from step {best['step']} (val mean {best['score']:.4f})", flush=True)
    cap = (lambda rows: rows[:args.max_eval]) if args.max_eval else (lambda rows: rows)
    metrics = {"val": evaluate(backbone, heads, tok, cap(val), args, device, amp_dtype),
               "test": evaluate(backbone, heads, tok, cap(test), args, device, amp_dtype),
               "train_rows": len(train), "steps": steps, "seconds": round(time.time() - t0), "base": args.model}
    print(json.dumps(metrics, indent=1), flush=True)
    model.save_pretrained(os.path.join(args.out, "adapter"))
    tok.save_pretrained(os.path.join(args.out, "adapter"))
    with open(os.path.join(args.out, "heads.json"), "w", encoding="utf-8") as f:
        json.dump({"__scale__": float(heads.log_scale.exp()), "__input__": "l2_normalised_last_hidden"} | {k: {"kind": v["kind"], "options": v["options"],
                       "weight": heads.layers[head_id(k)].weight.detach().float().cpu().tolist(),
                       "bias": heads.layers[head_id(k)].bias.detach().float().cpu().tolist()}
                   for k, v in spec.items()}, f)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(os.path.join(args.out, "merged"))
        tok.save_pretrained(os.path.join(args.out, "merged"))
    write_status(step=steps, state="done", test=metrics["test"], val=metrics["val"])
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
