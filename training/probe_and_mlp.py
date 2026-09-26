"""Two cheap baselines on the same splits, labels and metric as train_qwen / bench_ckpts:

  probe   heads-only training on frozen base-model features (linear-probe ablation of LoRA)
          .venv/Scripts/python probe_and_mlp.py probe --emb out/emb_qwen3_1p7b_base.npy --tag probe_1p7b
  mlp     MLP on parsed prompt features, optionally with pairwise differences/ratios of numeric
          features (tests whether the LLM's gain is relational)
          .venv/Scripts/python probe_and_mlp.py mlp --hidden 256,128 --pairs 20 --tag mlp_rel

Writes out/bench_1p7b/<tag>.json (val/test per head + channel means) and <tag>.{val,test}.rows.jsonl
in the bench_ckpts format, so stats.py can compare everything pairwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from student_generic import parse, build, vectorize  # noqa: E402


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def head_specs(rows):
    spec = {}
    for r in rows:
        for f, t in r["targets"].items():
            spec.setdefault(f"{r['channel']}/{f}", (t["kind"], len(t["options"])))
    return spec


def pair_features(rows, vocab, x, k):
    """Per channel: differences and ratios between its k highest-variance numeric features."""
    tr = np.array([r["split"] == "train" for r in rows])
    cols = []
    by_channel = {}
    for name, j in vocab.items():
        ch, feat = name.split("|", 1)
        if "=" not in feat:   # numeric (one-hots carry '=value')
            by_channel.setdefault(ch, []).append(j)
    for ch, idx in by_channel.items():
        mask = np.array([r["channel"] == ch for r in rows])
        var = x[mask & tr][:, idx].var(0)
        top = [idx[i] for i in np.argsort(-var)[:k] if var[i] > 0]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                u, v = x[:, top[a]], x[:, top[b]]
                cols.append(np.where(mask, u - v, 0.0))
                cols.append(np.where(mask, u / (np.abs(v) + 1.0), 0.0))
    return np.stack(cols, 1).astype(np.float32) if cols else np.zeros((len(rows), 0), np.float32)


class Net(nn.Module):
    def __init__(self, d_in, hidden, spec, normalise):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.normalise = normalise
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))
        self.heads = nn.ModuleDict({k.replace("/", "__").replace("@", "_").replace(".", "_"): nn.Linear(d, n) for k, (_, n) in spec.items()})

    def forward(self, x, key):
        h = self.body(x)
        if self.normalise:
            h = F.normalize(h, dim=-1) * self.log_scale.exp()
        return self.heads[key.replace("/", "__").replace("@", "_").replace(".", "_")](h)


def batches(rows, idx, bs, rng):
    idx = np.array(idx)
    rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        yield idx[s:s + bs]


def loss_on(net, X, rows, b, spec, n_fields, dev):
    groups = {}
    for i in b:
        for f, t in rows[i]["targets"].items():
            groups.setdefault(f"{rows[i]['channel']}/{f}", []).append(i)
    total = torch.zeros((), device=dev)
    for key, ids in groups.items():
        field = key.split("/", 1)[1]
        logits = net(X[ids], key)
        t = torch.tensor([rows[i]["targets"][field]["dist"] for i in ids], device=dev, dtype=torch.float32)
        w = torch.tensor([1.0 / n_fields[rows[i]["channel"]] for i in ids], device=dev)
        if spec[key][0] == "choice":
            legal = torch.tensor([rows[i]["targets"][field].get("legal", [True] * t.shape[1]) for i in ids], device=dev)
            t = t / t.sum(-1, keepdim=True).clamp_min(1e-8)
            l = -(t * F.log_softmax(logits.masked_fill(~legal, -1e4), -1)).sum(-1)
        else:
            l = F.binary_cross_entropy_with_logits(logits, t.clamp(0, 1), reduction="none").mean(-1)
        total = total + (l * w).sum()
    return total / len(b)


@torch.no_grad()
def evaluate(net, X, rows, idx, dev):
    stats, per_row = {}, {}
    groups = {}
    for i in idx:
        for f in rows[i]["targets"]:
            groups.setdefault(f"{rows[i]['channel']}/{f}", []).append(i)
    for key, ids in groups.items():
        field = key.split("/", 1)[1]
        logits = net(X[ids], key)
        tt = [rows[i]["targets"][field] for i in ids]
        if tt[0]["kind"] == "choice":
            legal = torch.tensor([t.get("legal", [True] * len(t["dist"])) for t in tt], device=dev)
            preds = logits.masked_fill(~legal, -1e4).argmax(-1).tolist()
            hits = [float(t["dist"][p] == max(t["dist"])) for t, p in zip(tt, preds)]
        else:
            preds = (torch.sigmoid(logits) >= 0.5).tolist()
            hits = []
            for t, pred in zip(tt, preds):
                gold = [p >= 0.5 for p in t["dist"]]
                tp = sum(a and b for a, b in zip(pred, gold))
                hits.append((2 * tp / (sum(pred) + sum(gold))) if (sum(pred) + sum(gold)) else 1.0)
        stats[key] = round(sum(hits) / len(hits), 4)
        for i, h in zip(ids, hits):
            per_row.setdefault(i, []).append(h)
    ch = {}
    for k, v in stats.items():
        ch.setdefault(k.split("/")[0], []).append(v)
    ch = {c: round(sum(v) / len(v), 4) for c, v in sorted(ch.items())}
    return stats, ch, round(sum(ch.values()) / len(ch), 4), per_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "mlp"])
    ap.add_argument("--data", default=os.path.join(HERE, "out", "sft_qwen_v2.jsonl"))
    ap.add_argument("--emb", default="")
    ap.add_argument("--hidden", default="")
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "bench_1p7b"))
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(args.data)
    spec = head_specs(rows)
    n_fields = {r["channel"]: len(r["targets"]) for r in rows}
    if args.mode == "probe":
        X = np.load(args.emb).astype(np.float32)
        normalise = True
    else:
        vocab = build(rows)
        X = vectorize(rows, vocab)
        if args.pairs:
            X = np.concatenate([X, pair_features(rows, vocab, X, args.pairs)], 1)
        tr = np.array([r["split"] == "train" for r in rows])
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        normalise = False
    print(f"{args.tag}: features {X.shape[1]} on {dev}", flush=True)
    Xt = torch.tensor(X, device=dev)
    hidden = [int(h) for h in args.hidden.split(",") if h]
    net = Net(X.shape[1], hidden, spec, normalise).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    split = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train", "val", "test")}
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        net.train()
        for b in batches(rows, split["train"], 128, rng):
            opt.zero_grad()
            loss_on(net, Xt, rows, b, spec, n_fields, dev).backward()
            opt.step()
        net.eval()
        _, _, vm, _ = evaluate(net, Xt, rows, split["val"], dev)
        if vm > best:
            best, best_state = vm, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep}: val {vm:.4f} (best {best:.4f})", flush=True)
    net.load_state_dict(best_state)
    res = {}
    for s in ("val", "test"):
        heads, ch, mean, per_row = evaluate(net, Xt, rows, split[s], dev)
        res[s] = {"heads": heads, "channels": ch, "mean": mean}
        local = {g: n for n, g in enumerate(split[s])}
        with open(os.path.join(args.out, f"{args.tag}.{s}.rows.jsonl"), "w", encoding="utf-8") as f:
            for g in split[s]:
                f.write(json.dumps({"i": local[g], "channel": rows[g]["channel"], "hit": round(sum(per_row[g]) / len(per_row[g]), 4)}) + "\n")
    res["summary"] = {"val_mean": res["val"]["mean"], "test_mean": res["test"]["mean"]}
    json.dump(res, open(os.path.join(args.out, f"{args.tag}.json"), "w", encoding="utf-8"), indent=1)
    print(args.tag, res["summary"], json.dumps(res["test"]["channels"]), flush=True)


if __name__ == "__main__":
    main()
