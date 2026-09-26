"""Track A for every channel: numpy MLP on features parsed from the canonical prompt text.

  python student_generic.py --data out/sft_v1.jsonl --out out/mlp_generic_v1.json

Same targets, same splits and the same metric definitions as train_qwen.py, so the two tracks
are directly comparable. Features: every "key=value" pair in the prompt (numbers as values,
strings as one-hot), plus boss tactic lines ("<option>: dmg=.. area=..") as per-option numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np

PAIR = re.compile(r"([A-Za-z_]+)=([^\s]+)")


def parse(prompt):
    feats, section = {}, ""
    for line in prompt.splitlines():
        if line.startswith("<"):
            section = line.split(">")[0][1:]
            continue
        prefix = ""
        m = re.match(r"^([a-z_]+):\s", line)
        if m:   # boss "<option>: dmg=..." lines
            prefix = m.group(1) + "."
        elif " " in line and "=" not in line.split(" ", 1)[0]:
            prefix = line.split(" ", 1)[0] + "."
        for k, v in PAIR.findall(line):
            key = f"{section}.{prefix}{k}"
            for part_i, part in enumerate(v.replace("->", " ").replace("/", " ").split()):
                name = key if part_i == 0 else f"{key}#{part_i}"
                try:
                    feats[name] = float(part)
                except ValueError:
                    feats[f"{name}={part}"] = 1.0
    return feats


def build(rows):
    vocab = {}
    for r in rows:
        if r["split"] == "train":
            for k in parse(r["prompt"]):
                vocab.setdefault(f"{r['channel']}|{k}", len(vocab))
    return vocab


def vectorize(rows, vocab):
    x = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, r in enumerate(rows):
        for k, v in parse(r["prompt"]).items():
            j = vocab.get(f"{r['channel']}|{k}")
            if j is not None:
                x[i, j] = v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/sft_v1.jsonl")
    ap.add_argument("--out", default="out/mlp_generic_v1.json")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    vocab = build(rows)
    x = vectorize(rows, vocab)
    tr = np.array([r["split"] == "train" for r in rows])
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    x = (x - mu) / sd
    heads = {}
    for r in rows:
        for f, t in r["targets"].items():
            heads.setdefault(f"{r['channel']}/{f}", (t["kind"], len(t["options"])))
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, math.sqrt(2 / x.shape[1]), (x.shape[1], args.hidden)).astype(np.float32)
    b1 = np.zeros(args.hidden, np.float32)
    HW = {k: (rng.normal(0, math.sqrt(1 / args.hidden), (args.hidden, n)).astype(np.float32), np.zeros(n, np.float32))
          for k, (_, n) in heads.items()}
    params = [W1, b1] + [p for w in HW.values() for p in w]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    by_head = {k: [] for k in heads}
    for i, r in enumerate(rows):
        for f in r["targets"]:
            by_head[f"{r['channel']}/{f}"].append(i)
    step = 0
    idx_train = np.where(tr)[0]
    for epoch in range(args.epochs):
        rng.shuffle(idx_train)
        for s in range(0, len(idx_train), 128):
            b = idx_train[s:s + 128]
            bset = set(b.tolist())
            h = np.maximum(0, x[b] @ W1 + b1)
            dh = np.zeros_like(h)
            grads = {}
            pos = {int(i): j for j, i in enumerate(b)}
            for key, (kind, n) in heads.items():
                ids = [i for i in by_head[key] if i in bset]
                if not ids:
                    grads[key] = (np.zeros_like(HW[key][0]), np.zeros_like(HW[key][1]))
                    continue
                hj = h[[pos[i] for i in ids]]
                field = key.split("/", 1)[1]
                t = np.array([rows[i]["targets"][field]["dist"] for i in ids], dtype=np.float32)
                z = hj @ HW[key][0] + HW[key][1]
                if kind == "choice":
                    legal = np.array([rows[i]["targets"][field].get("legal", [True] * n) for i in ids])
                    t = t / np.maximum(t.sum(1, keepdims=True), 1e-8)
                    z = np.where(legal, z, -1e9)
                    p = np.exp(z - z.max(1, keepdims=True)) * legal
                    p /= p.sum(1, keepdims=True)
                    dz = (p - t) / len(b)
                else:
                    dz = (1 / (1 + np.exp(-z)) - t) / (len(b) * n)
                grads[key] = (hj.T @ dz, dz.sum(0))
                dh[[pos[i] for i in ids]] += dz @ HW[key][0].T
            dh *= h > 0
            g = [x[b].T @ dh, dh.sum(0)] + [q for key in HW for q in grads[key]]
            step += 1
            for k2, (p_, g_) in enumerate(zip(params, g)):
                m[k2] = 0.9 * m[k2] + 0.1 * g_
                v[k2] = 0.999 * v[k2] + 0.001 * g_ * g_
                p_ -= args.lr * (m[k2] / (1 - 0.9 ** step)) / (np.sqrt(v[k2] / (1 - 0.999 ** step)) + 1e-8)

    def evaluate(split):
        stats = {}
        for key, (kind, n) in heads.items():
            field = key.split("/", 1)[1]
            ids = [i for i in by_head[key] if rows[i]["split"] == split]
            if not ids:
                continue
            h = np.maximum(0, x[ids] @ W1 + b1)
            z = h @ HW[key][0] + HW[key][1]
            hit = 0.0
            for j, i in enumerate(ids):
                t = rows[i]["targets"][field]
                if kind == "choice":
                    legal = np.array(t.get("legal", [True] * n))
                    pred = int(np.argmax(np.where(legal, z[j], -1e9)))
                    hit += float(t["dist"][pred] == max(t["dist"]))
                else:
                    pred, gold = z[j] > 0, np.array(t["dist"]) >= 0.5
                    denom = pred.sum() + gold.sum()
                    hit += (2 * (pred & gold).sum() / denom) if denom else 1.0
            stats[key] = round(hit / len(ids), 4)
        return stats

    metrics = {"val": evaluate("val"), "test": evaluate("test"), "features": len(vocab),
               "params": int(sum(p.size for p in params))}
    print(json.dumps(metrics, indent=1))
    json.dump({"metrics": metrics, "vocab": vocab, "mu": mu.tolist(), "sd": sd.tolist(), "W1": W1.tolist(), "b1": b1.tolist(),
               "heads": {k: {"kind": heads[k][0], "W": w.tolist(), "b": bb.tolist()} for k, (w, bb) in HW.items()}},
              open(args.out, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
