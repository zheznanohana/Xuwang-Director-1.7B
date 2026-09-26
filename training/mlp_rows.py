"""Per-row hits of the saved MLP baseline, in the same format as bench_ckpts.py rows files.

  python mlp_rows.py --model out/mlp_qwen_v2.json --data out/sft_qwen_v2.jsonl --out out/bench_1p7b/mlp
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from student_generic import vectorize


def hit(t, pred):
    if t["kind"] == "choice":
        return float(t["dist"][pred] == max(t["dist"]))
    gold = [p >= 0.5 for p in t["dist"]]
    tp = sum(a and b for a, b in zip(pred, gold))
    return (2 * tp / (sum(pred) + sum(gold))) if (sum(pred) + sum(gold)) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/mlp_qwen_v2.json")
    ap.add_argument("--data", default="out/sft_qwen_v2.jsonl")
    ap.add_argument("--out", default="out/bench_1p7b/mlp")
    args = ap.parse_args()
    m = json.load(open(args.model, encoding="utf-8"))
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    W1, b1 = np.asarray(m["W1"], np.float32), np.asarray(m["b1"], np.float32)
    mu, sd = np.asarray(m["mu"], np.float32), np.asarray(m["sd"], np.float32)
    heads = {k: (v["kind"], np.asarray(v["W"], np.float32), np.asarray(v["b"], np.float32)) for k, v in m["heads"].items()}
    summary = {}
    for split in ("val", "test"):
        data = [r for r in rows if r["split"] == split]
        h = np.maximum(0, ((vectorize(data, m["vocab"]) - mu) / sd) @ W1 + b1)
        per_head = {}
        with open(f"{args.out}.{split}.rows.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(data):
                hits = []
                for field, t in r["targets"].items():
                    key = f"{r['channel']}/{field}"
                    kind, W, b = heads[key]
                    z = h[i] @ W + b
                    if kind == "choice":
                        legal = np.asarray(t.get("legal", [True] * len(z)))
                        pred = int(np.argmax(np.where(legal, z, -1e9)))
                    else:
                        pred = [bool(x) for x in z >= 0]
                    hv = hit(t, pred)
                    hits.append(hv)
                    per_head.setdefault(key, []).append(hv)
                f.write(json.dumps({"i": i, "channel": r["channel"], "hit": round(sum(hits) / len(hits), 4)}) + "\n")
        ch = {}
        for k, v in per_head.items():
            ch.setdefault(k.split("/")[0], []).append(sum(v) / len(v))
        summary[split] = round(sum(sum(v) / len(v) for v in ch.values()) / len(ch), 4)
    print("mlp", summary, "(saved metrics:", {s: m["metrics"].get(s) is not None for s in ("val", "test")}, ")")


if __name__ == "__main__":
    main()
