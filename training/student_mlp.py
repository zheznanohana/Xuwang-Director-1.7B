"""Track A: tiny MLP distilled from the teacher's soft labels (numpy only).

  python student_mlp.py train --data out/boss_tactic_v1.jsonl --out out/mlp_v1.json

Weights are saved as plain JSON so GDScript can run the same forward pass without native code.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

import features
import sim

OPT_INDEX = {o: i for i, o in enumerate(sim.OPTIONS)}


def load(path, label="pikl"):
    xs, masks, targets, splits = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if label == "llm" and not r["labels"].get("llm", {}).get("dist"):
                continue  # every sample invalid or not labeled yet
            xs.append(features.vectorize(r["state"], r["options"]))
            m = np.zeros(len(sim.OPTIONS), dtype=bool)
            for a in r["legal"]:
                m[OPT_INDEX[a]] = True
            t = np.zeros(len(sim.OPTIONS))
            if label == "fsm":
                t[OPT_INDEX[r["labels"]["fsm"]["choice"]]] = 1.0
            else:
                for a, p in r["labels"][label]["dist"].items():
                    t[OPT_INDEX[a]] = p
            masks.append(m)
            targets.append(t / t.sum())
            splits.append(r["split"])
    return np.array(xs, dtype=np.float32), np.array(masks), np.array(targets, dtype=np.float32), np.array(splits)


def masked_softmax(z, mask):
    z = np.where(mask, z, -1e9)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z) * mask
    return e / e.sum(axis=1, keepdims=True)


class Student:
    def __init__(self, w1, b1, w2, b2):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    @staticmethod
    def init(n_in, n_hidden, n_out, rng):
        return Student(rng.normal(0, math.sqrt(2 / n_in), (n_in, n_hidden)).astype(np.float32),
                       np.zeros(n_hidden, np.float32),
                       rng.normal(0, math.sqrt(1 / n_hidden), (n_hidden, n_out)).astype(np.float32),
                       np.zeros(n_out, np.float32))

    def logits(self, x):
        h = np.maximum(0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2, h

    def policy(self, s):
        """Boss policy hook for generate.py eval (legal mask applied here, like the engine will)."""
        state, opts = features.public_state(s), features.option_features(s)
        x = np.array([features.vectorize(state, opts)], dtype=np.float32)
        mask = np.array([[opts[o]["legal"] for o in sim.OPTIONS]])
        p = masked_softmax(self.logits(x)[0], mask)[0]
        return {o: float(p[i]) for i, o in enumerate(sim.OPTIONS) if mask[0, i]}, {}

    def save(self, path, meta):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "options": sim.OPTIONS,
                       "w1": self.w1.tolist(), "b1": self.b1.tolist(),
                       "w2": self.w2.tolist(), "b2": self.b2.tolist()}, f)

    @staticmethod
    def load(path):
        d = json.load(open(path, encoding="utf-8"))
        return Student(*(np.array(d[k], dtype=np.float32) for k in ("w1", "b1", "w2", "b2")))


def metrics(student, x, mask, t):
    p = masked_softmax(student.logits(x)[0], mask)
    top1 = float((p.argmax(1) == t.argmax(1)).mean())
    kl = float(np.mean(np.sum(np.where(t > 0, t * (np.log(t + 1e-12) - np.log(p + 1e-12)), 0), axis=1)))
    conf, correct = p.max(1), p.argmax(1) == t.argmax(1)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = float(sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
                    for b in range(10) if (bins == b).any()))
    illegal_mass = float((masked_softmax(student.logits(x)[0], np.ones_like(mask)) * ~mask).sum(1).mean())
    return {"top1_vs_teacher_argmax": round(top1, 4), "kl": round(kl, 4), "ece": round(ece, 4),
            "pre_mask_illegal_mass": round(illegal_mass, 4)}


def train(args):
    x, mask, t, split = load(args.data, args.label)
    tr, va = split == "train", split == "val"
    rng = np.random.default_rng(args.seed)
    st = Student.init(x.shape[1], args.hidden, len(sim.OPTIONS), rng)
    params = [st.w1, st.b1, st.w2, st.b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step, idx = 0, np.where(tr)[0]
    for epoch in range(args.epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            z, h = st.logits(x[b])
            # Unmasked softmax so the model also learns to keep illegal options near zero.
            p = masked_softmax(z, np.ones_like(mask[b]))
            dz = (p - t[b]) / len(b)
            grads = [x[b].T @ ((dz @ st.w2.T) * (h > 0)), ((dz @ st.w2.T) * (h > 0)).sum(0), h.T @ dz, dz.sum(0)]
            grads[0] += args.wd * st.w1
            grads[2] += args.wd * st.w2
            step += 1
            for k, (pp, g) in enumerate(zip(params, grads)):
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                pp -= args.lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            print(f"epoch {epoch + 1}: val {metrics(st, x[va], mask[va], t[va])}")
    test = split == "test"
    result = {"train_rows": int(tr.sum()), "val": metrics(st, x[va], mask[va], t[va]),
              "test": metrics(st, x[test], mask[test], t[test]),
              "params": int(sum(p.size for p in params))}
    st.save(args.out, {"label": args.label, "hidden": args.hidden, "schema": features.SCHEMA,
                       "renderer": features.RENDERER_VERSION, **result})
    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="out/boss_tactic_v1.jsonl")
    t.add_argument("--out", default="out/mlp_v1.json")
    t.add_argument("--label", default="pikl", choices=["pikl", "fsm", "llm"])
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
