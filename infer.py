"""Xuwang Director: typed probabilities, not text generation. Apache-2.0."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from prompts import render_camp, render_flavor


def extract_embedding(node):
    if isinstance(node, dict) and "data" in node:
        node = node["data"]
    if isinstance(node, list) and node:
        node = node[0]
    if isinstance(node, dict):
        node = node.get("embedding", [])
    if isinstance(node, list) and node and isinstance(node[0], list):
        node = node[-1]
    h = np.asarray(node, dtype=np.float64)
    if h.ndim != 1 or not h.size or not np.isfinite(h).all():
        raise ValueError("Invalid embedding")
    return h


def apply_heads(heads, channel, hidden, masks=None):
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim != 1 or not h.size or not np.isfinite(h).all():
        raise ValueError("Invalid hidden state")
    norm = np.linalg.norm(h)
    if norm <= 1e-12:
        raise ValueError("Zero hidden state")
    scale = float(heads["__scale__"])
    if not np.isfinite(scale):
        raise ValueError("Invalid head scale")
    h = h / norm * scale
    out = {}
    for key, spec in heads.items():
        if not key.startswith(channel + "/"):
            continue
        field = key.split("/", 1)[1]
        w, b = np.asarray(spec["weight"]), np.asarray(spec["bias"])
        options = spec["options"]
        if w.shape != (len(options), h.size) or b.shape != (len(options),):
            raise ValueError("Head shape mismatch: " + key)
        logits = w @ h + b
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite logits: " + key)
        raw_mask = (masks or {}).get(field, [True] * len(options))
        if not isinstance(raw_mask, list) or any(type(x) is not bool for x in raw_mask):
            raise ValueError("Masks must contain JSON booleans: " + key)
        legal = np.asarray(raw_mask, dtype=bool)
        if legal.shape != logits.shape or not legal.any():
            raise ValueError("Empty or mismatched legal mask: " + key)
        if spec["kind"] == "choice":
            logits = np.where(legal, logits, -np.inf)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            selection = options[int(probs.argmax())]
        elif spec["kind"] == "multi":
            probs = np.exp(-np.logaddexp(0, -logits))
            probs = np.where(legal, probs, 0)
            selection = [o for o, p in zip(options, probs) if p >= 0.5]
        else:
            raise ValueError("Unknown head kind: " + key)
        out[field] = {"kind": spec["kind"], "options": options,
                      "probabilities": probs.tolist(), "selection": selection}
    if not out:
        raise ValueError("Unknown channel: " + channel)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heads", required=True, type=Path)
    ap.add_argument("--example", type=Path, default=Path(__file__).parent / "examples/camp.json")
    ap.add_argument("--server", default="http://127.0.0.1:18739")
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    example = json.loads(args.example.read_text(encoding="utf-8"))
    channel = example["channel"]
    renderers = {"camp_assessment@1": render_camp, "encounter_flavor@1": render_flavor}
    prompt = example.get("prompt")
    if prompt is None:
        prompt = renderers[channel](example["context"])
    if args.render_only:
        print(prompt)
        return
    url = urlparse(args.server)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("This example connects only to a local HTTP inference server")
    req = Request(args.server.rstrip("/") + "/embedding",
                  data=json.dumps({"content": prompt}).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as response:
        hidden = extract_embedding(json.load(response))
    heads = json.loads(args.heads.read_text(encoding="utf-8"))
    result = apply_heads(heads, channel, hidden, example.get("legal_masks"))
    print(json.dumps({"channel": channel, "fields": result,
                     "joint_constraints_checked": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
