"""Merge a trained LoRA adapter into its base model on CPU (no GPU needed) for GGUF conversion.

  python merge_adapter.py --run /path/to/runs/qwen3_1p7b --out /content/qwen_1p7b_best

Copies heads.json + metrics.json next to the merged model so quant_bench.py / deploy can use the dir.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with adapter/, heads.json, metrics.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    metrics = json.load(open(os.path.join(args.run, "metrics.json"), encoding="utf-8"))
    base = metrics.get("base", "Qwen/Qwen3-1.7B-Base")
    os.makedirs(args.out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, os.path.join(args.run, "adapter")).merge_and_unload()
    model.to(torch.float16).save_pretrained(os.path.join(args.out, "merged"))
    AutoTokenizer.from_pretrained(os.path.join(args.run, "adapter")).save_pretrained(os.path.join(args.out, "merged"))
    for f in ("heads.json", "metrics.json"):
        shutil.copy2(os.path.join(args.run, f), os.path.join(args.out, f))
    print("merged", base, "+", os.path.join(args.run, "adapter"), "->", args.out)


if __name__ == "__main__":
    main()
