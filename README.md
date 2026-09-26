# Xuwang-Director-1.7B · 虚妄决策模型

[中文说明](README.zh-CN.md) | English

**A local game director, not a chatbot.**

**不是让 AI 替我们做游戏，而是为游戏训练一个导演。**

Xuwang-Director-1.7B is a task-specific game decision model built from Qwen3-1.7B-Base using LoRA and 193 typed output heads. It scores decisions across ten game-director channels without autoregressive text generation. The game supplies constraints, validation and fallback; the model supplies preferences.

虚妄是为战棋卡牌游戏训练的本地 AI 导演。它不输出聊天文本，而是根据游戏状态，为难度调整、遭遇配置、卡牌奖励、Boss 战术和音乐编排等任务输出选项概率。运行时无需第三方云端 API Key；下载模型、安装依赖可能需要联网。

This is not a Jev implementation or a claim about Jev's internal architecture. Typed decision interfaces are a related design direction; this model uses a documented Qwen backbone and multi-task linear heads.

## Release contents

Code and documentation are in this repository. **Large model/data files are attached to [GitHub Releases](https://github.com/zheznanohana/Xuwang-Director-1.7B/releases)** rather than committed to Git history:

| Asset | Purpose |
|---|---|
| `Xuwang-Director-1.7B-Q4_K_M.gguf` | Merged, quantized step-2000 backbone (1,107,408,416 bytes) |
| `heads.json` | All 193 learned heads and learned feature scale; required for inference |
| `adapter-step2000.zip` | LoRA adapter and tokenizer; not a standalone decision model |
| `sft_qwen_v2.jsonl` | Teacher-ensemble dataset: 25,951 train / 4,512 validation / 4,481 test rows |
| `step_2000.pt`, `step_2800.pt`, `step_3000.pt` | Research checkpoints retained from training |
| `bench_1p7b.zip` | Offline evaluation reports and row-level evaluation results |
| `SHA256SUMS.txt` | Asset integrity checksums |

The source game, its art/audio, credentials, cloud account configuration and private development logs are **not** part of this release. Dataset test rows are now public: do not treat them as a secret benchmark for future models.

## Architecture

```text
Structured game state
  -> exact channel-specific prompt (ending in <decide>)
  -> Qwen3-1.7B-Base + merged LoRA
  -> last-token hidden state, L2 normalization, learned scale
  -> one linear head per (channel, field)
  -> softmax choices / sigmoid multi-label probabilities
  -> legal masks + game-specific search, validation and fallback
```

The 193 heads cover boss tactics, difficulty adaptation, nemesis design, fate analysis, encounter procurement, card procurement, music scene, music score, camp assessment and encounter flavour. See `schema.json` for exact option vocabularies.

**Type correctness is not joint validity.** A set of legal field values can still violate budgets, compatibility rules or cross-field constraints. The minimal Python demo returns raw head outputs, NOT a complete validated game proposal. The game's policy search and complete engine validators are not packaged as a standalone engine in this release.

## Quick start: local inference

1. Download the GGUF and `heads.json` from Releases into `models/`.
2. Install Python 3.10+ and `pip install -r requirements.txt`.
3. Obtain a compatible [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` build. Run (adjust executable path for your OS):

```sh
llama-server -m models/Xuwang-Director-1.7B-Q4_K_M.gguf --embeddings --pooling last --host 127.0.0.1 --port 18739 -c 1024 -b 1024 -ub 1024
```

4. In a second terminal:

```sh
python infer.py --heads models/heads.json --example examples/camp.json
```

The example returns four camp-assessment probability distributions and selections, plus `joint_constraints_checked: false`. It sends requests only to the local server. For GPU offload, add `-ngl 99` to a GPU-capable server build. Performance depends on backend, hardware, prompt length and concurrency.

Two raw-context renderers are included in the minimal demo (camp and flavour). For other channels, supply an exactly rendered `prompt`, `channel`, and optional boolean `legal_masks` in the example JSON. Full training-side renderers are in `training/export_sft.py`; arbitrary chat prompts are out of distribution.

Without weights, inspect the prompt and run unit tests:

```sh
python infer.py --heads models/heads.json --render-only
python -m unittest discover -s tests -v
```

## Reported results (historical, not a new benchmark run)

| Metric | Result | Meaning |
|---|---:|---|
| Offline test macro agreement | 0.7719 | Choice agreement / multi-label F1 against teacher ensemble, 4,481 rows |
| Frozen backbone + heads | 0.7273 | Same offline comparison |
| Feature MLP variants | 0.7648–0.7657 | Smaller baselines; advantage is modest |
| Q4 vs FP16 field agreement | 0.9842 | Quantization fidelity, 300-row sample; **not task accuracy** |
| Q4 sampled teacher score | 0.7672 | Separate 300-row quantization sample |

Full offline statistics and quantization results are in `evaluation/`. The reported Q4 CPU latency was p50 894.6 ms / p95 3457.4 ms with 8 threads; 4-thread p95 was 6013.6 ms. These are historical setup-specific measurements, not minimum hardware guarantees or end-to-end gameplay timings.

Game-harness results in the associated manuscript distinguish per-field argmax (30% raw encounter validity, 88% raw card validity), budget-aware single-candidate readout (50% raw encounter validity) and policy search (50% raw encounter validity, 100% acceptance after host repair), on 40 core contexts per relevant channel. Those results depend on the game policy and validator, not just this minimal runner. Information-matched chat baselines reached higher raw encounter validity in that experiment. These measurements do not establish better human enjoyment or universal model superiority.

## Training and reproduction

Training: LoRA rank 16, alpha 32, dropout 0.05, seven projection types; soft-label choice cross-entropy and multi-label BCE; per-channel field-count normalization. Historical training used 3,244 steps, effective batch 16 and sequence length 512. The selected checkpoint is step 2000, trained on September 25, 2026. Later folder timestamps reflect file cleanup, not retraining.

Teachers: game rules, a provider-hosted decision model, Qwen chat labels, and piKL search for boss states. The dataset records teacher tags per row. Jev is an evaluation comparator, not a listed training teacher. See `DATA_CARD.md` and `MODEL_CARD.md`.

`training/train_qwen.py` trains from the released JSONL. `training/merge_adapter.py` merges an adapter with the base. `training/bench_ckpts.py`, `probe_and_mlp.py`, and supporting files are research scripts. Install training dependencies separately (PyTorch, Transformers, PEFT, safetensors, NumPy and scikit-learn as applicable); the exact original environment is not fully pinned. Script defaults include older experiment paths: pass your own paths and `--base Qwen/Qwen3-1.7B-Base`. Reproduction is not claimed bit-for-bit. Inspect checkpoints with `torch.load(..., weights_only=True)` where supported; do not load untrusted pickle checkpoints.

## Licensing and scope

Original code and documentation in this repository: **Apache-2.0**, see `LICENSE`.

The base model is [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base), published under Apache-2.0. See `MODEL_TERMS.md` for attribution and the scope of this repository's code license.

This model is specialized to the source game's schemas, options and state distribution. New games, new vocabularies and new balance rules generally require new heads, retraining and fresh validation. No affiliation with Qwen, Alibaba, TypeSafe or Jev is implied.
