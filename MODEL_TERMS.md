# Attribution and license scope

## Original implementation

Original source code and documentation in this repository are released under Apache-2.0; see LICENSE. That code license applies to source code and documentation, not as a blanket additional license grant for all model/data release assets.

## Base model

The adapter and GGUF derive from Qwen/Qwen3-1.7B-Base, whose upstream model card identifies Apache-2.0. A copy is included in third_party/QWEN_BASE_LICENSE. Changes comprise LoRA fine-tuning, merging, and Q4_K_M quantization. The selected checkpoint is step 2000. Preserve applicable upstream attribution and license notices.

https://huggingface.co/Qwen/Qwen3-1.7B-Base

## Training provenance

Targets combine game rules, a provider-hosted decision model, Qwen chat labels and piKL search. Dataset rows retain teacher tags. See DATA_CARD.md. The repository's code license does not alter third-party rights or service agreements.

## Scope

Game art, music assets, story scripts, credentials, private logs and the complete game are outside this release. No affiliation with or endorsement by Qwen, Alibaba, TypeSafe or Jev is implied.
