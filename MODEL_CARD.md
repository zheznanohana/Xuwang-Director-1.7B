# Xuwang-Director-1.7B

- Publisher: zheznanohana
- Base: Qwen/Qwen3-1.7B-Base
- Architecture: decoder backbone + 193 per-field linear heads; one forward pass, no generated answer text
- Adaptation: LoRA r=16, alpha=32, dropout=0.05; step 2000 selected
- Training date: 2026-09-25 (owner's records)
- Languages: compact English prompts, game-specific option identifiers
- Input: exact channel-specific serialization, ending in `<decide>`
- Hidden representation: last token, L2 normalized, learned scale 8.69564437866211
- Outputs: softmax distributions for choice fields; sigmoid scores for multi-label fields
- Quantization: Q4_K_M GGUF; 1,107,408,416 bytes; 1.7B denotes base parameter scale, not file size
- Intended use: research and integration of local typed game directors within this task distribution
- Not intended as: a chat assistant, general game-playing agent, unconstrained story generator, or drop-in director for arbitrary games
- Code license: Apache-2.0
- Base model license and attribution: see MODEL_TERMS.md

## Evaluation and limits

The historical full test score is 0.7719 macro teacher agreement (95% CI 0.7659–0.7777); frozen features + heads score 0.7273. The four feature MLP variants score 0.7648–0.7657. These measure teacher fidelity, not objective decision correctness, human enjoyment or player retention.

The historical quantization study uses 300 rows: Q4 field agreement with the full-precision reference is 0.9842 and sampled teacher score is 0.7672. Sampled reference and Q4 teacher scores use different aggregation in the original script; do not interpret their small difference as an exact paired quantization loss.

Raw head choices need per-context legal masks and host-level joint validation. The public minimal demo does not run the game's full budget/compatibility validator or policy search. Independent heads do not guarantee a jointly valid assignment. Confidence is not guaranteed calibrated. New game balance or input formats can cause distribution shift.

See evaluation/ for historical reports and LOCAL_VALIDATION.md for the separately dated release smoke test. No new full test-set evaluation is implied by packaging this release.
