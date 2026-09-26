# Teacher-ensemble training data

Release asset: `sft_qwen_v2.jsonl`.

34,944 rows: 25,951 training, 4,512 validation, 4,481 test. Ten channels. Each row has `channel`, `split`, `key`, `prompt`, `targets`, `teachers`. Targets contain `kind`, `options`, soft `dist` and, where defined, a boolean `legal` mask. `key` is a game-context identifier, not an API key.

The data consists of serialized game contexts and teacher-derived soft targets, not player chat histories. Teacher tags are retained for provenance. Training-side labels were legally masked and averaged over available teacher sources; boss data can include piKL search targets. See `training/export_sft.py` for the merger and serialization logic.

The source study assigns splits by context identifiers (whole fight for boss states); this release preserves the existing split fields and does not regenerate or independently certify the original split construction. The 4,481 offline-test rows are distinct in scope from the 3,785-context DirectorShift benchmark described in the manuscript. Do not equate those evaluation sets.

Limitations: game-specific schemas, mixture of rules and provider model preferences, uneven channels, no human preference labels, publicly visible test targets. Retraining on the test rows invalidates comparisons with the historical test result.

Attribution and license scope: see MODEL_TERMS.md. Raw provider responses, credentials and account configuration are not included.
