# Release validation: 2026-09-27

- 9/9 Python unit tests passed.
- Published runner exercised against the staged Q4 GGUF via local Vulkan llama-server.
- Both synthetic examples returned typed fields; the documented CLI also passed.
- All 392 adapter tensors exactly match the selected step-2000 state.
- All exported head weights/biases and learned scale match that state (max absolute difference 0).
- SHA-256 hashes recorded for all release assets.
- Focused credential-pattern scan found no matches in publication source files or training JSONL. This is not a full security audit.

See evaluation/release_smoke.json for measured results. Smoke timings are not a performance benchmark. GGUF was not re-exported from the adapter during this release. No full test-set rerun is claimed.
