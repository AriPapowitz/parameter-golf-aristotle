# PR #1218 Port Plan

## Target Stack (conceptual, from PR #1218 direction)

| Component | Target Value |
|---|---|
| Vocab size | 4096 |
| Model class | ~32M params |
| MLP multiplier | 4 |
| `qk_gain_init` | 4.0 |
| Muon weight decay | ~0.085 (compression lever) |
| Data loader | coprime multi-shard |
| Quantization | GPTQ (already present) |
| Compressor | brotli + byte-shuffle |
| Extras removed | TTT, QAT, BigramHash, SmearGate, VE (not needed) |

---

## Current Status

- **Branch:** `pr1218_port`
- **Stage 1:** ✅ brotli compressor + byte-shuffle exporter (env-controlled, backward-compatible)
- **Stage 2:** ⬜ coprime multi-shard loader
- **Stage 3:** ⬜ architecture / training knobs (MLP×4, qk_gain 4, vocab 4096, Muon WD)
- **Stage 4:** ⬜ 4096 vocab migration (tokenizer + embedding)
- **Stage 5:** ⬜ full PR1218-style end-to-end run + submission

---

## Staged Migration Checklist

### Stage 1 — Exporter / Backend Prep ✅

- [x] Add `COMPRESSOR=brotli` support with clear ImportError on missing package
- [x] Add `BYTE_SHUFFLE=1/0` env var
- [x] Implement `byte_shuffle_encode` / `byte_shuffle_decode` (stride-4 transpose)
- [x] Add `_assert_byte_shuffle_roundtrip()` internal sanity check
- [x] Run sanity check before compression when `BYTE_SHUFFLE=1`
- [x] Thread `shuffle` flag through `compress_bytes` / `decompress_bytes`
- [x] Artifact filename includes `_shuf` tag when shuffle enabled
- [x] Log compressor name and byte_shuffle flag at export time
- [x] Existing `zlib`/`lzma`/`zstd` paths unaffected
- [x] `python -m py_compile train_gpt.py` passes

### Stage 2 — Loader Port ⬜

- [ ] Audit current `DistributedTokenLoader` / `TokenStream`
- [ ] Implement coprime multi-shard loader (multiple independent shard streams
      with coprime step sizes so each rank reads a different interleaving)
- [ ] Keep existing single-shard loader as fallback (`LOADER=legacy`)
- [ ] Smoke test: confirm token counts are reproducible across ranks
- [ ] Push and verify on pod

### Stage 3 — Architecture / Training Knobs ⬜

- [ ] `MLP_MULT=4` (currently defaulting to 2)
- [ ] `QK_GAIN_INIT=4.0` (currently 1.5)
- [ ] Muon weight decay: add `MUON_WD` env var (default 0.0 to not break existing runs)
- [ ] Confirm model param count with new defaults (~32M target)
- [ ] Confirm training step throughput on 1×H100
- [ ] Push and smoke test

### Stage 4 — 4096 Vocab Migration ⬜

- [ ] Prepare/download 4096-vocab SentencePiece tokenizer
- [ ] Prepare/download corresponding dataset shards (fineweb 4096-vocab splits)
- [ ] Set `VOCAB_SIZE=4096` default in Hyperparameters (keep current 1024 as override)
- [ ] Confirm embedding size accounting in submission size calculation
- [ ] Full validation run at 4096 vocab

### Stage 5 — Full PR1218-Style Run ⬜

- [ ] Set all knobs: vocab=4096, MLP×4, qk_gain=4, Muon WD, coprime loader
- [ ] `EXPORT_BITS=6`, `COMPRESSOR=brotli`, `BYTE_SHUFFLE=1`, GPTQ on mlp+attn
- [ ] 600s run on 1×H100, confirm val_bpb in target range
- [ ] Check final artifact size (target: competitive with #1218 submission)
- [ ] Git tag `pr1218_v1`, push to origin, document result

---

## Implementation Order Rationale

1. **Exporter first** — backend changes are pure additions, zero training risk, smoke-testable immediately.
2. **Loader second** — new loader is independent of model shape, easier to isolate bugs.
3. **Arch knobs third** — once loader is stable, scale up model; verify throughput before vocab change.
4. **Vocab last** — requires matching tokenizer + dataset; most disruptive change, save for when everything else works.
5. **Full run fifth** — assemble all pieces only after each stage validates.

---

## Notes

- All Stage 1 changes are guarded behind env vars; defaults preserve existing behavior.
- GPTQ calibration and mixed-precision export (int6/int8 by pattern) are already present and carry forward unchanged.
- Compressor `brotli` at `quality=11` typically beats `zstd` level 19 by 3-8% on weight blobs.
- `byte_shuffle` (stride-4 float byte transposition) typically saves an additional 5-12% on top of the compressor.
