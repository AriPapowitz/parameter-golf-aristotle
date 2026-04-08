# Canonical Control + Stage 1 Runbook

## Baseline control command (600s)

Run this exact command with only `SEED` and `RUN_ID` changed:

```bash
RUN_ID=control_seed42 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
EXPORT_BITS=6 \
COMPRESSOR=brotli \
BYTE_SHUFFLE=1 \
GPTQ_ENABLE=1 \
GPTQ_TARGETS=mlp,attn \
GPTQ_CALIB_MODE=train \
GPTQ_CALIB_BATCHES=64 \
GPTQ_MAX_ROWS=12000 \
GPTQ_BLOCK_SIZE=128 \
GPTQ_DAMPING=0.0002 \
QUANT_BITS_BY_PATTERN="attn.c_q:8,attn.c_k:8,attn.c_v:8,attn.proj:8,mlp.fc:6,mlp.proj:8" \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Second control is identical except:
- `RUN_ID=control_seed314`
- `SEED=314`

## Stage 1 GPTQ grid bundle

Keep the baseline command and override one variable at a time:

```bash
# A) rows sweep
RUN_ID=s1_rows8k   GPTQ_MAX_ROWS=8000   SEED=42 ...
RUN_ID=s1_rows12k  GPTQ_MAX_ROWS=12000  SEED=42 ...
RUN_ID=s1_rows16k  GPTQ_MAX_ROWS=16000  SEED=42 ...

# B) calib batches
RUN_ID=s1_calib64  GPTQ_CALIB_BATCHES=64 SEED=42 ...
RUN_ID=s1_calib96  GPTQ_CALIB_BATCHES=96 SEED=42 ...

# C) block size
RUN_ID=s1_blk128 GPTQ_BLOCK_SIZE=128 SEED=42 ...
RUN_ID=s1_blk96  GPTQ_BLOCK_SIZE=96  SEED=42 ...

# D) damping
RUN_ID=s1_damp1e4 GPTQ_DAMPING=0.0001 SEED=42 ...
RUN_ID=s1_damp2e4 GPTQ_DAMPING=0.0002 SEED=42 ...

# E) mlp.proj int8 vs int6
RUN_ID=s1_mlp_proj8 QUANT_BITS_BY_PATTERN="attn.c_q:8,attn.c_k:8,attn.c_v:8,attn.proj:8,mlp.fc:6,mlp.proj:8" ...
RUN_ID=s1_mlp_proj6 QUANT_BITS_BY_PATTERN="attn.c_q:8,attn.c_k:8,attn.c_v:8,attn.proj:8,mlp.fc:6,mlp.proj:6" ...
```

Use the command prefix from baseline; replace `...` with the rest of that baseline command.

## Metrics to archive per run

- `step:... val_bpb:...` (best pre-quant)
- `final_int*_roundtrip_exact val_bpb:...`
- `quant_summary ... worst_rel_mse_top10:...`
- `Total submission size ...`
