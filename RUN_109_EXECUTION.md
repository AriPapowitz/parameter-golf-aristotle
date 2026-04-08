# Path to ~1.09 Execution Sheet

This file operationalizes the staged roadmap into concrete command bundles.

## Stage 2 — Coprime loader validation

Run baseline command from `RUN_STAGE1_BROTLI.md` and only override:

```bash
RUN_ID=s2_loader_seq TRAIN_LOADER_MODE=sequential LOADER_COPRIME_STRIDE=7 SEED=42 ...
RUN_ID=s2_loader_c7  TRAIN_LOADER_MODE=coprime   LOADER_COPRIME_STRIDE=7 SEED=42 ...
RUN_ID=s2_loader_c11 TRAIN_LOADER_MODE=coprime   LOADER_COPRIME_STRIDE=11 SEED=42 ...
```

Promote if pre-quant `val_bpb` improves >= 0.01 or quantized improves >= 0.008 at same wallclock.

## Stage 3 — Architecture / optimizer knobs

One-variable protocol:

```bash
# MLP multiplier
RUN_ID=s3_mlp2 MLP_MULT=2 SEED=42 ...
RUN_ID=s3_mlp4 MLP_MULT=4 SEED=42 ...

# QK gain
RUN_ID=s3_qk15 QK_GAIN_INIT=1.5 SEED=42 ...
RUN_ID=s3_qk4  QK_GAIN_INIT=4.0 SEED=42 ...

# Muon weight decay
RUN_ID=s3_wd002 MUON_WEIGHT_DECAY=0.02 SEED=42 ...
RUN_ID=s3_wd004 MUON_WEIGHT_DECAY=0.04 SEED=42 ...
RUN_ID=s3_wd006 MUON_WEIGHT_DECAY=0.06 SEED=42 ...
RUN_ID=s3_wd0085 MUON_WEIGHT_DECAY=0.085 SEED=42 ...

# Moderate model scale
RUN_ID=s3_scale_a NUM_LAYERS=10 MODEL_DIM=544 NUM_HEADS=8 NUM_KV_HEADS=4 SEED=42 ...
RUN_ID=s3_scale_b NUM_LAYERS=11 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 SEED=42 ...
```

## Stage 4 — Fuller GPTQ quality path

```bash
# Training-data calibration + numerical upgrades
RUN_ID=s4_gptq_train \
GPTQ_ENABLE=1 GPTQ_CALIB_MODE=train GPTQ_ORDER_BY_DIAG=1 GPTQ_HESSIAN_FP64=1 \
GPTQ_BLOCK_SIZE=96 GPTQ_DAMPING=0.0002 SEED=42 ...

# AR self-generated calibration (optional legality path)
RUN_ID=s4_gptq_ar \
GPTQ_ENABLE=1 GPTQ_CALIB_MODE=ar GPTQ_AR_BATCH_SIZE=8 GPTQ_AR_SEQ_LEN=256 \
GPTQ_ORDER_BY_DIAG=1 GPTQ_HESSIAN_FP64=1 SEED=42 ...
```

Promote only if quantized `val_bpb` gain >= 0.01.

## Stage 5 — 4096 vocab migration

```bash
RUN_ID=s5_4096_base \
USE_VOCAB4096=1 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
SEED=42 ...

RUN_ID=s5_4096_scaled \
USE_VOCAB4096=1 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
NUM_LAYERS=10 MODEL_DIM=576 MLP_MULT=4 MUON_WEIGHT_DECAY=0.06 \
SEED=42 ...
```

Target total size window: ~15.8MB to ~15.95MB.

## Stage 6 — Final 3-seed convergence

For top 2 candidates, run each seed:

```bash
RUN_ID=final_candA_s42  SEED=42  ...
RUN_ID=final_candA_s314 SEED=314 ...
RUN_ID=final_candA_s999 SEED=999 ...

RUN_ID=final_candB_s42  SEED=42  ...
RUN_ID=final_candB_s314 SEED=314 ...
RUN_ID=final_candB_s999 SEED=999 ...
```

Winner rule: lowest mean quantized `val_bpb`, tie-break by variance then smaller artifact.
