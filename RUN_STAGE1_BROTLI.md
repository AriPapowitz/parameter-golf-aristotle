# Stage 1 — Brotli + Byte-Shuffle Smoke Test

## 1. Local commit and push

```bash
git add train_gpt.py PORT_PR1218_PLAN.md RUN_STAGE1_BROTLI.md
git commit -m "Stage 1: brotli compressor + byte-shuffle exporter (pr1218_port)"
git push -u origin pr1218_port
```

---

## 2. Pod: pull the branch

```bash
cd ~/parameter-golf
git fetch origin
git checkout pr1218_port
git pull origin pr1218_port
pip install brotli          # only needed for COMPRESSOR=brotli
# pip install zstandard     # already installed if using zstd elsewhere
```

---

## 3. Smoke test — 120 s, existing 1024-vocab dataset, brotli + byte-shuffle

```bash
RUN_ID=stage1_brotli_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=0 \
EXPORT_BITS=6 \
COMPRESSOR=brotli \
BYTE_SHUFFLE=1 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Expected log lines to look for

```
Serialized model int6+brotli byte_shuffle:True: ... bytes ...
```

Artifact file will be named: `final_model.int6.brotli_shuf.ptz`

---

## 4. Verification greps (run from repo root)

```bash
# Confirm brotli branch exists in compress_bytes
grep -n "brotli" train_gpt.py

# Confirm byte_shuffle functions exist
grep -n "byte_shuffle" train_gpt.py

# Confirm shuffle flag is threaded through compress call
grep -n "shuffle=args.byte_shuffle" train_gpt.py

# Confirm sanity check is called before export
grep -n "_assert_byte_shuffle_roundtrip" train_gpt.py

# Confirm Hyperparameter exists
grep -n "BYTE_SHUFFLE" train_gpt.py

# Confirm plan file is present
ls PORT_PR1218_PLAN.md RUN_STAGE1_BROTLI.md
```

---

## 5. Quick sanity — zlib baseline (verify no regression)

```bash
RUN_ID=stage1_zlib_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=0 \
EXPORT_BITS=6 \
COMPRESSOR=zlib \
BYTE_SHUFFLE=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

This should behave identically to the previous branch (artifact: `final_model.int6.zlib.ptz`).

---

## 6. Notes

- If `brotli` is not installed you will see a clear error:
  `ImportError: brotli compressor requires the 'brotli' package: pip install brotli`
- The `byte_shuffle` roundtrip sanity check fires once before compression; it uses `os.urandom`
  on 1031 bytes (odd length to exercise padding). If it fails, the run aborts immediately.
- `BYTE_SHUFFLE=1` can be combined with any compressor (`zlib`, `lzma`, `zstd`, `brotli`).
