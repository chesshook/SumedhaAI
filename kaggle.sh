#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="https://github.com/chesshook/SumedhaAI.git"
ROOT="/kaggle/working/SumedhaAI"

echo "============================================================"
echo "                 SUMEDHA AI — KAGGLE"
echo "============================================================"

if [[ -d "$ROOT/.git" ]]; then
  echo "[1/5] Updating existing SUMEDHA checkout..."
  git -C "$ROOT" pull --ff-only
else
  echo "[1/5] Fetching SUMEDHA AI from GitHub..."
  git clone --depth 1 "$REPO_URL" "$ROOT"
fi

cd "$ROOT"
echo "[2/5] Installing data dependencies (PyTorch is left untouched)..."
python -m pip install --disable-pip-version-check -q -r requirements.txt

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export SUMEDHA_DATA_DIR="${SUMEDHA_DATA_DIR:-/kaggle/working/sumedha-data-v2}"
export SUMEDHA_OUTPUT_DIR="${SUMEDHA_OUTPUT_DIR:-/kaggle/working/sumedha-output-v2}"
mkdir -p "$SUMEDHA_DATA_DIR" "$SUMEDHA_OUTPUT_DIR"
exec > >(tee -a "$SUMEDHA_OUTPUT_DIR/training.log") 2>&1

# Restore only v2 artefacts. Old live-Stockfish v1 caches are deliberately ignored.
if [[ ! -s "$SUMEDHA_DATA_DIR/positions_v2.npz" ]]; then
  PREVIOUS_DATA="$(find /kaggle/input -type f -path '*/sumedha-data-v2/positions_v2.npz' -print -quit 2>/dev/null || true)"
  if [[ -n "$PREVIOUS_DATA" ]]; then
    echo "Restoring the v2 precomputed dataset from a saved Kaggle output..."
    cp "$PREVIOUS_DATA" "$SUMEDHA_DATA_DIR/positions_v2.npz"
  fi
fi
if [[ ! -s "$SUMEDHA_OUTPUT_DIR/latest.pt" ]]; then
  PREVIOUS_CHECKPOINT="$(find /kaggle/input -type f -path '*/sumedha-output-v2/latest.pt' -print -quit 2>/dev/null || true)"
  if [[ -n "$PREVIOUS_CHECKPOINT" ]]; then
    echo "Restoring the v2 SUMEDHA checkpoint..."
    cp "$PREVIOUS_CHECKPOINT" "$SUMEDHA_OUTPUT_DIR/latest.pt"
    PREVIOUS_OUTPUT_DIR="$(dirname "$PREVIOUS_CHECKPOINT")"
    if [[ -s "$PREVIOUS_OUTPUT_DIR/best.pt" ]]; then
      cp "$PREVIOUS_OUTPUT_DIR/best.pt" "$SUMEDHA_OUTPUT_DIR/best.pt"
    fi
  fi
fi

echo "[3/5] Checking the Kaggle runtime..."
python - <<'PY'
import sys, torch
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible CUDA devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} | {p.total_memory / 1024**3:.1f} GiB")
if not torch.cuda.is_available():
    raise SystemExit("SUMEDHA needs CUDA. Select GPU T4 x2 in Kaggle Notebook settings.")
PY
python -m sumedha.selftest

DATA_FILE="$SUMEDHA_DATA_DIR/positions_v2.npz"
if [[ "${SUMEDHA_REBUILD_DATA:-0}" == "1" ]]; then
  echo "SUMEDHA_REBUILD_DATA=1: rebuilding only from the fixed precomputed archives."
  rm -f "$DATA_FILE"
fi
if [[ -s "$DATA_FILE" ]]; then
  echo "[4/5] Existing v2 precomputed dataset found; reusing it: $DATA_FILE"
else
  echo "[4/5] Downloading/reading fixed precomputed chess corpora. No engine searches."
  python -m sumedha.data \
    --output "$DATA_FILE" \
    --eval-positions "${SUMEDHA_EVAL_POSITIONS:-350000}" \
    --human-positions "${SUMEDHA_HUMAN_POSITIONS:-150000}" \
    --human-games "${SUMEDHA_HUMAN_GAMES:-150000}" \
    --minimum-depth "${SUMEDHA_MINIMUM_DEPTH:-20}" \
    --minimum-knodes "${SUMEDHA_MINIMUM_KNODES:-1000}" \
    --rowgroups-per-shard "${SUMEDHA_ROWGROUPS_PER_SHARD:-32}" \
    --seed "${SUMEDHA_SEED:-20261121}"
fi

echo "[5/5] Training SUMEDHA on all visible GPUs..."
GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
COMMON_ARGS=(
  --data "$DATA_FILE"
  --output-dir "$SUMEDHA_OUTPUT_DIR"
  --epochs "${SUMEDHA_EPOCHS:-24}"
  --batch-size "${SUMEDHA_BATCH_SIZE:-128}"
  --channels "${SUMEDHA_CHANNELS:-96}"
  --blocks "${SUMEDHA_BLOCKS:-6}"
  --workers "${SUMEDHA_WORKERS:-2}"
  --seed "${SUMEDHA_SEED:-20261121}"
)

if [[ "$GPU_COUNT" -gt 1 ]]; then
  echo "Launching DistributedDataParallel across $GPU_COUNT GPUs."
  OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" -m sumedha.train "${COMMON_ARGS[@]}"
else
  echo "One GPU visible; using single-GPU training."
  python -m sumedha.train "${COMMON_ARGS[@]}"
fi

echo
echo "SUMEDHA TRAINING FINISHED"
echo "Checkpoint: $SUMEDHA_OUTPUT_DIR/latest.pt"
echo "Dataset:    $DATA_FILE"
echo "No Stockfish executable was run during data preparation or neural training."
echo "Save a Kaggle Notebook version with outputs enabled to preserve the v2 corpora and weights."
