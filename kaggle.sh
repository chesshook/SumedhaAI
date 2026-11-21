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
echo "[2/5] Installing chess dependencies (PyTorch is left untouched)..."
python -m pip install --disable-pip-version-check -q -r requirements.txt

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export SUMEDHA_DATA_DIR="${SUMEDHA_DATA_DIR:-/kaggle/working/sumedha-data}"
export SUMEDHA_OUTPUT_DIR="${SUMEDHA_OUTPUT_DIR:-/kaggle/working/sumedha-output}"
mkdir -p "$SUMEDHA_DATA_DIR" "$SUMEDHA_OUTPUT_DIR"
exec > >(tee -a "$SUMEDHA_OUTPUT_DIR/training.log") 2>&1

# Restore outputs attached from an earlier saved Kaggle Notebook version.
if [[ ! -s "$SUMEDHA_DATA_DIR/positions.npz" ]]; then
  PREVIOUS_DATA="$(find /kaggle/input -type f -path '*/sumedha-data/positions.npz' -print -quit 2>/dev/null || true)"
  if [[ -n "$PREVIOUS_DATA" ]]; then
    echo "Restoring the previous labelled dataset..."
    cp "$PREVIOUS_DATA" "$SUMEDHA_DATA_DIR/positions.npz"
  fi
fi
if [[ ! -s "$SUMEDHA_OUTPUT_DIR/latest.pt" ]]; then
  PREVIOUS_CHECKPOINT="$(find /kaggle/input -type f -path '*/sumedha-output/latest.pt' -print -quit 2>/dev/null || true)"
  if [[ -n "$PREVIOUS_CHECKPOINT" ]]; then
    echo "Restoring the previous SUMEDHA checkpoint..."
    cp "$PREVIOUS_CHECKPOINT" "$SUMEDHA_OUTPUT_DIR/latest.pt"
    PREVIOUS_OUTPUT_DIR="$(dirname "$PREVIOUS_CHECKPOINT")"
    if [[ -s "$PREVIOUS_OUTPUT_DIR/best.pt" ]]; then
      cp "$PREVIOUS_OUTPUT_DIR/best.pt" "$SUMEDHA_OUTPUT_DIR/best.pt"
    fi
  fi
fi
exec > >(tee -a "$SUMEDHA_OUTPUT_DIR/training.log") 2>&1

echo "[3/5] Checking the Kaggle runtime..."
python - <<'PY'
import os, sys, torch
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible CUDA devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} | {p.total_memory / 1024**3:.1f} GiB")
if not torch.cuda.is_available():
    raise SystemExit("SUMEDHA needs a CUDA GPU. Select GPU T4 x2 in Kaggle Notebook settings.")
PY
python -m sumedha.selftest

DATA_FILE="$SUMEDHA_DATA_DIR/positions.npz"
if [[ "${SUMEDHA_REBUILD_DATA:-0}" == "1" ]]; then
  echo "SUMEDHA_REBUILD_DATA=1: rebuilding labelled data from current inputs/settings."
  rm -f "$DATA_FILE"
fi
if [[ -s "$DATA_FILE" ]]; then
  echo "[4/5] Existing dataset found; reusing it: $DATA_FILE"
else
  echo "[4/5] Building Stockfish-labelled chess positions..."
  python -m sumedha.data \
    --output "$DATA_FILE" \
    --games "${SUMEDHA_GAMES:-300}" \
    --max-plies "${SUMEDHA_MAX_PLIES:-80}" \
    --play-nodes "${SUMEDHA_PLAY_NODES:-250}" \
    --nodes "${SUMEDHA_NODES:-8000}" \
    --multipv "${SUMEDHA_MULTIPV:-6}" \
    --opening-positions "${SUMEDHA_OPENING_POSITIONS:-3000}" \
    --pgn-positions "${SUMEDHA_PGN_POSITIONS:-5000}" \
    --pgn-game-limit "${SUMEDHA_PGN_GAME_LIMIT:-10000}" \
    --max-positions "${SUMEDHA_MAX_POSITIONS:-16000}" \
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
echo "To preserve these files, save a Kaggle Notebook version with outputs enabled."
