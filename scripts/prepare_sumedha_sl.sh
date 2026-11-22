#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=============================================================="
echo " SUMEDHA-SL+AB | prepare frozen Searchless 270M + search core"
echo "=============================================================="

python -m pip install --upgrade pip
python -m pip install -r requirements-sl.txt

mkdir -p third_party checkpoints
if [[ ! -d third_party/searchless_chess/.git ]]; then
  echo "[1/3] Fetching official Google DeepMind Searchless Chess source..."
  git clone --depth 1 https://github.com/google-deepmind/searchless_chess.git third_party/searchless_chess
else
  echo "[1/3] Updating official Searchless Chess source..."
  git -C third_party/searchless_chess pull --ff-only
fi

CHECKPOINT_DIR="$ROOT/checkpoints/270M"
if [[ ! -d "$CHECKPOINT_DIR/6400000/params" ]]; then
  echo "[2/3] Downloading official 270M checkpoint (large file)..."
  mkdir -p checkpoints
  curl -fL --retry 5 --retry-delay 3 \
    https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip \
    -o checkpoints/270M.zip
  unzip -oq checkpoints/270M.zip -d checkpoints
  rm -f checkpoints/270M.zip
fi

if [[ ! -d "$CHECKPOINT_DIR/6400000/params" ]]; then
  echo "ERROR: expected checkpoint path $CHECKPOINT_DIR/6400000/params"
  echo "Inspect the extracted ZIP layout before proceeding."
  exit 2
fi

echo "[3/3] Checking JAX devices and SUMEDHA search core..."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python - <<'PY'
import jax
print("JAX:", jax.__version__)
for device in jax.devices():
    print("Device:", device)
PY
python -m sumedha_sl.selftest

echo
echo "SUMEDHA-SL+AB preparation complete."
echo "Frozen model: $CHECKPOINT_DIR"
echo "Start UCI with:"
echo "  python -m sumedha_sl.uci --checkpoint-dir checkpoints/270M --vendor-dir third_party"
echo "Run model diagnostics with:"
echo "  python -m sumedha_sl.diagnose --checkpoint-dir checkpoints/270M --vendor-dir third_party"
