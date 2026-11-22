# SUMEDHA-SL+AB v0.1

This branch adds a new engine path without altering the existing SUMEDHA v2 CNN.

## What it is

- **Frozen neural foundation:** Google's official Searchless Chess 270M action-value model, step 6,400,000.
- **Inference:** official Searchless Transformer, tokenizer, action vocabulary and 128-bin return distribution.
- **Decision layer:** SUMEDHA iterative-deepening negamax, Principal Variation Search, alpha-beta, transposition table, killer/history move ordering and quiescence.
- **No Stockfish at runtime:** Stockfish is not imported, queried or launched by this engine. It may be used separately as a benchmark/opponent.

The neural checkpoint is not converted to PyTorch. It remains the original JAX/Haiku/Orbax model. Searchless source is fetched into `third_party/searchless_chess` at setup time. The pretrained checkpoint is downloaded separately.

## Prepare on Kaggle (Linux, T4 recommended)

Select **GPU T4 x2** and enable Internet. In a notebook cell:

```bash
!git clone --depth 1 -b sumedha-sl-ab-v1 https://github.com/chesshook/SumedhaAI.git /kaggle/working/SumedhaAI
%cd /kaggle/working/SumedhaAI
!bash scripts/prepare_sumedha_sl.sh
```

The setup installs only inference dependencies, fetches the official Searchless source pinned to commit `90ae0e6b121673fc3079aaeffa047580bb600c0a`, downloads the 270M checkpoint and runs the search-core self-test. The checkpoint is a large archive; confirm Kaggle working-disk space before downloading and keep the extracted checkpoint in Notebook Output if you need it in a later session.

Current JAX documentation supports CUDA 12 on NVIDIA GPUs with SM 5.2 or newer; T4 is SM 7.5. The CUDA-enabled JAX path is Linux. Native Windows GPU JAX is not supported; use Kaggle or WSL2 for NVIDIA acceleration.

## Start as a UCI engine

```bash
python -m sumedha_sl.uci --checkpoint-dir checkpoints/270M --vendor-dir third_party --hash 64
```

The UCI process loads the frozen model and responds to standard `uci`, `isready`, `position`, `go`, `stop` and `quit` commands. It is suitable for a UCI-compatible chess GUI on the same machine.

## Two-GPU inference

The current inference bridge replicates the frozen parameters on each visible GPU and explicitly shards the candidate-action batch across the GPUs using JAX `Mesh` and `NamedSharding`. This is data-parallel inference: each GPU stores a full model copy; GPU memory is not pooled. A 32-sequence batch is split across the two T4s when both are visible. This is intended to accelerate legal-action evaluation; search-tree traversal remains CPU-orchestrated and sequential in this first version.

## Important implementation details

Searchless action values are predicted win-probability distributions from the side-to-move perspective. SUMEDHA computes the expected value over the 128 published return buckets. For alpha-beta, it maps that expected probability back to an approximate centipawn scale using the inverse of Searchless' published centipawn-to-win-probability transform.

At a search node, action values order all legal moves. At the search horizon, the best predicted action value is used as the static position evaluation. Before iterative deepening, the root's highest-valued legal action is retained as a safe fallback if the first search iteration cannot finish within a very short time budget. Negamax flips perspective at every ply. Terminal checkmate/draw outcomes come from python-chess rules, not the neural network.

The first version intentionally avoids null-move pruning and aggressive selective search. Late-move reductions are conservative and can be disabled in code for comparison. Search results are experimental and must be benchmarked; this commit does not claim a particular Elo.

## Self-test

```bash
python -m sumedha_sl.selftest
```

This tests legal move production, mate-in-one detection and terminal checkmate handling with a deterministic dummy evaluator. It does not test the 270M checkpoint; run the model diagnostics for that.

## Model diagnostic and opponent arena

Run the frozen-model smoke diagnostic after setup:

```bash
python -m sumedha_sl.diagnose --checkpoint-dir checkpoints/270M --vendor-dir third_party --depth 4 --seconds 8
```

Run a paired-colour pilot against a local Stockfish executable:

```bash
python -m sumedha_sl.arena --sf /path/to/stockfish --games 20 --sumedha-time 1.0 --sf-time 1.0
```

The arena writes complete PGNs and a JSON report. It deliberately does not invent an Elo from a small pilot; use the recorded games for larger, calibrated engine testing.

## Attribution and licenses

The Searchless Chess source is Apache-2.0. Google's released pretrained model weights are CC-BY-4.0 and must be attributed. See `third_party/searchless_chess/LICENSE` and the official repository's README for details:
https://github.com/google-deepmind/searchless_chess

This branch keeps the upstream code and weights separate from SUMEDHA's original files. Any distribution of weights or derived artifacts must preserve the applicable attribution and license terms.
