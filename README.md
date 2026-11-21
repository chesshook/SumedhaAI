# SUMEDHA AI

**A chess-native neural engine.** SUMEDHA learns a policy over legal moves and a value estimate for chess positions. It is not a language model.

## Data rule: precomputed corpora only

The Kaggle data pipeline does **not** launch Stockfish, run self-play, analyse positions, generate engine moves, or create new evaluations. It only reads two existing archives:

### 1. Precomputed engine-evaluation database — approximately 6.6 GB

Three fixed Parquet shards from [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations):

- `data_0000.parquet` — 2,190,285,438 bytes
- `data_0001.parquet` — 2,189,026,305 bytes
- `0011.parquet` — approximately 2.19 GB

Together these are approximately **6.6 GB of already-calculated Stockfish evaluations**. Rows contain a position FEN, a principal variation, search depth, searched kilo-nodes and a centipawn or mate score. The source data is published under CC0. SUMEDHA checks the SHA-256 of each shard before reading it.

The loader reads selected row groups from each shard, filters for depth and node-count quality, and samples precomputed positions. It never invokes an engine to calculate a label.

### 2. Elite human-game and opening corpus

SUMEDHA fetches only `chessmont.pgn.zst` from the public Kaggle dataset [Chessmont Big Dataset](https://www.kaggle.com/datasets/chessmontdb/chessmont-big-dataset). Its published description lists **21.5+ million unique games** merged from Lichess 2500+ history, Chess.com top-10K blitz 2500+, TWIC and PGN Mentor.

The published corpus contains 21.5+ million unique games (the site describes its main game sources as 2500+ Elo). The loader logs the actual compressed archive size after download, then samples opening-phase and later-game positions. Its policy label is the actual move recorded by the human player; its value target is based on the recorded game result. There is no move generation or engine re-analysis.

Chessmont declares CC BY-SA 4.0 for its collection. The dataset metadata records the source and license; see the [Chessmont Open Database](https://database.chessmont.com/en/) and [CC BY-SA 4.0 terms](https://creativecommons.org/licenses/by-sa/4.0/). Lichess evaluation data is CC0; see [Lichess Open Database](https://database.lichess.org/).

**Terminology:** the second source is a large elite human-game corpus from which opening move sequences are learned, not a single 6 GB Polyglot `.bin` file. A Polyglot book stores position-to-move entries; the PGN corpus preserves complete played games and therefore supplies opening, middlegame and endgame experience.

## One-command Kaggle launch

1. Create a Kaggle Notebook.
2. Select **Accelerator → GPU T4 x2** and **Internet → On**.
3. Run this in one cell:

```python
!bash -c "$(curl -fsSL https://raw.githubusercontent.com/chesshook/SumedhaAI/main/kaggle.sh)"
```

The launcher fetches this repository, installs the chess/Parquet/Zstandard readers without reinstalling PyTorch, checks both GPUs, runs a chess/network smoke test, downloads and verifies the three fixed evaluation shards, fetches the prebuilt Chessmont PGN, builds a training array from the two corpora, and starts two-GPU distributed training.

It does **not** download a Stockfish binary or launch Stockfish.

## Data controls

Set environment variables before the launcher when you need different sample sizes:

| Variable | Default | Meaning |
|---|---:|---|
| `SUMEDHA_EVAL_POSITIONS` | 350000 | Maximum precomputed evaluation positions sampled |
| `SUMEDHA_HUMAN_POSITIONS` | 150000 | Maximum recorded human-move positions |
| `SUMEDHA_HUMAN_GAMES` | 150000 | Maximum elite PGN games scanned |
| `SUMEDHA_MINIMUM_DEPTH` | 20 | Minimum precomputed engine depth |
| `SUMEDHA_MINIMUM_KNODES` | 1000 | Minimum precomputed searched kilo-nodes |
| `SUMEDHA_ROWGROUPS_PER_SHARD` | 32 | Row groups sampled from each fixed Parquet shard |
| `SUMEDHA_REBUILD_DATA` | 0 | Set to `1` to rebuild v2 arrays from the fixed archives |
| `SUMEDHA_EPOCHS` | 24 | Training epochs |
| `SUMEDHA_BATCH_SIZE` | 128 | Batch size per GPU |
| `SUMEDHA_CHANNELS` | 96 | Residual-network width |
| `SUMEDHA_BLOCKS` | 6 | Number of residual blocks |
| `SUMEDHA_WORKERS` | 2 | Data-loader workers per GPU |

For a custom PGN file already attached to Kaggle, set `SUMEDHA_HUMAN_PGN=/kaggle/input/.../your_games.pgn.zst`. The default is the published Chessmont file.

## Saved outputs and continuation

The v2 pipeline uses separate directories so it cannot accidentally reuse the earlier live-Stockfish-generated v1 dataset or a v1 checkpoint:

- `/kaggle/working/sumedha-data-v2/positions_v2.npz`
- `/kaggle/working/sumedha-output-v2/latest.pt`
- `/kaggle/working/sumedha-output-v2/best.pt`
- `/kaggle/working/sumedha-output-v2/training.log`

Use Kaggle **Save Version** with notebook outputs enabled. In a later session, attach that saved Notebook Output; the launcher automatically restores the v2 dataset/checkpoint when present.

To extend training to 48 total epochs, for example:

```python
%env SUMEDHA_EPOCHS=48
!bash -c "$(curl -fsSL https://raw.githubusercontent.com/chesshook/SumedhaAI/main/kaggle.sh)"
```

The latest checkpoint stores model, optimizer, scheduler, scaler, epoch and step state. The training loop resumes from `latest.pt`.

## Use the trained network

After training, load the checkpoint as a UCI engine:

```bash
python -m sumedha.uci --checkpoint /kaggle/working/sumedha-output-v2/latest.pt --simulations 128
```

Connect it to a chess GUI that supports UCI engines. The initial inference program uses neural move priors, neural position values and PUCT search. It is an experimental baseline, not a claim of grandmaster or superhuman strength.

## Technical design

- Input: 19 planes of 8×8 board state (piece planes, side to move, castling rights, en-passant square and halfmove clock).
- Policy: 20,480 logits, covering from-square, to-square and promotion type.
- Value: scalar in [-1, 1], from the current side-to-move perspective.
- Loss: soft-target policy cross-entropy plus Huber value loss.
- Optimizer: AdamW with FP16 autocast for CUDA operations and cosine learning-rate scheduling.
- Model parameters and saved checkpoints remain FP32; no weight quantization is used.
- Position arrays are saved as uncompressed NPZ.
- Multi-GPU: one process per GPU using NCCL DistributedDataParallel.

## Limitations

This is supervised learning from fixed evaluations and recorded human moves. It is not AlphaZero-style self-play and does not prove consciousness or subjective feeling. Strength must be measured with held-out positions and match testing. Future self-play, if ever added, should be a separately controlled phase—not silently mixed into this fixed-corpus data pipeline.
