# SUMEDHA AI

**A chess-native neural engine.** SUMEDHA learns a policy over legal moves and a value estimate for chess positions. It is not a language model.

## What this repository builds

- A residual convolutional policy/value network.
- Stockfish 19 teacher-data generation from diverse self-play positions.
- Curated opening-line sampling from the public-domain Lichess Chess Openings dataset (CC0).
- Optional Stockfish re-scoring of PGN games attached as Kaggle inputs.
- Two-GPU PyTorch DistributedDataParallel training when Kaggle exposes T4 x2.
- Atomic checkpoints, automatic resume, and a UCI inference program with PUCT search.

The initial model is a research baseline, not a claim of grandmaster or superhuman strength. Strength comes from data quality, much larger position counts, repeated self-play, search, and testing against independent opponents.

## One-command Kaggle launch

1. Create a Kaggle Notebook.
2. Set **Accelerator → GPU T4 x2** and **Internet → On**.
3. Run this in one notebook cell:

```python
!bash -c "$(curl -fsSL https://raw.githubusercontent.com/chesshook/SumedhaAI/main/kaggle.sh)"
```

The launcher clones this repository, installs only the small Python chess dependency (it does not reinstall PyTorch), runs a chess/model/search smoke test, checks the visible GPUs, restores prior notebook outputs when attached, downloads and SHA256-verifies the official Stockfish 19 universal Linux binary, builds a chess-position dataset, then trains on every visible CUDA GPU. It uses both T4s automatically when two are available.

Outputs are written to:

- `/kaggle/working/sumedha-output/latest.pt`
- `/kaggle/working/sumedha-output/best.pt`
- `/kaggle/working/sumedha-output/training.log`
- `/kaggle/working/sumedha-data/positions.npz`

Use Kaggle **Save Version** with notebook outputs enabled to preserve checkpoints and data after the session ends. To continue in a fresh session, attach the previous notebook version under **Input → Notebook Output Files**; the launcher automatically restores `latest.pt`, `best.pt` and `positions.npz` when it finds them.

## Optional: add master games

Attach a Kaggle dataset containing PGN files under `/kaggle/input`. The data builder discovers `.pgn` and `.pgn.gz` files, samples positions from their mainlines, and asks Stockfish to label the sampled positions. Games with Elo headers are prioritized when both players are rated at least 2200. The PGN remains in the Kaggle input; it is not copied into this repository.

Opening lines are downloaded from [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings). The opening dataset README declares the collection CC0. Dataset provenance is logged in the output directory.

## Training controls

Set environment variables before running the one-cell launcher to adjust resource use:

| Variable | Default | Meaning |
|---|---:|---|
| `SUMEDHA_GAMES` | 300 | Stockfish-vs-Stockfish games |
| `SUMEDHA_MAX_PLIES` | 80 | Maximum half-moves per generated game |
| `SUMEDHA_PLAY_NODES` | 250 | Cheap Stockfish nodes per move when generating games |
| `SUMEDHA_NODES` | 8000 | Stockfish nodes per labelled position |
| `SUMEDHA_MULTIPV` | 6 | Candidate moves evaluated per position |
| `SUMEDHA_OPENING_POSITIONS` | 3000 | Opening positions sampled from the opening dataset |
| `SUMEDHA_PGN_POSITIONS` | 5000 | Maximum PGN positions to re-score |
| `SUMEDHA_PGN_GAME_LIMIT` | 10000 | Maximum games scanned in attached PGN files |
| `SUMEDHA_MAX_POSITIONS` | 16000 | Labelled positions in the final training set |
| `SUMEDHA_EPOCHS` | 24 | Training epochs |
| `SUMEDHA_BATCH_SIZE` | 128 | Batch size per GPU |
| `SUMEDHA_CHANNELS` | 96 | Residual-network width |
| `SUMEDHA_BLOCKS` | 6 | Number of residual blocks |
| `SUMEDHA_WORKERS` | 2 | Data-loader workers per GPU |

For a quick smoke run, reduce games, positions, nodes, epochs, channels and blocks. For stronger training, grow the dataset substantially and continue from saved checkpoints. The default data build targets up to 16,000 labelled positions; it is a substantial starter corpus, not a billion-position engine training run. A single Kaggle run is not enough to reproduce the scale of AlphaZero or Leela Chess Zero.

## Use the trained network

After training, the checkpoint can be used as a UCI engine:

```bash
python -m sumedha.uci --checkpoint /kaggle/working/sumedha-output/latest.pt --simulations 128
```

Connect that process to a chess GUI that supports UCI engines. The first implementation uses neural priors, neural position values and PUCT search. It is not Stockfish and will be weaker until trained on substantially more data.

## Technical design

- Input: 19 planes of 8×8 board state (piece planes, turn, castling rights, en-passant square and halfmove clock).
- Policy: 20,480 logits, covering from-square, to-square and promotion type.
- Value: scalar in [-1, 1], from the current side-to-move perspective.
- Loss: soft-target policy cross-entropy plus Huber value loss.
- Optimizer: AdamW, mixed precision on CUDA, cosine learning-rate schedule.
- Multi-GPU: one process per GPU using NCCL DistributedDataParallel.
- Resume: the latest checkpoint contains model, optimizer, scheduler, scaler, epoch and step state.

## Data integrity and limitations

Stockfish is used as a teacher, not as part of the neural network. Its search is CPU work; the T4 GPUs are reserved for training. Stockfish labels are still engine evaluations, not objective truth. The opening repertoire is useful for opening coverage but does not replace master-game collections. The initial self-play corpus is only a starting point and should be expanded through repeated data-generation and training cycles.

No claim of consciousness, subjective feeling, or human-like understanding is made. SUMEDHA is an experimental chess decision system whose chess strength must be measured through held-out positions and match testing.

## References

- [Stockfish official repository and releases](https://github.com/official-stockfish/Stockfish)
- [Lichess chess-openings dataset](https://github.com/lichess-org/chess-openings)
- [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [python-chess](https://python-chess.readthedocs.io/)
