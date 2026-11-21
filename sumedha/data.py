"""Build SUMEDHA training examples ONLY from precomputed chess corpora.

No engine executable is installed or started here:
- Lichess Stockfish-evaluated positions are downloaded as fixed Parquet shards.
- Chessmont's prebuilt elite human PGN collection supplies played moves/opening lines.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zstandard

from .encoding import encode_board, move_to_index

# Three fixed 2.19–2.22 GB shards: approximately 6.5 GB of pre-calculated
# Stockfish evaluations. Hashes are checked before any position is consumed.
EVAL_SHARDS = (
    {
        "name": "data_0000.parquet",
        "url": "https://huggingface.co/datasets/Lichess/chess-position-evaluations/resolve/main/data/data_0000.parquet",
        "sha256": "b9306270a185eee925863b34f9d6d3d7c9b1ba772df251b2c76d421f7cf13d3b",
    },
    {
        "name": "data_0001.parquet",
        "url": "https://huggingface.co/datasets/Lichess/chess-position-evaluations/resolve/main/data/data_0001.parquet",
        "sha256": "e5def35cb4ab5229019540c194e15740cd6a877a0871e655cecaca85efa1638a",
    },
    {
        "name": "data_0011.parquet",
        "url": "https://huggingface.co/datasets/Lichess/chess-position-evaluations/resolve/main/data/data_0011.parquet",
        "sha256": "b9b6c20a2b7afa3cc251f339046a03332af09821009ce76192548d44efc4114b",
    },
)
CHESSMONT_DATASET = "chessmontdb/chessmont-big-dataset"
CHESSMONT_FILE = "chessmont.pgn.zst"


def log(message: str) -> None:
    print(f"[SUMEDHA data] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_attached_file(root: Path, names: set[str]) -> Path | None:
    if not root.exists():
        return None
    matches = [p for p in root.rglob("*") if p.is_file() and p.name in names]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_size)


def download_one(url: str, target: Path, expected_sha256: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256_file(target) == expected_sha256:
        log(f"Verified cached precomputed shard: {target.name}")
        return target
    target.unlink(missing_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    temporary.unlink(missing_ok=True)

    log(f"Downloading precomputed Stockfish shard {target.name}...")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SUMEDHA-AI-precomputed-dataset/0.2"},
    )
    received = 0
    last_report = 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            received += len(chunk)
            if received - last_report >= 256 * 1024 * 1024:
                log(f"  {target.name}: {received / (1024**3):.2f} GiB downloaded")
                last_report = received
    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch for {target.name}: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(target)
    log(f"Verified {target.name}: {received / (1024**3):.2f} GiB")
    return target


def get_eval_shards(data_dir: Path) -> list[Path]:
    attached = Path("/kaggle/input")
    paths: list[Path] = []
    for shard in EVAL_SHARDS:
        local = find_attached_file(attached, {shard["name"]})
        if local is not None and sha256_file(local) == shard["sha256"]:
            log(f"Using attached, hash-verified evaluation shard: {local}")
            paths.append(local)
        else:
            paths.append(
                download_one(
                    shard["url"],
                    data_dir / "lichess-evaluations" / shard["name"],
                    shard["sha256"],
                )
            )
    return paths


def get_chessmont_pgn(data_dir: Path) -> Path:
    configured = os.environ.get("SUMEDHA_HUMAN_PGN", "").strip()
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError(f"SUMEDHA_HUMAN_PGN does not exist: {path}")
        return path

    attached = find_attached_file(Path("/kaggle/input"), {CHESSMONT_FILE, "chessmont.pgn"})
    if attached is not None:
        log(
            f"Using attached elite human-game corpus: {attached} "
            f"({attached.stat().st_size / (1024**3):.2f} GiB)"
        )
        return attached

    log("Fetching the prebuilt Chessmont elite PGN file (no games are generated)...")
    try:
        import kagglehub
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", "kagglehub"]
        )
        import kagglehub

    output_dir = data_dir / "chessmont"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = kagglehub.dataset_download(
        CHESSMONT_DATASET,
        path=CHESSMONT_FILE,
        output_dir=str(output_dir),
    )
    result_path = Path(result)
    if result_path.is_file():
        chosen = result_path
    else:
        chosen = find_attached_file(result_path if result_path.is_dir() else output_dir, {CHESSMONT_FILE})
    if chosen is None:
        raise RuntimeError(
            "KaggleHub did not return chessmont.pgn.zst. Attach the public dataset "
            f"{CHESSMONT_DATASET} as a Kaggle Input and rerun."
        )
    log(f"Elite human PGN archive size: {chosen.stat().st_size / (1024**3):.2f} GiB")
    return chosen


def open_text_archive(path: Path):
    """Open ordinary PGN or Zstandard-compressed PGN as a streaming text handle."""
    raw = path.open("rb")
    if path.name.lower().endswith(".zst"):
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_size=1024 * 1024)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def normalise_fen(fen: str) -> str:
    fields = fen.strip().split()
    if len(fields) == 4:
        fields.extend(["0", "1"])
    if len(fields) != 6:
        raise ValueError("Position FEN must contain 4 or 6 fields.")
    return " ".join(fields)


def score_to_value(score_white_cp: float, turn: chess.Color) -> float:
    score_stm = score_white_cp if turn == chess.WHITE else -score_white_cp
    return float(np.tanh(np.clip(score_stm, -6000.0, 6000.0) / 600.0))


def parse_eval_move(board: chess.Board, uci: str) -> chess.Move | None:
    """Parse normal UCI and Lichess UCI_Chess960 castling notation for standard boards."""
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if move in board.legal_moves:
        return move

    # Lichess evaluation exports describe the PV using UCI_Chess960 notation.
    # In orthodox chess, a king-to-own-rook move is equivalent to standard UCI
    # castling (king-to-c/g-file). Translate that representation when legal.
    piece = board.piece_at(move.from_square)
    rook = board.piece_at(move.to_square)
    if (
        piece is not None and piece.piece_type == chess.KING
        and rook is not None and rook.piece_type == chess.ROOK
        and piece.color == rook.color
        and chess.square_rank(move.from_square) == chess.square_rank(move.to_square)
    ):
        rank = chess.square_rank(move.from_square)
        king_side = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        target = chess.square(6 if king_side else 2, rank)
        translated = chess.Move(move.from_square, target)
        if translated in board.legal_moves:
            return translated
    return None


def read_eval_row(row: dict[str, Any]):
    fen = row.get("fen")
    line = row.get("line")
    if not fen or not line:
        return None
    try:
        board = chess.Board(normalise_fen(str(fen)))
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            return None
        first_move = str(line).split()[0]
        move = parse_eval_move(board, first_move)
        if move is None:
            return None

        cp = row.get("cp")
        mate = row.get("mate")
        if mate is not None:
            mate = int(mate)
            white_score = (1 if mate > 0 else -1) * (10000.0 - min(abs(mate), 99) * 100.0)
        elif cp is not None:
            white_score = float(cp)
        else:
            return None

        return encode_board(board), move_to_index(move), score_to_value(white_score, board.turn)
    except (ValueError, TypeError, chess.InvalidMoveError, chess.IllegalMoveError):
        return None


def sample_eval_shards(
    paths: list[Path],
    limit: int,
    minimum_depth: int,
    minimum_knodes: int,
    rowgroups_per_shard: int,
    seed: int,
):
    """Sample already-evaluated positions from distributed Parquet row groups."""
    rng = np.random.default_rng(seed)
    files = [(path, pq.ParquetFile(path)) for path in paths]
    total_groups = sum(max(1, min(rowgroups_per_shard, parquet.num_row_groups)) for _, parquet in files)
    per_group = max(1, int(np.ceil(limit / max(1, total_groups))))
    output = []
    total_rows_seen = 0
    total_valid = 0

    for path, parquet in files:
        group_count = parquet.num_row_groups
        take_groups = min(rowgroups_per_shard, group_count)
        chosen_groups = np.linspace(0, group_count - 1, num=take_groups, dtype=int)
        log(
            f"Reading pre-calculated evaluations from {path.name}: "
            f"{take_groups}/{group_count} row groups."
        )
        for group_index in chosen_groups:
            table = parquet.read_row_group(
                int(group_index),
                columns=["fen", "line", "depth", "knodes", "cp", "mate"],
                use_threads=True,
            )
            total_rows_seen += table.num_rows
            mask = pc.and_(
                pc.greater_equal(table["depth"], minimum_depth),
                pc.greater_equal(table["knodes"], minimum_knodes),
            )
            has_score = pc.or_(pc.is_valid(table["cp"]), pc.is_valid(table["mate"]))
            table = table.filter(pc.and_(mask, has_score))
            if table.num_rows == 0:
                continue

            if table.num_rows > per_group:
                picked = rng.choice(table.num_rows, size=per_group, replace=False)
                table = table.take(pa.array(picked, type=pa.int64()))

            for row in table.to_pylist():
                sample = read_eval_row(row)
                if sample is None:
                    continue
                output.append(sample)
                total_valid += 1
                if len(output) >= limit:
                    break
            if len(output) >= limit:
                break

        log(
            f"  {path.name}: rows read={total_rows_seen:,}; "
            f"valid samples collected={len(output):,}"
        )
        if len(output) >= limit:
            break

    rng.shuffle(output)
    log(f"Precomputed evaluation samples retained: {len(output):,}")
    return output


def result_white_score(result: str) -> float | None:
    return {
        "1-0": 1.0,
        "0-1": -1.0,
        "1/2-1/2": 0.0,
    }.get(result)


def sample_human_games(
    pgn_path: Path,
    position_limit: int,
    games_limit: int,
    seed: int,
):
    """Extract played human moves from the prebuilt elite PGN corpus.

    Per accepted game, one opening-phase position and (where possible) one
    later-game position are sampled. The move is the actual recorded human
    move; the value target comes from the recorded game result.
    """
    rng = random.Random(seed + 31)
    samples = []
    games_seen = 0
    games_used = 0
    rejected = 0
    log(f"Reading pre-existing human games from {pgn_path.name}...")
    stream = open_text_archive(pgn_path)
    try:
        while games_seen < games_limit and len(samples) < position_limit:
            try:
                game = chess.pgn.read_game(stream)
            except (ValueError, UnicodeError) as exc:
                rejected += 1
                log(f"Stopping on a malformed PGN stream record: {exc}")
                break
            except zstandard.ZstdError as exc:
                log(f"Reached the end of the available PGN stream: {exc}")
                break
            if game is None:
                break
            games_seen += 1
            if game.errors:
                rejected += 1
                continue

            result_score = result_white_score(game.headers.get("Result", "?"))
            if result_score is None:
                continue
            variant = game.headers.get("Variant", "Standard").lower()
            if variant not in ("standard", "chess", ""):
                continue

            try:
                moves = list(game.mainline_moves())
            except Exception:
                rejected += 1
                continue
            if len(moves) < 8:
                continue

            opening_ply = rng.randrange(min(20, len(moves)))
            later_start = min(20, len(moves) - 1)
            later_ply = (
                rng.randrange(later_start, len(moves))
                if len(moves) > 22 else None
            )
            selected = sorted(set([opening_ply] + ([later_ply] if later_ply is not None else [])))
            board = game.board()
            wanted = set(selected)
            game_samples = []

            for ply, move in enumerate(moves):
                if move not in board.legal_moves:
                    rejected += 1
                    game_samples = []
                    break
                if ply in wanted:
                    value = result_score if board.turn == chess.WHITE else -result_score
                    game_samples.append(
                        (encode_board(board), move_to_index(move), float(value))
                    )
                board.push(move)
                if len(game_samples) == len(selected):
                    break

            for sample in game_samples:
                samples.append(sample)
                if len(samples) >= position_limit:
                    break
            if game_samples:
                games_used += 1

            if games_seen % 10000 == 0:
                log(
                    f"Human games scanned={games_seen:,}/{games_limit:,}; "
                    f"move samples={len(samples):,}"
                )
    finally:
        stream.close()

    rng.shuffle(samples)
    log(
        f"Human corpus: games scanned={games_seen:,}; games used={games_used:,}; "
        f"samples={len(samples):,}; rejected games={rejected:,}"
    )
    return samples


def pack_samples(eval_samples, human_samples, seed: int):
    rows = []
    for x, move, value in eval_samples:
        rows.append((x, [move], [1.0], value, "precomputed_stockfish"))
    for x, move, value in human_samples:
        rows.append((x, [move], [1.0], value, "elite_human_game"))

    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < 32:
        raise RuntimeError(
            f"Only {len(rows)} usable samples were extracted. "
            "Check the archives and increase the source limits."
        )

    x = np.stack([row[0] for row in rows]).astype(np.float32, copy=False)
    move_indices = np.asarray([row[1] for row in rows], dtype=np.int32)
    move_probs = np.asarray([row[2] for row in rows], dtype=np.float32)
    values = np.asarray([row[3] for row in rows], dtype=np.float32)
    source_counts = {
        "precomputed_stockfish": sum(row[4] == "precomputed_stockfish" for row in rows),
        "elite_human_game": sum(row[4] == "elite_human_game" for row in rows),
    }
    return x, move_indices, move_probs, values, source_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-positions", type=int, default=350000)
    parser.add_argument("--human-positions", type=int, default=150000)
    parser.add_argument("--human-games", type=int, default=150000)
    parser.add_argument("--minimum-depth", type=int, default=20)
    parser.add_argument("--minimum-knodes", type=int, default=1000)
    parser.add_argument("--rowgroups-per-shard", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()

    if min(
        args.eval_positions, args.human_positions, args.human_games,
        args.minimum_depth, args.rowgroups_per_shard,
    ) < 1:
        raise SystemExit("Dataset limits and quality thresholds must be positive.")

    data_dir = Path(os.environ.get("SUMEDHA_DATA_DIR", "/kaggle/working/sumedha-data-v2"))
    data_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    log("DATA MODE: precomputed archives only. Stockfish will NOT be started.")
    log("Source 1: Lichess Stockfish evaluation Parquet shards, approximately 6.6 GB.")
    log("Source 2: Chessmont's prebuilt 21.5M elite human-game PGN corpus.")

    eval_paths = get_eval_shards(data_dir)
    human_pgn = get_chessmont_pgn(data_dir)

    eval_samples = sample_eval_shards(
        eval_paths,
        limit=args.eval_positions,
        minimum_depth=args.minimum_depth,
        minimum_knodes=args.minimum_knodes,
        rowgroups_per_shard=args.rowgroups_per_shard,
        seed=args.seed,
    )
    human_samples = sample_human_games(
        human_pgn,
        position_limit=args.human_positions,
        games_limit=args.human_games,
        seed=args.seed,
    )

    x, moves, probs, values, source_counts = pack_samples(
        eval_samples, human_samples, args.seed
    )
    metadata = {
        "project": "SUMEDHA AI",
        "schema_version": 2,
        "position_count": int(len(values)),
        "sources": source_counts,
        "eval_source": "Lichess/chess-position-evaluations",
        "eval_source_license": "CC0-1.0",
        "eval_shards": [p.name for p in eval_paths],
        "human_source": CHESSMONT_DATASET,
        "human_source_file": human_pgn.name,
        "human_source_license": "CC BY-SA 4.0 (as declared by Chessmont Open Database)",
        "stockfish_runtime_used": False,
        "teacher_labels": "precomputed Lichess evaluations; no engine search performed",
        "human_policy_labels": "moves recorded in the source PGN; no moves generated",
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            x=x,
            move_indices=moves,
            move_probs=probs,
            values=values,
            metadata=np.asarray(json.dumps(metadata)),
        )
    os.replace(temporary, output)
    log(f"PRECOMPUTED DATASET READY: {output}")
    log(f"Total samples: {len(values):,}")
    log(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
