"""Build Stockfish-labelled SUMEDHA position datasets in Kaggle."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

import chess
import chess.engine
import chess.pgn
import numpy as np

from .encoding import encode_board, move_to_index

STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/"
    "stockfish-linux-x86-64-universal.tar.gz"
)
STOCKFISH_SHA256 = "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"
OPENINGS_BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"


def log(message: str) -> None:
    print(f"[SUMEDHA data] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_stockfish(cache_dir: Path) -> Path:
    """Download the immutable official Stockfish 19 Linux release and verify it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = cache_dir / "stockfish"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary

    archive = cache_dir / "stockfish-19-linux-x86-64-universal.tar.gz"
    if not archive.is_file() or sha256_file(archive) != STOCKFISH_SHA256:
        log("Downloading official Stockfish 19 universal Linux binary...")
        request = urllib.request.Request(
            STOCKFISH_URL, headers={"User-Agent": "SUMEDHA-AI-training/0.1"}
        )
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=4 * 1024 * 1024)
        actual = sha256_file(temporary)
        if actual != STOCKFISH_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Stockfish archive SHA256 mismatch: expected {STOCKFISH_SHA256}, got {actual}"
            )
        temporary.replace(archive)

    log("Verifying and extracting Stockfish...")
    extract_dir = cache_dir / "stockfish-extract"
    extract_dir.mkdir(exist_ok=True)
    candidate = None
    with tarfile.open(archive, "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if m.isfile() and Path(m.name).name.lower().startswith("stockfish") and m.size > 5_000_000
        ]
        if not members:
            raise RuntimeError("The verified Stockfish archive did not contain an expected engine binary.")
        # The universal package should contain one main executable; prefer the largest file.
        member = max(members, key=lambda m: m.size)
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError("Could not read the Stockfish executable from its archive.")
        candidate = extract_dir / "stockfish"
        with source, candidate.open("wb") as out:
            shutil.copyfileobj(source, out, length=4 * 1024 * 1024)

    candidate.chmod(0o755)
    os.replace(candidate, binary)
    shutil.rmtree(extract_dir, ignore_errors=True)
    # Verify the executable actually speaks UCI before starting expensive labelling.
    engine = chess.engine.SimpleEngine.popen_uci(str(binary), timeout=30)
    try:
        engine.ping()
        log(f"Stockfish ready: {engine.id.get('name', 'unknown engine')}")
    finally:
        engine.quit()
    return binary


def download_opening_positions(rng: random.Random, limit: int) -> tuple[list[str], list[str]]:
    """Fetch the CC0 Lichess opening lines and return position and game-seed FENs."""
    positions: list[str] = []
    seeds: list[str] = []
    if limit <= 0:
        return positions, seeds

    for letter in "abcde":
        url = f"{OPENINGS_BASE}/{letter}.tsv"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SUMEDHA-AI/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8")
        except Exception as exc:
            log(f"Could not download opening volume {letter.upper()}: {exc}")
            continue

        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        rows = list(reader)
        rng.shuffle(rows)
        for row in rows:
            pgn_text = (row.get("pgn") or "").strip()
            if not pgn_text:
                continue
            try:
                game = chess.pgn.read_game(io.StringIO(pgn_text))
                if game is None or game.errors:
                    continue
                board = game.board()
                line_fens = []
                for move in game.mainline_moves():
                    if move not in board.legal_moves:
                        break
                    board.push(move)
                    if board.is_game_over(claim_draw=True):
                        break
                    line_fens.append(board.fen())
                if line_fens:
                    seeds.append(line_fens[-1])
                    # Keep a spread of positions, not only the end of each variation.
                    stride = max(1, len(line_fens) // 4)
                    positions.extend(line_fens[::stride])
            except Exception:
                continue
            if len(positions) >= limit * 3 and len(seeds) >= 200:
                break
        if len(positions) >= limit * 3 and len(seeds) >= 200:
            break

    rng.shuffle(positions)
    rng.shuffle(seeds)
    return positions[:limit * 3], seeds[:max(200, limit // 2)]


def _open_pgn(path: Path):
    if path.name.lower().endswith(".pgn.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def collect_master_positions(
    rng: random.Random, limit: int, game_limit: int
) -> list[str]:
    """Sample diverse game positions, preferring games where both players are 2200+."""
    if limit <= 0:
        return []
    root = Path("/kaggle/input")
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and (p.name.lower().endswith(".pgn") or p.name.lower().endswith(".pgn.gz"))
    ) if root.exists() else []
    if not files:
        log("No PGN inputs found. Attach a PGN dataset to add master-game positions.")
        return []

    strong: list[str] = []
    other: list[str] = []
    cap = max(limit * 5, 1000)
    seen_games = 0
    for path in files:
        log(f"Reading PGN: {path}")
        try:
            stream = _open_pgn(path)
        except Exception as exc:
            log(f"Skipping unreadable PGN {path}: {exc}")
            continue
        with stream:
            while seen_games < game_limit:
                try:
                    game = chess.pgn.read_game(stream)
                except Exception:
                    break
                if game is None:
                    break
                seen_games += 1
                try:
                    moves = list(game.mainline_moves())
                    if len(moves) < 10:
                        continue
                    # Choose a non-opening position across the game's full length.
                    lo = min(8, len(moves) - 2)
                    hi = max(lo, len(moves) - 1)
                    ply = rng.randint(lo, hi)
                    board = game.board()
                    for move in moves[:ply]:
                        if move not in board.legal_moves:
                            break
                        board.push(move)
                    else:
                        if not board.is_valid() or not board.legal_moves:
                            continue
                        fen = board.fen()
                        try:
                            white_elo = int(game.headers.get("WhiteElo", "0"))
                            black_elo = int(game.headers.get("BlackElo", "0"))
                        except (TypeError, ValueError):
                            white_elo = black_elo = 0
                        bucket = strong if white_elo >= 2200 and black_elo >= 2200 else other
                        if len(bucket) < cap:
                            bucket.append(fen)
                        else:
                            # Reservoir sampling prevents early-file bias.
                            index = rng.randrange(seen_games)
                            if index < cap:
                                bucket[index] = fen
                except Exception:
                    continue
                if seen_games >= game_limit:
                    break

    rng.shuffle(strong)
    rng.shuffle(other)
    selected = strong[:limit]
    if len(selected) < limit:
        selected.extend(other[:limit - len(selected)])
    log(
        f"PGN scan: {seen_games} games; {len(strong)} strong-game candidates; "
        f"{len(selected)} selected positions."
    )
    return selected


def generate_selfplay_positions(
    engine: chess.engine.SimpleEngine,
    rng: random.Random,
    games: int,
    max_plies: int,
    play_nodes: int,
    opening_seeds: list[str],
) -> list[str]:
    """Generate varied Stockfish self-play trajectories at a deliberately cheap budget."""
    positions: list[str] = []
    options = engine.options
    can_set_skill = "Skill Level" in options
    for game_index in range(games):
        if opening_seeds and rng.random() < 0.75:
            board = chess.Board(rng.choice(opening_seeds))
        else:
            board = chess.Board()
            # Tiny legal opening perturbations add variety before teacher play.
            for _ in range(rng.randint(0, 4)):
                if board.is_game_over(claim_draw=True):
                    break
                board.push(rng.choice(list(board.legal_moves)))
        if can_set_skill:
            try:
                engine.configure({"Skill Level": rng.randint(8, 20)})
            except Exception:
                can_set_skill = False

        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            if len(board.move_stack) > 0 and len(board.move_stack) % 2 == 0:
                # Avoid storing a large number of consecutive near-identical positions.
                positions.append(board.fen())
            try:
                result = engine.play(board, chess.engine.Limit(nodes=play_nodes))
            except Exception as exc:
                log(f"Self-play game {game_index + 1} stopped: {exc}")
                break
            if result.move is None or result.move not in board.legal_moves:
                break
            board.push(result.move)

        if (game_index + 1) % 10 == 0 or game_index + 1 == games:
            log(f"Self-play games: {game_index + 1}/{games}; positions collected: {len(positions)}")
    return positions


def choose_balanced(
    sources: dict[str, list[str]], maximum: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Select a balanced mix so long self-play runs do not drown out master/opening play."""
    weights = {"selfplay": 0.40, "opening": 0.25, "master": 0.35}
    active = [name for name, values in sources.items() if values]
    if not active:
        return []
    if len(active) == 1:
        weights = {active[0]: 1.0}
    else:
        total = sum(weights.get(name, 0.0) for name in active)
        weights = {name: weights.get(name, 0.0) / total for name in active}

    selected: list[tuple[str, str]] = []
    used: set[str] = set()
    for name in active:
        values = list(dict.fromkeys(sources[name]))
        rng.shuffle(values)
        quota = min(len(values), int(maximum * weights[name]))
        for fen in values[:quota]:
            if fen not in used:
                selected.append((name, fen))
                used.add(fen)

    if len(selected) < maximum:
        remaining = [
            (name, fen) for name in active for fen in dict.fromkeys(sources[name])
            if fen not in used
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: maximum - len(selected)])
    rng.shuffle(selected)
    return selected[:maximum]


def label_position(
    engine: chess.engine.SimpleEngine, fen: str, multipv: int, nodes: int, temperature: float
):
    board = chess.Board(fen)
    if not board.is_valid() or not any(board.legal_moves):
        return None
    try:
        infos = engine.analyse(
            board, chess.engine.Limit(nodes=nodes), multipv=multipv
        )
    except Exception:
        return None
    if isinstance(infos, dict):
        infos = [infos]

    candidate_moves: list[int] = []
    candidate_scores: list[float] = []
    legal = set(board.legal_moves)
    for info in infos:
        pv = info.get("pv") or []
        if not pv or pv[0] not in legal or "score" not in info:
            continue
        try:
            score = info["score"].pov(board.turn)
            cp = score.score(mate_score=10000)
            if cp is None:
                cp = 0
            index = move_to_index(pv[0])
        except Exception:
            continue
        if index in candidate_moves:
            continue
        candidate_moves.append(index)
        candidate_scores.append(float(cp))

    if not candidate_moves:
        return None
    scores = np.asarray(candidate_scores, dtype=np.float64)
    scaled = (scores - scores.max()) / max(1.0, temperature)
    probabilities = np.exp(np.clip(scaled, -40.0, 0.0))
    probabilities /= probabilities.sum()
    # Value is from the side-to-move perspective, matching the board encoding.
    value = float(np.tanh(candidate_scores[0] / 600.0))
    return (
        encode_board(board),
        candidate_moves,
        probabilities.astype(np.float16),
        value,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--play-nodes", type=int, default=250)
    parser.add_argument("--nodes", type=int, default=8000)
    parser.add_argument("--multipv", type=int, default=6)
    parser.add_argument("--opening-positions", type=int, default=3000)
    parser.add_argument("--pgn-positions", type=int, default=5000)
    parser.add_argument("--pgn-game-limit", type=int, default=10000)
    parser.add_argument("--max-positions", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()

    if min(args.games, args.max_plies, args.play_nodes, args.nodes, args.multipv,
           args.max_positions) < 1:
        raise SystemExit("Games, plies, node budgets, MultiPV and max positions must be positive.")

    rng = random.Random(args.seed)
    data_dir = Path(os.environ.get("SUMEDHA_DATA_DIR", "/kaggle/working/sumedha-data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    log("Collecting curated chess opening positions (Lichess dataset, CC0)...")
    opening_fens, opening_seeds = download_opening_positions(rng, args.opening_positions)
    log(f"Opening positions: {len(opening_fens)}; self-play seed positions: {len(opening_seeds)}")

    log("Sampling positions from attached PGN collections...")
    master_fens = collect_master_positions(rng, args.pgn_positions, args.pgn_game_limit)

    stockfish = ensure_stockfish(data_dir / "engine")
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish), timeout=60)
    try:
        threads = max(1, min(2, (os.cpu_count() or 2) // 2))
        engine.configure({"Threads": threads, "Hash": 512})
        log(f"Generating {args.games} self-play games (teacher play budget: {args.play_nodes} nodes/move)...")
        selfplay_fens = generate_selfplay_positions(
            engine, rng, args.games, args.max_plies, args.play_nodes, opening_seeds
        )

        sources = {
            "selfplay": selfplay_fens,
            "opening": opening_fens,
            "master": master_fens,
        }
        selected = choose_balanced(sources, args.max_positions, rng)
        log(
            "Selected positions: " + ", ".join(
                f"{name}={sum(1 for source, _ in selected if source == name)}"
                for name in ("selfplay", "opening", "master")
            )
        )
        if len(selected) < 32:
            raise RuntimeError(
                f"Only {len(selected)} unique positions were collected. "
                "Check Kaggle Internet and increase SUMEDHA_GAMES or opening data."
            )

        x_rows: list[np.ndarray] = []
        move_rows: list[list[int]] = []
        probability_rows: list[list[float]] = []
        value_rows: list[float] = []
        kept_sources: dict[str, int] = {"selfplay": 0, "opening": 0, "master": 0}

        log(f"Labelling {len(selected)} positions with Stockfish MultiPV={args.multipv}, nodes={args.nodes}...")
        for number, (source, fen) in enumerate(selected, 1):
            row = label_position(
                engine, fen, args.multipv, args.nodes, args.temperature
            )
            if row is None:
                continue
            board_x, indices, probabilities, value = row
            # Pad to a fixed candidate count; zero probabilities make padding inert.
            k = args.multipv
            indices = indices[:k]
            probabilities = probabilities[:k]
            padded_indices = indices + [0] * (k - len(indices))
            padded_probabilities = probabilities.tolist() + [0.0] * (k - len(probabilities))
            x_rows.append(board_x)
            move_rows.append(padded_indices)
            probability_rows.append(padded_probabilities)
            value_rows.append(value)
            kept_sources[source] += 1

            if number % 100 == 0 or number == len(selected):
                log(f"Teacher labels: {number}/{len(selected)}; valid: {len(x_rows)}")

        if len(x_rows) < 32:
            raise RuntimeError(f"Stockfish successfully labelled only {len(x_rows)} positions.")

        metadata = {
            "project": "SUMEDHA AI",
            "schema_version": 1,
            "position_count": len(x_rows),
            "sources": kept_sources,
            "stockfish": "Stockfish 19",
            "stockfish_release": "sf_19",
            "stockfish_archive_sha256": STOCKFISH_SHA256,
            "teacher_nodes": args.nodes,
            "teacher_multipv": args.multipv,
            "policy_temperature_cp": args.temperature,
            "created_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }
        temp_output = output.with_suffix(output.suffix + ".tmp")
        with temp_output.open("wb") as stream:
            np.savez_compressed(
                stream,
                x=np.stack(x_rows).astype(np.float16),
                move_indices=np.asarray(move_rows, dtype=np.int32),
                move_probs=np.asarray(probability_rows, dtype=np.float16),
                values=np.asarray(value_rows, dtype=np.float32),
                metadata=np.asarray(json.dumps(metadata)),
            )
        os.replace(temp_output, output)
        log(f"DATASET READY: {output} ({len(x_rows)} positions)")
        log(json.dumps(metadata, indent=2))
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
