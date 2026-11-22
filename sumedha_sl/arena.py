"""Reproducible SUMEDHA-SL+AB vs UCI-engine match runner.

Games are paired by opening and colour. Every completed game is saved to PGN.
No blended or invented Elo is reported.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from .model import Searchless270M
from .search import SearchlessPVS


def make_openings(count: int, seed: int, min_plies: int, max_plies: int):
    rng = random.Random(seed)
    openings = []
    for _ in range(count):
        board = chess.Board()
        target = rng.randint(min_plies, max_plies)
        for _ply in range(target):
            if board.is_game_over(claim_draw=True):
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            # Random legal opening diversity; seed guarantees reproducibility.
            board.push(rng.choice(moves))
        openings.append(board)
    return openings


def play_one(
    board: chess.Board,
    sumedha_white: bool,
    searcher: SearchlessPVS,
    stockfish: chess.engine.SimpleEngine,
    sumedha_seconds: float,
    stockfish_seconds: float,
    max_plies: int,
):
    ply_nodes = 0
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        is_sumedha_turn = (board.turn == chess.WHITE) == sumedha_white
        if is_sumedha_turn:
            result = searcher.search(
                board,
                max_depth=64,
                time_limit=sumedha_seconds,
            )
            move = result.best_move
            ply_nodes += result.nodes
        else:
            response = stockfish.play(
                board,
                limit=chess.engine.Limit(time=stockfish_seconds),
            )
            move = response.move

        if move is None or move not in board.legal_moves:
            raise RuntimeError("An engine returned no move or an illegal move.")
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "*", "Move cap reached", ply_nodes
    if outcome.winner is None:
        return "1/2-1/2", str(outcome.termination), ply_nodes
    result = "1-0" if outcome.winner == chess.WHITE else "0-1"
    return result, str(outcome.termination), ply_nodes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired-opening SUMEDHA-SL+AB match benchmark."
    )
    parser.add_argument("--sf", required=True, help="Stockfish UCI executable.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/270M"))
    parser.add_argument("--vendor-dir", type=Path, default=Path("third_party"))
    parser.add_argument("--games", type=int, default=20, help="Must be an even number.")
    parser.add_argument("--sumedha-time", type=float, default=1.0, help="Seconds per SUMEDHA move.")
    parser.add_argument("--sf-time", type=float, default=1.0, help="Seconds per Stockfish move.")
    parser.add_argument("--sf-threads", type=int, default=1)
    parser.add_argument("--sf-hash", type=int, default=256)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--opening-min-plies", type=int, default=8)
    parser.add_argument("--opening-max-plies", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output", type=Path, default=Path("benchmark"))
    args = parser.parse_args()

    if args.games < 2 or args.games % 2:
        parser.error("--games must be an even number >= 2 (paired colours).")
    if min(args.sumedha_time, args.sf_time) <= 0:
        parser.error("Both move-time limits must be positive.")
    if args.max_plies < 20:
        parser.error("--max-plies must be at least 20.")
    if args.opening_min_plies < 0 or args.opening_max_plies < args.opening_min_plies:
        parser.error("Invalid opening ply range.")
    sf_path = Path(args.sf).expanduser()
    if not sf_path.is_file():
        parser.error(f"Stockfish executable not found: {sf_path}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = Searchless270M(args.checkpoint_dir, args.vendor_dir)
    searcher = SearchlessPVS(model, hash_mb=64)
    stockfish = chess.engine.SimpleEngine.popen_uci(str(sf_path.resolve()))

    try:
        stockfish.configure({
            "Threads": args.sf_threads,
            "Hash": args.sf_hash,
        })
        openings = make_openings(
            args.games // 2,
            args.seed,
            args.opening_min_plies,
            args.opening_max_plies,
        )

        pgn_path = args.output / "sumedha_sl_ab_vs_stockfish.pgn"
        json_path = args.output / "sumedha_sl_ab_vs_stockfish.json"
        wins = draws = losses = incomplete = 0
        games_report = []
        started = time.perf_counter()

        with pgn_path.open("w", encoding="utf-8") as pgn_file:
            game_number = 0
            for opening_number, opening in enumerate(openings, start=1):
                for sumedha_white in (True, False):
                    game_number += 1
                    searcher.clear()
                    board = opening.copy(stack=True)
                    result, termination, nodes = play_one(
                        board=board,
                        sumedha_white=sumedha_white,
                        searcher=searcher,
                        stockfish=stockfish,
                        sumedha_seconds=args.sumedha_time,
                        stockfish_seconds=args.sf_time,
                        max_plies=args.max_plies,
                    )

                    game = chess.pgn.Game.from_board(board)
                    game.headers["Event"] = "SUMEDHA-SL+AB paired benchmark"
                    game.headers["Site"] = "Local"
                    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                    game.headers["Round"] = str(game_number)
                    game.headers["White"] = "SUMEDHA-SL+AB" if sumedha_white else "Stockfish"
                    game.headers["Black"] = "Stockfish" if sumedha_white else "SUMEDHA-SL+AB"
                    game.headers["Result"] = result
                    game.headers["Termination"] = termination
                    game.headers["OpeningPair"] = str(opening_number)
                    pgn_file.write(str(game))
                    pgn_file.write("\n\n")

                    if result == "*":
                        incomplete += 1
                    else:
                        sumedha_result = (
                            1.0 if (result == "1-0") == sumedha_white
                            else 0.0 if result in ("1-0", "0-1")
                            else 0.5
                        )
                        if sumedha_result == 1.0:
                            wins += 1
                        elif sumedha_result == 0.5:
                            draws += 1
                        else:
                            losses += 1

                    games_report.append({
                        "game": game_number,
                        "opening_pair": opening_number,
                        "sumedha_white": sumedha_white,
                        "result": result,
                        "termination": termination,
                        "plies": board.ply(),
                        "sumedha_search_nodes": nodes,
                        "final_fen": board.fen(),
                    })
                    print(
                        f"Game {game_number:03d}/{args.games}: "
                        f"{'SUMEDHA White' if sumedha_white else 'SUMEDHA Black'} | "
                        f"{result} | {termination} | {board.ply()} plies",
                        flush=True,
                    )

        elapsed = time.perf_counter() - started
        completed = wins + draws + losses
        score = (wins + 0.5 * draws) / completed if completed else None
        report = {
            "engine": "SUMEDHA-SL+AB",
            "opponent": str(sf_path.resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "games_requested": args.games,
            "games_completed": completed,
            "incomplete_games": incomplete,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_rate": score,
            "sumedha_seconds_per_move": args.sumedha_time,
            "stockfish_seconds_per_move": args.sf_time,
            "stockfish_threads": args.sf_threads,
            "stockfish_hash_mb": args.sf_hash,
            "opening_seed": args.seed,
            "paired_colours": True,
            "elapsed_seconds": round(elapsed, 2),
            "pgn_file": str(pgn_path.resolve()),
            "games": games_report,
            "elo_estimate": None,
            "rating_note": (
                "No Elo is assigned by this script. A rating requires a calibrated "
                "opponent pool, larger sample, fixed hardware/time controls and "
                "statistical uncertainty analysis."
            ),
        }
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\nSUMEDHA-SL+AB RESULT")
        print(f"{wins}W {draws}D {losses}L; incomplete={incomplete}")
        print(f"Score: {score if score is not None else 'N/A'}")
        print(f"PGN: {pgn_path.resolve()}")
        print(f"JSON: {json_path.resolve()}")
        print("Elo: not estimated by this pilot harness.")
    finally:
        stockfish.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
