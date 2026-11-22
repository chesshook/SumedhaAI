"""Run a small model-and-search diagnostic on the frozen Searchless checkpoint."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess

from .model import Searchless270M
from .search import SearchlessPVS


POSITIONS = [
    ("Initial position", chess.STARTING_FEN),
    ("Queen mate in one", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
    ("Rook mate in one", "6k1/5ppp/8/8/8/8/8/3RK3 w - - 0 1"),
    ("Scholar's mate", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/2N5/PPPP1PPP/R1B1K1NR w KQkq - 4 4"),
    ("Must evade check", "4r1k1/8/8/8/8/8/8/4K3 w - - 0 1"),
    ("Promotion decision", "7k/1P6/2K5/8/8/8/8/8 w - - 0 1"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/270M"))
    parser.add_argument("--vendor-dir", type=Path, default=Path("third_party"))
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--positions", type=int, default=len(POSITIONS))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.depth < 1 or args.seconds <= 0 or args.positions < 1:
        parser.error("depth, seconds and positions must be positive")

    model = Searchless270M(
        checkpoint_dir=args.checkpoint_dir,
        vendor_dir=args.vendor_dir,
        batch_size=args.batch_size,
    )
    engine = SearchlessPVS(model, hash_mb=64)

    print("\nSUMEDHA-SL+AB | frozen Searchless 270M diagnostic")
    print("No training. No Stockfish runtime. No Elo claim.")
    print(f"JAX devices: {model.devices}")
    print(f"Data-parallel action batch: {model.batch_size}")

    for name, fen in POSITIONS[:args.positions]:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(f"Invalid diagnostic position: {name}")

        inference_started = time.perf_counter()
        action_values = model.action_values(board)
        inference_elapsed = time.perf_counter() - inference_started
        ranked = sorted(
            action_values.items(), key=lambda item: item[1], reverse=True
        )[:5]

        started = time.perf_counter()
        result = engine.search(
            board,
            max_depth=args.depth,
            time_limit=args.seconds,
        )
        elapsed = time.perf_counter() - started

        print("\n" + "=" * 72)
        print(name, "|", "White" if board.turn else "Black", "to move")
        print(f"Root action-value inference: {inference_elapsed:.2f}s")
        print("Top Searchless action values:")
        for uci, probability in ranked:
            move = chess.Move.from_uci(uci)
            print(
                f"  {board.san(move):10s}  "
                f"expected-return={probability:.4f}"
            )
        if result.best_move is None:
            print("SUMEDHA move: none (terminal position)")
        else:
            print("SUMEDHA move:", board.san(result.best_move), result.best_move.uci())
        print(
            f"Search: depth={result.depth}, nodes={result.nodes}, "
            f"score={result.score_cp:+d} cp*, time={elapsed:.2f}s"
        )
        print("PV:", board.variation_san(result.pv) if result.pv else "(no completed PV)")
        if result.depth == 0:
            print("SEARCH STATUS: no full PVS iteration completed; move is neural-ranked fallback.")
        print("* cp is an approximate inverse mapping of Searchless' win-probability scale.")

    print("\nDiagnostic complete. This is not a playing-strength or Elo benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
