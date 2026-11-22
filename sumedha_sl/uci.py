"""UCI entry point for SUMEDHA-SL+AB."""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import chess

from .model import Searchless270M
from .search import MATE_THRESHOLD, SearchlessPVS


class SumedhaUCI:
    def __init__(self, checkpoint_dir: str, vendor_dir: str, hash_mb: int):
        self.model = Searchless270M(
            checkpoint_dir=checkpoint_dir,
            vendor_dir=vendor_dir,
        )
        self.searcher = SearchlessPVS(self.model, hash_mb=hash_mb)
        self.board = chess.Board()
        self.default_depth = 64
        self.default_movetime = 1.0
        self._worker = None
        self._stop = threading.Event()
        self._output_lock = threading.Lock()

    def send(self, line: str) -> None:
        with self._output_lock:
            print(line, flush=True)

    def stop_search(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._stop.set()
            self._worker.join()
        self._worker = None

    def set_position(self, args: list[str]) -> None:
        if not args:
            raise ValueError("position command is empty")
        if args[0] == "startpos":
            board = chess.Board()
            remainder = args[1:]
        elif args[0] == "fen":
            try:
                moves_idx = args.index("moves")
                fen_tokens = args[1:moves_idx]
                remainder = args[moves_idx + 1:]
            except ValueError:
                fen_tokens = args[1:]
                remainder = []
            board = chess.Board(" ".join(fen_tokens))
            if not board.is_valid():
                raise ValueError("invalid FEN")
        else:
            raise ValueError("expected 'startpos' or 'fen'")

        if remainder:
            for uci_move in remainder:
                move = chess.Move.from_uci(uci_move)
                if move not in board.legal_moves:
                    raise ValueError(f"illegal move in position: {uci_move}")
                board.push(move)
        self.board = board

    @staticmethod
    def _go_limits(tokens: list[str], board: chess.Board):
        values = {}
        flags = {"infinite", "ponder"}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in flags:
                values[token] = True
                i += 1
            elif i + 1 < len(tokens):
                try:
                    values[token] = int(tokens[i + 1])
                    i += 2
                except ValueError:
                    i += 1
            else:
                i += 1

        depth = values.get("depth", 64)
        nodes = values.get("nodes")
        if "movetime" in values:
            seconds = max(0.02, values["movetime"] / 1000.0 - 0.04)
            return depth, seconds, nodes
        if "infinite" in values or "ponder" in values:
            return depth, None, nodes

        clock_key = "wtime" if board.turn == chess.WHITE else "btime"
        inc_key = "winc" if board.turn == chess.WHITE else "binc"
        if clock_key in values:
            remaining = max(0, values[clock_key]) / 1000.0
            increment = max(0, values.get(inc_key, 0)) / 1000.0
            moves_to_go = max(1, values.get("movestogo", 30))
            allocation = remaining / moves_to_go + 0.7 * increment
            allocation = min(allocation, max(0.02, remaining * 0.08))
            seconds = max(0.02, allocation - 0.03)
            return depth, seconds, nodes
        return depth, 1.0, nodes

    def start_search(self, tokens: list[str]) -> None:
        self.stop_search()
        depth, seconds, nodes = self._go_limits(tokens, self.board)
        board_snapshot = self.board.copy(stack=True)
        self._stop = threading.Event()

        def report(info: dict) -> None:
            score = int(info["score_cp"])
            if abs(score) >= MATE_THRESHOLD:
                mate_plies = max(0, 30000 - abs(score))
                mate_moves = (mate_plies + 1) // 2
                score_text = f"mate {mate_moves if score > 0 else -mate_moves}"
            else:
                score_text = f"cp {score}"
            pv_text = " ".join(m.uci() for m in info["pv"])
            self.send(
                f"info depth {info['depth']} score {score_text} "
                f"nodes {info['nodes']} nps {info['nps']} "
                f"time {info['time_ms']} pv {pv_text}"
            )

        def worker():
            try:
                result = self.searcher.search(
                    board_snapshot,
                    max_depth=max(1, depth),
                    time_limit=seconds,
                    node_limit=nodes,
                    stop_event=self._stop,
                    info_callback=report,
                )
                if result.best_move is None:
                    self.send("bestmove 0000")
                else:
                    self.send(f"bestmove {result.best_move.uci()}")
            except Exception as exc:
                self.send(f"info string search error: {type(exc).__name__}: {exc}")
                self.send("bestmove 0000")

        self._worker = threading.Thread(
            target=worker, name="sumedha-search", daemon=True
        )
        self._worker.start()

    def loop(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            command, args = tokens[0], tokens[1:]
            try:
                if command == "uci":
                    self.send("id name SUMEDHA-SL+AB 0.1")
                    self.send("id author SUMEDHA AI")
                    self.send("option name Hash type spin default 64 min 8 max 2048")
                    self.send("uciok")
                elif command == "isready":
                    self.send("readyok")
                elif command == "ucinewgame":
                    self.stop_search()
                    self.board = chess.Board()
                    self.searcher.clear()
                elif command == "position":
                    self.stop_search()
                    self.set_position(args)
                elif command == "go":
                    self.start_search(args)
                elif command == "stop":
                    self.stop_search()
                elif command == "setoption":
                    if len(args) >= 4 and args[0] == "name" and args[1] == "Hash" and args[2] == "value":
                        self.searcher.set_hash(int(args[3]))
                    elif len(args) >= 3 and args[0] == "name" and args[1] == "Clear" and args[2] == "Hash":
                        self.searcher.clear()
                elif command == "quit":
                    self.stop_search()
                    return
                elif command in ("debug", "register", "ponderhit"):
                    continue
            except Exception as exc:
                self.send(f"info string command error ({command}): {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SUMEDHA Searchless 270M + PVS/alpha-beta UCI engine"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/270M"),
        help="Extracted official Searchless 270M checkpoint directory.",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=Path("third_party"),
        help="Parent directory containing third_party/searchless_chess.",
    )
    parser.add_argument("--hash", type=int, default=64, help="TT size in MB (approximate).")
    args = parser.parse_args()

    if args.hash < 8:
        parser.error("--hash must be at least 8 MB")

    torch_threads = 1
    try:
        import torch
        torch.set_num_threads(torch_threads)
    except ImportError:
        pass

    engine = SumedhaUCI(
        checkpoint_dir=str(args.checkpoint_dir),
        vendor_dir=str(args.vendor_dir),
        hash_mb=args.hash,
    )
    engine.loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
