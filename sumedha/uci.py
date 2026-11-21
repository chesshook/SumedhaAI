"""UCI interface and neural PUCT search for a trained SUMEDHA checkpoint."""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from .encoding import encode_board, move_to_index
from .model import SumedhaNet


@dataclass
class Edge:
    move: chess.Move
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: "Node | None" = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    board: chess.Board
    visits: int = 0
    edges: list[Edge] = field(default_factory=list)
    expanded: bool = False


class NeuralPUCT:
    def __init__(self, model, device, cpuct: float = 1.5):
        self.model = model
        self.device = device
        self.cpuct = cpuct

    def terminal_value(self, board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    @torch.inference_mode()
    def expand(self, node: Node) -> float:
        board = node.board
        if board.is_game_over(claim_draw=True):
            node.expanded = True
            return self.terminal_value(board)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            node.expanded = True
            return self.terminal_value(board)

        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        indices = torch.tensor(
            [move_to_index(move) for move in legal_moves],
            dtype=torch.long, device=self.device
        )
        legal_logits = logits[0].float().index_select(0, indices)
        probabilities = torch.softmax(legal_logits, dim=0).cpu().numpy()
        node.edges = [
            Edge(move=move, prior=float(probability))
            for move, probability in zip(legal_moves, probabilities)
        ]
        node.expanded = True
        return float(value[0].item())

    def select(self, node: Node) -> Edge:
        root_scale = math.sqrt(node.visits + 1.0)
        best_edge = None
        best_score = -float("inf")
        for edge in node.edges:
            exploration = self.cpuct * edge.prior * root_scale / (1 + edge.visits)
            score = edge.q + exploration
            if score > best_score:
                best_score = score
                best_edge = edge
        if best_edge is None:
            raise RuntimeError("PUCT reached a node without legal edges.")
        return best_edge

    def principal_variation(self, root: Node, maximum: int = 24) -> list[chess.Move]:
        pv = []
        node = root
        for _ in range(maximum):
            if not node.edges:
                break
            edge = max(node.edges, key=lambda candidate: candidate.visits)
            if edge.visits <= 0:
                break
            pv.append(edge.move)
            if edge.child is None:
                break
            node = edge.child
        return pv

    def search(
        self,
        board: chess.Board,
        simulations: int,
        stop_event: threading.Event,
        deadline: float | None = None,
    ) -> tuple[chess.Move | None, int, float, list[chess.Move]]:
        root = Node(board.copy(stack=True))
        root_value = self.expand(root)
        if not root.edges:
            return None, 0, root_value, []

        completed = 0
        while completed < simulations and not stop_event.is_set():
            if completed > 0 and deadline is not None and time.monotonic() >= deadline:
                break
            node = root
            path: list[tuple[Node, Edge]] = []

            # Traverse until an unseen leaf is reached.
            while node.expanded and node.edges:
                edge = self.select(node)
                path.append((node, edge))
                if edge.child is None:
                    child_board = node.board.copy(stack=True)
                    child_board.push(edge.move)
                    edge.child = Node(child_board)
                node = edge.child
                if not node.expanded:
                    break

            if node.board.is_game_over(claim_draw=True):
                leaf_value = self.terminal_value(node.board)
                node.expanded = True
            elif not node.expanded:
                leaf_value = self.expand(node)
            else:
                # A terminal expanded node has no outgoing edges.
                leaf_value = self.terminal_value(node.board)

            # Values alternate perspective at every ply.
            for parent, edge in reversed(path):
                leaf_value = -leaf_value
                edge.visits += 1
                edge.value_sum += leaf_value
                parent.visits += 1
            completed += 1

        if not root.edges:
            return None, completed, root_value, []
        chosen = max(root.edges, key=lambda edge: edge.visits)
        return chosen.move, completed, root_value, self.principal_variation(root)


class SumedhaUCI:
    def __init__(self, checkpoint: str, simulations: int, cpuct: float):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location=self.device)
        config = state.get("model_config") or {"channels": 96, "blocks": 6, "embedding": 64}
        self.model = SumedhaNet(**config).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.searcher = NeuralPUCT(self.model, self.device, cpuct=cpuct)
        self.default_simulations = simulations
        self.board = chess.Board()
        self.search_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def send(self, line: str) -> None:
        print(line, flush=True)

    def stop_search(self) -> None:
        self.stop_event.set()
        thread = self.search_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=30)
        self.search_thread = None

    def set_position(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("position command is empty")
        if tokens[0] == "startpos":
            board = chess.Board()
            remainder = tokens[1:]
        elif tokens[0] == "fen":
            try:
                moves_index = tokens.index("moves")
                fen_tokens = tokens[1:moves_index]
                remainder = tokens[moves_index:]
            except ValueError:
                fen_tokens = tokens[1:]
                remainder = []
            board = chess.Board(" ".join(fen_tokens))
            if not board.is_valid():
                raise ValueError("FEN is not a valid chess position")
        else:
            raise ValueError("position must start with startpos or fen")

        if remainder:
            if remainder[0] != "moves":
                raise ValueError("expected 'moves' after start position")
            for uci in remainder[1:]:
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError(f"illegal move in position command: {uci}")
                board.push(move)
        self.board = board

    def parse_go(self, tokens: list[str]) -> tuple[int, float | None]:
        values = {}
        flags = {"infinite", "ponder"}
        i = 0
        while i < len(tokens):
            key = tokens[i]
            if key in flags:
                values[key] = True
                i += 1
            elif i + 1 < len(tokens):
                try:
                    values[key] = int(tokens[i + 1])
                except ValueError:
                    i += 1
                    continue
                i += 2
            else:
                i += 1

        if "movetime" in values:
            seconds = max(0.05, values["movetime"] / 1000.0)
            deadline = time.monotonic() + max(0.02, seconds - 0.03)
            return 10_000_000, deadline
        if "nodes" in values:
            return max(1, values["nodes"]), None
        if "depth" in values:
            return max(1, values["depth"] * 32), None
        if "infinite" in values or "ponder" in values:
            return 10_000_000, None

        # Basic clock allocation when the GUI supplies clocks instead of movetime.
        remaining_key = "wtime" if self.board.turn == chess.WHITE else "btime"
        increment_key = "winc" if self.board.turn == chess.WHITE else "binc"
        if remaining_key in values:
            remaining = max(0, values[remaining_key]) / 1000.0
            increment = max(0, values.get(increment_key, 0)) / 1000.0
            seconds = min(max(0.05, remaining * 0.035 + increment * 0.8), max(0.05, remaining * 0.25))
            return 10_000_000, time.monotonic() + max(0.02, seconds - 0.03)
        return self.default_simulations, None

    def start_search(self, tokens: list[str]) -> None:
        self.stop_search()
        simulations, deadline = self.parse_go(tokens)
        self.stop_event = threading.Event()
        board_snapshot = self.board.copy(stack=True)

        def worker():
            try:
                move, completed, value, pv = self.searcher.search(
                    board_snapshot, simulations, self.stop_event, deadline
                )
                bounded_value = float(np.clip(value, -0.999, 0.999))
                cp = int(600.0 * np.arctanh(bounded_value))
                pv_text = " ".join(m.uci() for m in pv)
                self.send(
                    f"info nodes {completed} score cp {cp}"
                    + (f" pv {pv_text}" if pv_text else "")
                )
                self.send(f"bestmove {move.uci() if move is not None else '0000'}")
            except Exception as exc:
                print(f"[SUMEDHA] search error: {exc}", file=sys.stderr, flush=True)
                self.send("bestmove 0000")

        self.search_thread = threading.Thread(target=worker, daemon=True, name="sumedha-search")
        self.search_thread.start()

    def loop(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            command, args = tokens[0], tokens[1:]
            try:
                if command == "uci":
                    self.send("id name SUMEDHA AI 0.1")
                    self.send("id author chesshook")
                    self.send("option name Simulations type spin default " + str(self.default_simulations) + " min 1 max 1000000")
                    self.send("uciok")
                elif command == "isready":
                    self.send("readyok")
                elif command == "ucinewgame":
                    self.stop_search()
                    self.board = chess.Board()
                elif command == "position":
                    self.stop_search()
                    self.set_position(args)
                elif command == "go":
                    self.start_search(args)
                elif command == "stop":
                    self.stop_search()
                elif command == "setoption":
                    if len(args) >= 4 and args[0] == "name" and args[1] == "Simulations" and args[2] == "value":
                        self.default_simulations = max(1, int(args[3]))
                elif command == "quit":
                    self.stop_search()
                    break
                elif command in ("ponderhit",):
                    continue
            except Exception as exc:
                print(f"[SUMEDHA] UCI command error ({command}): {exc}", file=sys.stderr, flush=True)
                if command == "go":
                    self.send("bestmove 0000")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--cpuct", type=float, default=1.5)
    args = parser.parse_args()
    if args.simulations < 1 or args.cpuct <= 0:
        raise SystemExit("simulations must be positive and cpuct must be greater than zero.")
    torch.set_num_threads(1)
    engine = SumedhaUCI(args.checkpoint, args.simulations, args.cpuct)
    print(f"[SUMEDHA] Loaded checkpoint on {engine.device}", file=sys.stderr, flush=True)
    engine.loop()


if __name__ == "__main__":
    main()
