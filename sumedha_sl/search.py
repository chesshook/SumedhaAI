"""Searchless-guided adversarial search: PVS, alpha-beta, TT and quiescence.

The frozen Searchless action-value model supplies:
  1. move ordering at every expanded position;
  2. the static evaluation at the search horizon.

The search itself is independent Python chess search. Stockfish is never
launched or queried by this module.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import chess

MATE = 30_000
MATE_THRESHOLD = 29_000
INF = 32_000

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    bound: int  # 0 exact, 1 lower bound, 2 upper bound
    best_move: Optional[str]


@dataclass
class SearchResult:
    best_move: Optional[chess.Move]
    score_cp: int
    depth: int
    nodes: int
    elapsed: float
    pv: List[chess.Move]


class SearchStopped(Exception):
    pass


class SearchlessPVS:
    """Iterative-deepening negamax PVS driven by Searchless action values."""

    def __init__(
        self,
        evaluator,
        hash_mb: int = 64,
        max_qply: int = 8,
        use_lmr: bool = True,
    ):
        self.evaluator = evaluator
        self.max_qply = int(max_qply)
        self.use_lmr = bool(use_lmr)
        self.tt_max_entries = max(8_000, int(hash_mb) * 1_000)
        self.tt: Dict[tuple, TTEntry] = {}
        self.killers: List[List[Optional[str]]] = [
            [None, None] for _ in range(128)
        ]
        self.history: Dict[tuple, int] = {}
        self.nodes = 0
        self.start_time = 0.0
        self.deadline: Optional[float] = None
        self.node_limit: Optional[int] = None
        self.stop_event = threading.Event()
        self.pv_table: List[List[chess.Move]] = [
            [] for _ in range(128)
        ]

    @staticmethod
    def _terminal_score(board: chess.Board, ply: int) -> Optional[int]:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0
        # Values are always from the side-to-move perspective.
        if outcome.winner == board.turn:
            return MATE - ply
        return -MATE + ply

    @staticmethod
    def _tt_key(board: chess.Board) -> tuple:
        # Include recent reversible history to reduce unsafe reuse across
        # positions with different repetition rights.
        recent = tuple(move.uci() for move in board.move_stack[-12:])
        return (
            board.board_fen(),
            board.turn,
            int(board.castling_rights),
            board.ep_square,
            board.halfmove_clock,
            recent,
        )

    @staticmethod
    def _score_to_tt(score: int, ply: int) -> int:
        if score > MATE_THRESHOLD:
            return score + ply
        if score < -MATE_THRESHOLD:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score: int, ply: int) -> int:
        if score > MATE_THRESHOLD:
            return score - ply
        if score < -MATE_THRESHOLD:
            return score + ply
        return score

    def clear(self) -> None:
        self.tt.clear()
        self.history.clear()
        self.killers = [[None, None] for _ in range(128)]

    def set_hash(self, hash_mb: int) -> None:
        if hash_mb < 1:
            raise ValueError("Hash must be positive.")
        self.tt_max_entries = max(8_000, int(hash_mb) * 1_000)
        if len(self.tt) > self.tt_max_entries:
            self.tt.clear()

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise SearchStopped
        if self.node_limit is not None and self.nodes >= self.node_limit:
            raise SearchStopped
        # Check clocks at every node; model inference itself cannot be
        # interrupted, but search stops as soon as that inference returns.
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise SearchStopped

    def _visit(self) -> None:
        self.nodes += 1
        self._check_stop()

    def _action_values(self, board: chess.Board) -> tuple[list[chess.Move], dict[str, float]]:
        moves = list(board.legal_moves)
        if not moves:
            return [], {}
        values = self.evaluator.action_values(board, moves)
        return moves, values

    def _static_eval(self, board: chess.Board) -> int:
        moves, values = self._action_values(board)
        if not moves:
            terminal = self._terminal_score(board, 0)
            return 0 if terminal is None else terminal
        best_probability = max(values[m.uci()] for m in moves)
        return self.evaluator.probability_to_cp(best_probability)

    def _move_order_score(
        self,
        board: chess.Board,
        move: chess.Move,
        action_values: dict[str, float],
        tt_move: Optional[str],
        ply: int,
    ) -> float:
        uci = move.uci()
        if tt_move is not None and uci == tt_move:
            return 1_000_000_000.0

        score = action_values.get(uci, 0.5) * 10_000.0
        if move.promotion:
            score += 20_000.0 + PIECE_VALUE.get(move.promotion, 0)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if board.is_en_passant(move):
                victim_value = PIECE_VALUE[chess.PAWN]
            else:
                victim_value = PIECE_VALUE.get(
                    victim.piece_type, 0
                ) if victim else 0
            attacker = board.piece_at(move.from_square)
            attacker_value = PIECE_VALUE.get(
                attacker.piece_type, 0
            ) if attacker else 0
            score += 15_000.0 + victim_value * 10.0 - attacker_value
        elif ply < len(self.killers):
            if uci == self.killers[ply][0]:
                score += 12_000.0
            elif uci == self.killers[ply][1]:
                score += 11_000.0
            score += self.history.get((board.turn, uci), 0)
        return score

    def _ordered(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        values: dict[str, float],
        tt_move: Optional[str],
        ply: int,
    ) -> list[chess.Move]:
        return sorted(
            moves,
            key=lambda m: self._move_order_score(
                board, m, values, tt_move, ply
            ),
            reverse=True,
        )

    def _record_quiet_cutoff(
        self, board: chess.Board, move: chess.Move, depth: int, ply: int
    ) -> None:
        uci = move.uci()
        if ply < len(self.killers) and uci not in self.killers[ply]:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = uci
        key = (board.turn, uci)
        self.history[key] = min(
            20_000, self.history.get(key, 0) + depth * depth
        )

    def _quiescence(
        self, board: chess.Board, alpha: int, beta: int, ply: int, qply: int
    ) -> int:
        self._visit()
        if ply < len(self.pv_table):
            self.pv_table[ply] = []
        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal

        in_check = board.is_check()
        stand_pat = -INF
        if not in_check:
            stand_pat = self._static_eval(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

        if qply >= self.max_qply:
            if not in_check:
                return stand_pat
            # At the safety cap, still examine every legal check evasion once.
            evasions = list(board.legal_moves)
            if not evasions:
                return -MATE + ply
            best_evasion = -INF
            for evasion in evasions:
                board.push(evasion)
                try:
                    score = -self._static_eval(board)
                finally:
                    board.pop()
                best_evasion = max(best_evasion, score)
            return best_evasion

        moves, values = self._action_values(board)
        if not moves:
            terminal = self._terminal_score(board, ply)
            return 0 if terminal is None else terminal

        if in_check:
            tactical = moves  # All evasions must be considered.
        else:
            tactical = [
                m for m in moves
                if board.is_capture(m) or m.promotion is not None
            ]

        tactical = self._ordered(board, tactical, values, None, ply)
        for move in tactical:
            board.push(move)
            try:
                score = -self._quiescence(
                    board, -beta, -alpha, ply + 1, qply + 1
                )
            finally:
                board.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score
                if ply < len(self.pv_table):
                    self.pv_table[ply] = [move] + self.pv_table[ply + 1]
        return alpha

    def _negamax(
        self, board: chess.Board, depth: int,
        alpha: int, beta: int, ply: int,
    ) -> int:
        self._visit()
        if ply >= 120:
            return self._static_eval(board)

        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, 0)

        if ply < len(self.pv_table):
            self.pv_table[ply] = []

        alpha_original = alpha
        beta_original = beta
        key = self._tt_key(board)
        entry = self.tt.get(key)
        tt_move = entry.best_move if entry else None

        if entry is not None and entry.depth >= depth:
            tt_score = self._score_from_tt(entry.score, ply)
            if entry.bound == 0:
                return tt_score
            if entry.bound == 1:
                alpha = max(alpha, tt_score)
            elif entry.bound == 2:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

        moves, action_values = self._action_values(board)
        if not moves:
            terminal = self._terminal_score(board, ply)
            return 0 if terminal is None else terminal

        ordered = self._ordered(board, moves, action_values, tt_move, ply)
        best_score = -INF
        best_move = None
        in_check = board.is_check()

        for index, move in enumerate(ordered):
            quiet = not board.is_capture(move) and move.promotion is None
            board.push(move)
            try:
                if index == 0:
                    score = -self._negamax(
                        board, depth - 1, -beta, -alpha, ply + 1
                    )
                else:
                    reduction = 0
                    if (
                        self.use_lmr and depth >= 4 and index >= 4
                        and quiet and not in_check and not board.is_check()
                    ):
                        reduction = 1
                    score = -self._negamax(
                        board, depth - 1 - reduction,
                        -alpha - 1, -alpha, ply + 1
                    )
                    if score > alpha and reduction:
                        score = -self._negamax(
                            board, depth - 1, -alpha - 1, -alpha, ply + 1
                        )
                    if alpha < score < beta:
                        score = -self._negamax(
                            board, depth - 1, -beta, -alpha, ply + 1
                        )
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
                if ply < len(self.pv_table):
                    self.pv_table[ply] = [move] + self.pv_table[ply + 1]
            if alpha >= beta:
                if quiet:
                    self._record_quiet_cutoff(board, move, depth, ply)
                break

        if best_move is None:
            return self._static_eval(board)

        bound = 0
        if best_score <= alpha_original:
            bound = 2
        elif best_score >= beta_original:
            bound = 1

        if len(self.tt) >= self.tt_max_entries:
            self.tt.clear()
        self.tt[key] = TTEntry(
            depth=depth,
            score=self._score_to_tt(best_score, ply),
            bound=bound,
            best_move=best_move.uci(),
        )
        return best_score

    def search(
        self,
        board: chess.Board,
        max_depth: int = 20,
        time_limit: Optional[float] = None,
        node_limit: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
        info_callback: Optional[Callable[[dict], None]] = None,
    ) -> SearchResult:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        self.stop_event = stop_event or threading.Event()
        self.nodes = 0
        self.start_time = time.monotonic()
        self.deadline = (
            self.start_time + max(0.01, float(time_limit))
            if time_limit is not None else None
        )
        self.node_limit = node_limit
        self.pv_table = [[] for _ in range(128)]

        initial = self._terminal_score(board, 0)
        if initial is not None:
            return SearchResult(None, initial, 0, 0, 0.0, [])

        best_move = next(iter(board.legal_moves), None)
        best_score = 0
        completed_depth = 0
        best_pv: list[chess.Move] = []

        # Establish a neural-policy fallback before starting deeper search.
        # If one transformer inference exceeds a very short move budget, the
        # engine still returns its best Searchless-ranked legal root move.
        try:
            self._check_stop()
            root_moves, root_values = self._action_values(board)
            if root_moves:
                best_move = max(
                    root_moves, key=lambda move: root_values[move.uci()]
                )
        except SearchStopped:
            pass

        for depth in range(1, max_depth + 1):
            try:
                score = self._negamax(board, depth, -INF, INF, 0)
            except SearchStopped:
                break

            completed_depth = depth
            best_score = score
            candidate_pv = self.pv_table[0]
            if candidate_pv:
                best_pv = list(candidate_pv)
                best_move = best_pv[0]

            elapsed = max(time.monotonic() - self.start_time, 1e-6)
            info = {
                "depth": depth,
                "score_cp": score,
                "nodes": self.nodes,
                "nps": int(self.nodes / elapsed),
                "time_ms": int(elapsed * 1000),
                "pv": list(best_pv),
            }
            if info_callback is not None:
                info_callback(info)

            if abs(score) >= MATE_THRESHOLD:
                break
            if self.deadline is not None and time.monotonic() >= self.deadline:
                break
            if self.node_limit is not None and self.nodes >= self.node_limit:
                break

        elapsed = max(time.monotonic() - self.start_time, 0.0)
        return SearchResult(
            best_move=best_move,
            score_cp=best_score,
            depth=completed_depth,
            nodes=self.nodes,
            elapsed=elapsed,
            pv=best_pv,
        )
