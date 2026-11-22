"""Fast tests for the search core; uses a deterministic dummy evaluator."""
from __future__ import annotations

import chess

from sumedha_sl.search import SearchlessPVS


class DummyEvaluator:
    @staticmethod
    def action_values(board, moves=None):
        moves = list(board.legal_moves if moves is None else moves)
        return {move.uci(): 0.5 for move in moves}

    @staticmethod
    def probability_to_cp(probability):
        return 0


def main():
    engine = SearchlessPVS(DummyEvaluator(), hash_mb=8)

    start = chess.Board()
    result = engine.search(start, max_depth=2, time_limit=2.0)
    assert result.best_move in start.legal_moves
    assert result.depth >= 1

    mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = engine.search(mate, max_depth=2, time_limit=2.0)
    assert result.best_move is not None
    after = mate.copy()
    after.push(result.best_move)
    assert after.is_checkmate(), (
        f"Expected mate in one, got {result.best_move.uci()}"
    )

    checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    result = engine.search(checkmate, max_depth=2, time_limit=1.0)
    assert result.best_move is None

    print("SUMEDHA-SL SEARCH SELFTEST PASSED")
    print("  start-position legal move")
    print("  mate-in-one found")
    print("  terminal checkmate handled")


if __name__ == "__main__":
    main()
