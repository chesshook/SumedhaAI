"""Fast runtime, encoding, model and PUCT smoke tests."""
from __future__ import annotations

import threading

import chess
import torch

from .encoding import POLICY_SIZE, encode_board, legal_move_indices, move_to_index
from .model import SumedhaNet
from .uci import NeuralPUCT
from .data import parse_eval_move, read_eval_row


def main():
    torch.set_num_threads(1)
    start = chess.Board()
    assert start.is_valid()
    encoded = encode_board(start)
    assert encoded.shape == (19, 8, 8)
    assert encoded.dtype.name == "float32"

    moves, indices = legal_move_indices(start)
    assert len(moves) == 20
    assert len(set(indices.tolist())) == 20
    assert int(indices.min()) >= 0 and int(indices.max()) < POLICY_SIZE
    assert all(move_to_index(move) == int(index) for move, index in zip(moves, indices))

    # En-passant and promotion moves must have valid, distinct policy slots.
    ep = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        ep.push_uci(uci)
    ep_moves = list(ep.legal_moves)
    ep_capture = chess.Move.from_uci("e5d6")
    assert ep_capture in ep_moves and move_to_index(ep_capture) < POLICY_SIZE

    promotion = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    promotion_moves = [m for m in promotion.legal_moves if m.promotion]
    assert len(promotion_moves) == 4
    assert len({move_to_index(m) for m in promotion_moves}) == 4

    model = SumedhaNet(channels=32, blocks=2, embedding=32).eval()
    sample = torch.from_numpy(encoded).unsqueeze(0)
    with torch.inference_mode():
        policy, value = model(sample)
    assert policy.shape == (1, POLICY_SIZE)
    assert value.shape == (1,)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()

    # Check precomputed evaluation import (four-field FEN, White POV score).
    labelled = read_eval_row({
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -",
        "line": "e7e5 e2e4",
        "depth": 30,
        "knodes": 50000,
        "cp": 120,
        "mate": None,
    })
    assert labelled is not None
    _, eval_move_index, black_to_move_value = labelled
    assert eval_move_index == move_to_index(chess.Move.from_uci("e7e5"))
    assert black_to_move_value < 0.0

    # Check Lichess UCI_Chess960 king-to-rook notation is canonicalised
    # to orthodox UCI, using a minimal position with an unobstructed legal castle.
    castle_board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    assert castle_board.is_valid()
    parsed_castle = parse_eval_move(castle_board, "e1h1")
    assert parsed_castle == chess.Move.from_uci("e1g1"), (
        f"UCI960 e1h1 should canonicalise to e1g1; got {parsed_castle!r}; "
        f"legal castles={[m.uci() for m in castle_board.legal_moves if castle_board.is_castling(m)]}"
    )

    # Exercise legal-move masking and a complete pair of tree simulations.
    searcher = NeuralPUCT(model, torch.device("cpu"), cpuct=1.5)
    best, completed, _, _ = searcher.search(start, 2, threading.Event())
    assert best in start.legal_moves
    assert completed == 2

    print(
        "SUMEDHA SELF-TEST PASSED | board encoding | move indexing | "
        "en-passant | promotions | Parquet-data parser | UCI960 castling | "
        "policy/value heads | PUCT search",
        flush=True,
    )


if __name__ == "__main__":
    main()
