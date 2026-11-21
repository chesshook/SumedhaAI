"""Fast runtime, encoding, model and PUCT smoke tests."""
from __future__ import annotations

import threading

import chess
import torch

from .encoding import POLICY_SIZE, encode_board, legal_move_indices, move_to_index
from .model import SumedhaNet
from .uci import NeuralPUCT


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

    # Exercise legal-move masking and a complete pair of tree simulations.
    searcher = NeuralPUCT(model, torch.device("cpu"), cpuct=1.5)
    best, completed, _, _ = searcher.search(start, 2, threading.Event())
    assert best in start.legal_moves
    assert completed == 2

    print(
        "SUMEDHA SELF-TEST PASSED | board encoding | move indexing | "
        "en-passant | promotions | policy/value heads | PUCT search",
        flush=True,
    )


if __name__ == "__main__":
    main()
