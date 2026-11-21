"""Chess-board and move-index encodings used by SUMEDHA.

Squares use python-chess indexing: a1=0, b1=1, ..., h8=63.
The board tensor is [19, 8, 8] and keeps the same square orientation.
"""
from __future__ import annotations

import numpy as np
import chess

INPUT_PLANES = 19
PROMOTION_TO_INDEX = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}
INDEX_TO_PROMOTION = {v: k for k, v in PROMOTION_TO_INDEX.items()}
POLICY_SIZE = 64 * 64 * 5


def encode_board(board: chess.Board) -> np.ndarray:
    """Return uint8/binary-ish feature planes, shape [19, 8, 8]."""
    x = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        plane = piece.piece_type - 1 if piece.color == chess.WHITE else 6 + piece.piece_type - 1
        x[plane, rank, file] = 1.0

    x[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    x[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    x[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    x[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    x[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        x[17, rank, file] = 1.0
    x[18, :, :] = min(board.halfmove_clock, 100) / 100.0
    return x


def move_to_index(move: chess.Move) -> int:
    """Map a legal chess.Move to one of 20,480 policy slots."""
    promotion = PROMOTION_TO_INDEX.get(move.promotion)
    if promotion is None:
        raise ValueError(f"Unsupported promotion piece: {move.promotion}")
    return (move.from_square * 64 + move.to_square) * 5 + promotion


def legal_move_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    indices = np.asarray([move_to_index(m) for m in moves], dtype=np.int64)
    return moves, indices
