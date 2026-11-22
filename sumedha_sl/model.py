"""Frozen Searchless Chess 270M action-value inference bridge.

Loads Google's official Haiku/Orbax checkpoint. No model weights are modified.
The upstream repository is cloned into third_party/searchless_chess.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable

import chess
import numpy as np


@dataclasses.dataclass
class _PredictorProtocol:
    initial_params: object
    predict: object


def _upstream_imports(vendor_dir: Path):
    """Import only Searchless modules needed for inference.

    The upstream transformer imports constants.Predictor for typing and
    construction. Importing upstream constants also pulls in training-only
    Apache Beam/Grain dependencies, so provide the tiny Predictor protocol
    it actually uses. No model implementation is replaced.
    """
    vendor_dir = vendor_dir.resolve()
    if not vendor_dir.is_dir():
        raise FileNotFoundError(
            f"Missing upstream checkout: {vendor_dir}. "
            "Run scripts/setup_searchless_kaggle.sh first."
        )
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))

    import searchless_chess.src as searchless_src

    shim_name = "searchless_chess.src.constants"
    if shim_name not in sys.modules:
        shim = types.ModuleType(shim_name)
        shim.Predictor = _PredictorProtocol
        sys.modules[shim_name] = shim
        setattr(searchless_src, "constants", shim)

    from searchless_chess.src import tokenizer, transformer, utils
    return tokenizer, transformer, utils


class Searchless270M:
    """Batched action-value inference using the official 270M parameters."""

    STEP = 6_400_000
    RETURN_BUCKETS = 128
    WIN_PROB_CP_SCALE = 0.00368208

    def __init__(
        self,
        checkpoint_dir: str | Path,
        vendor_dir: str | Path = "third_party",
        batch_size: int = 32,
        cache_size: int = 50_000,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        import haiku as hk  # noqa: F401; required by the upstream transformer

        self.jax = jax
        self.jnp = jnp
        self.batch_size = int(batch_size)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[tuple, Dict[str, float]] = OrderedDict()

        vendor = Path(vendor_dir)
        # Accept either the third_party parent or the actual checkout path.
        if (vendor / "searchless_chess").is_dir():
            vendor = vendor / "searchless_chess"
        self.tokenizer, transformer, self.utils = _upstream_imports(vendor.parent)

        config = transformer.TransformerConfig(
            vocab_size=self.utils.NUM_ACTIONS,
            output_size=self.RETURN_BUCKETS,
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=self.tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=8,
            num_layers=16,
            embedding_dim=1024,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        predictor = transformer.build_transformer_predictor(config)

        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        params_path = self.checkpoint_dir / str(self.STEP) / "params"
        if not params_path.exists():
            raise FileNotFoundError(
                f"Searchless checkpoint parameters not found: {params_path}\n"
                "Expected the official 270M checkpoint, step 6400000."
            )

        initial_params = predictor.initial_params(
            rng=jax.random.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        )
        restore_args = ocp.checkpoint_utils.construct_restore_args(initial_params)
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        self.params = checkpointer.restore(
            str(params_path), restore_args=restore_args
        )

        self.predict = jax.jit(predictor.predict)
        _, self.return_bucket_values = self.utils.get_uniform_buckets_edges_values(
            self.RETURN_BUCKETS
        )
        self.return_bucket_values = np.asarray(
            self.return_bucket_values, dtype=np.float32
        )

        devices = jax.devices()
        print(
            f"[SUMEDHA] Loaded Searchless 270M step {self.STEP}; "
            f"JAX devices={devices}; batch={self.batch_size}",
            file=sys.stderr,
            flush=True,
        )

        # Force a small compile/inference check before search begins.
        self._warmup()

    @staticmethod
    def _position_key(board: chess.Board) -> tuple:
        # Include halfmove clock because the draw rule affects state value.
        return (
            board.board_fen(),
            board.turn,
            int(board.castling_rights),
            board.ep_square,
            board.halfmove_clock,
        )

    def _warmup(self) -> None:
        board = chess.Board()
        moves = list(board.legal_moves)
        values = self.action_values(board, moves)
        if len(values) != len(moves) or not all(
            np.isfinite(v) and 0.0 <= v <= 1.0 for v in values.values()
        ):
            raise RuntimeError("Searchless warm-up returned invalid action values.")

    def _predict_sequences(self, sequences: np.ndarray) -> np.ndarray:
        results = []
        for offset in range(0, len(sequences), self.batch_size):
            chunk = sequences[offset : offset + self.batch_size]
            count = len(chunk)
            if count < self.batch_size:
                padding = np.repeat(chunk[-1:], self.batch_size - count, axis=0)
                chunk = np.concatenate([chunk, padding], axis=0)
            output = self.predict(
                params=self.params,
                targets=self.jnp.asarray(chunk, dtype=self.jnp.int32),
                rng=None,
            )
            log_probs = np.asarray(self.jax.device_get(output[:, -1, :]))
            results.append(log_probs[:count])
        return np.concatenate(results, axis=0)

    def action_values(
        self,
        board: chess.Board,
        legal_moves: Iterable[chess.Move] | None = None,
    ) -> Dict[str, float]:
        """Return P(win) expectation for each legal move, from mover's POV."""
        moves = list(board.legal_moves if legal_moves is None else legal_moves)
        if not moves:
            return {}

        key = self._position_key(board)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        try:
            action_ids = [self.utils.MOVE_TO_ACTION[m.uci()] for m in moves]
        except KeyError as exc:
            raise ValueError(
                f"Searchless action vocabulary does not contain move {exc.args[0]}"
            ) from exc

        fen_tokens = self.tokenizer.tokenize(board.fen()).astype(np.int32)
        state_batch = np.repeat(fen_tokens[None, :], len(moves), axis=0)
        actions = np.asarray(action_ids, dtype=np.int32)[:, None]
        dummy_returns = np.zeros((len(moves), 1), dtype=np.int32)
        sequences = np.concatenate(
            [state_batch, actions, dummy_returns], axis=1
        )
        log_probs = self._predict_sequences(sequences)
        probabilities = np.exp(log_probs.astype(np.float64))
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-30)
        win_probabilities = probabilities @ self.return_bucket_values
        values = {
            move.uci(): float(np.clip(value, 0.0, 1.0))
            for move, value in zip(moves, win_probabilities)
        }

        if self.cache_size:
            self._cache[key] = values
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return values

    @classmethod
    def probability_to_cp(cls, probability: float) -> int:
        """Invert Searchless' published centipawn-to-win-probability map."""
        p = float(np.clip(probability, 1e-5, 1.0 - 1e-5))
        cp = np.log(p / (1.0 - p)) / cls.WIN_PROB_CP_SCALE
        return int(np.clip(round(cp), -5000, 5000))
