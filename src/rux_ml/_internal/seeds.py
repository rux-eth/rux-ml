"""Per-trial seed derivation (per PR-013 + D9).

Single canonical surface for the reproducibility-grade seed pipeline. Every
trial — sweep or one-off — derives a :class:`SeedBag` of four child seeds
(split / cv / sampler / xgb) from either a study-level master entropy + the
Optuna ``trial.number``, or from a previously-recorded ``entropy_hex`` (used
by :mod:`rux_ml.registry.promote` to re-fit the trial with identical seeds).

Pattern (numpy canonical, per
[the SeedSequence docs](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html)
and [the scientific-python NumPy RNG best-practices post](https://blog.scientific-python.org/numpy/numpy-rng/)):

1. ``parent = SeedSequence(entropy=master, spawn_key=(trial_number,))``
2. Capture a stable per-trial 128-bit entropy from ``parent.generate_state(4, dtype=uint32)``.
   This value depends on ``(master, trial_number)`` and serves as the
   ``entropy_hex`` recorded in ``TrialAttrs.entropy_hex``.
3. Rebuild a fresh ``SeedSequence`` from that per-trial entropy and call
   ``.spawn(4)`` to derive the four child streams.
4. Each child yields one uint32; mask to ``int32`` range (``[0, 2**31)``)
   so the values are stable across (Python int → JSON → SQLite TEXT)
   round-trips and never trigger XGBoost's negative-int handling.

Reconstruction from a recorded ``entropy_hex`` (skips step 1+2 — caller
supplies the per-trial entropy directly): see :func:`make_seed_bag_from_hex`.

This module's numpy import is **lazy inside the function bodies** so that
``trial_runner.py`` can call ``pin_threads(cfg.memory)`` BEFORE the heavy
imports per PR-008. Top-level imports stay stdlib + pydantic only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import numpy as np


# XGBoost / sklearn ``random_state`` accept any non-negative int. We mask to
# 31 bits so the values fit in a signed int32, which is the safest envelope
# for SQLite TEXT round-trips (Optuna's user_attrs serializer) and matches
# sklearn's documented expectation (`random_state: int in [0, 2**32 - 1]`,
# but `int32` keeps us inside the signed range that JSON handles trivially).
_INT32_MAX = (1 << 31) - 1


class SeedBag(BaseModel):
    """Four derived seeds for one trial + the per-trial entropy fingerprint.

    ``entropy_hex`` is the **per-trial** entropy (post-``spawn_key`` tagging),
    NOT the study-level master. Storing the per-trial entropy is what lets
    :func:`make_seed_bag_from_hex` reconstruct an identical bag without
    needing the master or the trial number — the contract PR-010's
    ``registry/promote.py`` relies on for deterministic re-fit at promotion.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entropy_hex: str
    split_seed: int
    cv_seed: int
    sampler_seed: int
    xgb_seed: int


def _seed_from_child(child: np.random.SeedSequence) -> int:
    """Convert one spawned ``SeedSequence`` to an ``int32``-safe seed."""
    raw = child.generate_state(1, dtype="uint32")
    return int(raw[0]) & _INT32_MAX


def _derive_from_entropy(entropy: int) -> tuple[int, int, int, int]:
    """Spawn four child seeds from a per-trial entropy integer."""
    import numpy as np  # noqa: PLC0415 — lazy per PR-008 module-load order

    per_trial = np.random.SeedSequence(entropy=entropy)
    children = per_trial.spawn(4)
    return (
        _seed_from_child(children[0]),
        _seed_from_child(children[1]),
        _seed_from_child(children[2]),
        _seed_from_child(children[3]),
    )


def make_seed_bag(*, master_entropy: int | None, trial_number: int) -> SeedBag:
    """Derive a per-trial :class:`SeedBag` from a study master + trial number.

    The recorded ``entropy_hex`` depends deterministically on the
    ``(master_entropy, trial_number)`` pair — two trials with the same pair
    produce the same bag; distinct ``trial_number``s produce distinct bags
    even with the same master.

    If ``master_entropy`` is ``None``, NumPy auto-pins a fresh entropy from
    ``os.urandom`` on first use; the per-trial ``entropy_hex`` recorded in
    the returned bag fully captures the derivation so reconstruction does
    not need the master afterwards.
    """
    import numpy as np  # noqa: PLC0415 — lazy per PR-008 module-load order

    # Step 1+2: spawn-key-tagged parent → 128-bit per-trial entropy.
    parent = np.random.SeedSequence(entropy=master_entropy, spawn_key=(trial_number,))
    pool = parent.generate_state(4, dtype="uint32")
    per_trial_entropy = int.from_bytes(pool.tobytes(), byteorder="little")
    entropy_hex = f"{per_trial_entropy:032x}"

    # Step 3+4: rebuild from per-trial entropy → spawn 4 children → mask to int32.
    split_s, cv_s, sampler_s, xgb_s = _derive_from_entropy(per_trial_entropy)

    return SeedBag(
        entropy_hex=entropy_hex,
        split_seed=split_s,
        cv_seed=cv_s,
        sampler_seed=sampler_s,
        xgb_seed=xgb_s,
    )


def make_seed_bag_from_hex(entropy_hex: str) -> SeedBag:
    """Reconstruct a :class:`SeedBag` from a previously-recorded ``entropy_hex``.

    Used by :mod:`rux_ml.registry.promote` to re-fit a model with the exact
    seeds the originating trial used. Without this, the re-fit at promotion
    would use a different split / xgb_seed and the registered bundle would
    not match the trial's evaluated metrics.

    Raises:
        ValueError: if ``entropy_hex`` is not a valid hex string.
    """
    per_trial_entropy = int(entropy_hex, 16)
    split_s, cv_s, sampler_s, xgb_s = _derive_from_entropy(per_trial_entropy)
    return SeedBag(
        entropy_hex=entropy_hex,
        split_seed=split_s,
        cv_seed=cv_s,
        sampler_seed=sampler_s,
        xgb_seed=xgb_s,
    )
