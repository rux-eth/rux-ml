"""Tests for :mod:`rux_ml._internal.seeds` (per PR-013)."""

from __future__ import annotations

import pytest

from rux_ml._internal.seeds import SeedBag, make_seed_bag, make_seed_bag_from_hex


def test_same_master_and_trial_number_produce_same_bag() -> None:
    """Reproducibility floor: ``(master_entropy, trial_number)`` is the seed."""
    a = make_seed_bag(master_entropy=42, trial_number=0)
    b = make_seed_bag(master_entropy=42, trial_number=0)
    assert a == b


def test_distinct_trial_numbers_produce_distinct_bags() -> None:
    """``spawn_key=(trial_number,)`` ensures per-trial isolation."""
    a = make_seed_bag(master_entropy=42, trial_number=0)
    b = make_seed_bag(master_entropy=42, trial_number=1)
    assert a.entropy_hex != b.entropy_hex
    # All four child seeds should also differ (negligible collision probability).
    assert a.split_seed != b.split_seed
    assert a.cv_seed != b.cv_seed
    assert a.sampler_seed != b.sampler_seed
    assert a.xgb_seed != b.xgb_seed


def test_distinct_masters_produce_distinct_bags() -> None:
    """Different master entropy → different bags even at the same trial number."""
    a = make_seed_bag(master_entropy=42, trial_number=0)
    b = make_seed_bag(master_entropy=43, trial_number=0)
    assert a.entropy_hex != b.entropy_hex
    assert a.xgb_seed != b.xgb_seed


def test_none_master_produces_random_bag() -> None:
    """``master_entropy=None`` auto-pins from os.urandom → different bags each call."""
    a = make_seed_bag(master_entropy=None, trial_number=0)
    b = make_seed_bag(master_entropy=None, trial_number=0)
    assert a.entropy_hex != b.entropy_hex


def test_all_seeds_fit_in_int32_range() -> None:
    """Child seeds masked to [0, 2**31) so SQLite TEXT round-trips never overflow."""
    bag = make_seed_bag(master_entropy=0xDEADBEEF, trial_number=7)
    int32_max = (1 << 31) - 1
    for seed in (bag.split_seed, bag.cv_seed, bag.sampler_seed, bag.xgb_seed):
        assert 0 <= seed <= int32_max


def test_entropy_hex_is_32_lowercase_hex_chars() -> None:
    """128-bit entropy → 32 hex chars; format matches `f"{int:032x}"` output."""
    bag = make_seed_bag(master_entropy=12345, trial_number=0)
    assert len(bag.entropy_hex) == 32
    assert all(c in "0123456789abcdef" for c in bag.entropy_hex)


def test_make_seed_bag_from_hex_round_trip() -> None:
    """The contract registry/promote.py depends on: ``entropy_hex`` alone
    is sufficient to reconstruct the bag."""
    original = make_seed_bag(master_entropy=0xC0FFEE, trial_number=3)
    reconstructed = make_seed_bag_from_hex(original.entropy_hex)
    assert reconstructed == original


def test_make_seed_bag_from_hex_rejects_invalid_hex() -> None:
    with pytest.raises(ValueError):
        make_seed_bag_from_hex("not-a-hex-string")


def test_seed_bag_is_frozen() -> None:
    """``SeedBag`` is immutable to prevent accidental mutation after recording."""
    bag = make_seed_bag(master_entropy=42, trial_number=0)
    with pytest.raises(Exception):  # noqa: B017 — pydantic.ValidationError or attr error
        bag.split_seed = 9999  # type: ignore[misc]


def test_make_seed_bag_serializable_to_optuna_user_attrs() -> None:
    """All fields are plain ints + str so Optuna's JSON serializer accepts them."""
    bag = make_seed_bag(master_entropy=42, trial_number=0)
    dumped = bag.model_dump()
    for value in dumped.values():
        assert isinstance(value, (int, str))


def test_make_seed_bag_returns_seed_bag_type() -> None:
    bag = make_seed_bag(master_entropy=42, trial_number=0)
    assert isinstance(bag, SeedBag)
