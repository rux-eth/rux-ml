"""Structural hygiene test for ``configs/base.toml`` discriminator tables.

Per PR-022: when a ``base.toml`` table corresponds to a Pydantic
discriminated union (``[cv]``, ``[training]``, ``[solving]``), only
fields present in *every* variant of that union may live at the
base-table position. Variant-specific knobs (``kfold.shuffle``,
``xgboost.tree_method``, ``time_series.gap``, ``cpcv.embargo_size`` ...)
belong in ``configs/problems/<name>.toml``.

**This is a config (TOML content) rule, not a runtime behavior change.**
Pydantic's ``extra="forbid"`` on ``StrictModel`` and ``deep_merge``'s
plain-dict semantics are both working as designed — the resulting
``ValidationError`` is a *good* diagnostic. The rule prevents
variant-specific defaults from leaking across the discriminator on
``deep_merge`` and tripping the strict validator when a problem TOML
switches ``kind``.

Surfaced live during the 2026-05-18 first-real-dataset run: base.toml
had ``[cv] shuffle = true`` (a ``KFoldCV``-only field); any problem
TOML setting ``kind = "time_series"`` failed with ``extra_forbidden``
because ``TimeSeriesSplitCV`` has no ``shuffle`` field. The fix is
this test (so a future contributor adding ``subsample = 0.8`` to
``[training]`` — which would similarly break ``kind = "lightgbm"`` —
fails CI, not the user) plus removal of the offending fields.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic.fields import FieldInfo

from rux_ml.config import RuxMLConfig
from rux_ml.config.cv import CVConfig
from rux_ml.config.solving import SolvingConfig
from rux_ml.config.training import TrainingConfig

BASE_TOML_PATH = Path(__file__).resolve().parents[2] / "configs" / "base.toml"


def _variants_and_discriminator(annotated_union: Any) -> tuple[tuple[type, ...], str]:
    """Extract variants + discriminator field name from an Annotated discriminated union.

    Handles ``Annotated[A | B | C, Field(discriminator="kind")]`` shape (the
    project convention for ``CVConfig`` / ``TrainingConfig`` / ``SolvingConfig``).
    Returns the discriminator field name (e.g., ``"kind"`` or ``"type"``) so
    the test caller can confirm which key Pydantic dispatches on, even though
    it isn't directly consulted by the body of the variant-coverage check.
    """
    args = get_args(annotated_union)
    if len(args) < 2:
        msg = f"expected Annotated[Union, Field(discriminator=...)], got {annotated_union!r}"
        raise TypeError(msg)

    union_type = args[0]
    variants = get_args(union_type)
    if not variants:
        msg = f"no variants in union {union_type!r}"
        raise TypeError(msg)

    discriminator: str | None = None
    for meta in args[1:]:
        if isinstance(meta, FieldInfo) and meta.discriminator is not None:
            # FieldInfo.discriminator may be a plain str or a pydantic Discriminator
            # object (when supplied with a callable). Coerce both to a name string
            # for diagnostic display only — the variant-coverage check below does
            # not consume the discriminator value semantically.
            disc = meta.discriminator
            discriminator = disc if isinstance(disc, str) else str(disc)
            break
    if discriminator is None:
        msg = f"could not find Field(discriminator=...) in {annotated_union!r}"
        raise TypeError(msg)

    return variants, discriminator


# (TOML section name at base-table position, discriminated-union type) pairs.
# Extend when a new top-level discriminated union becomes settable in
# ``configs/base.toml``. ``[tuning.search_space.*]`` is NOT included because
# search-space entries live in study TOMLs, not at the base level.
DISCRIMINATED_TABLES: list[tuple[str, Any]] = [
    ("cv", CVConfig),
    ("training", TrainingConfig),
    ("solving", SolvingConfig),
]


@pytest.fixture(scope="module")
def base_toml() -> dict[str, Any]:
    with BASE_TOML_PATH.open("rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize(
    ("section", "union"),
    DISCRIMINATED_TABLES,
    ids=[name for name, _ in DISCRIMINATED_TABLES],
)
def test_base_toml_discriminator_table_only_holds_common_fields(
    section: str,
    union: Any,
    base_toml: dict[str, Any],
) -> None:
    """Every field at ``base.toml [<section>]`` must exist in every union variant.

    Variant-specific fields trip Pydantic's ``extra="forbid"`` when a problem
    TOML switches ``kind`` to a sibling variant. Failure message names the
    offending key and the variants missing it — direct enough to fix without
    re-reading this docstring.
    """
    table = base_toml.get(section)
    if not table:
        pytest.skip(f"[{section}] not present in base.toml — nothing to check")

    variants, discriminator = _variants_and_discriminator(union)

    offenders: list[tuple[str, list[str]]] = []
    for key in table:
        missing_from = [variant.__name__ for variant in variants if key not in variant.model_fields]
        if missing_from:
            offenders.append((key, missing_from))

    if offenders:
        details = "\n".join(f"  - {key!r}: missing from {missing}" for key, missing in offenders)
        msg = (
            f"[{section}] in base.toml contains variant-specific field(s):\n"
            f"{details}\n"
            f"Discriminator: {discriminator!r}. Move the offending key(s) out "
            f"of base.toml into configs/problems/<name>.toml, OR add the field "
            f"to the missing variants' schemas if it is genuinely common to "
            f"every variant of {union.__name__}."
        )
        pytest.fail(msg)


def test_discriminated_tables_list_matches_schema() -> None:
    """Tripwire: every top-level discriminated union appearing in ``RuxMLConfig``
    must be covered by ``DISCRIMINATED_TABLES``.

    If a new layer adds a discriminated-union config (e.g., ``ReportingConfig``
    later), this test fails until the maintainer adds its row above. Prevents
    silent under-coverage of the structural rule.
    """
    covered_unions = {union for _, union in DISCRIMINATED_TABLES}
    expected_unions: set[Any] = set()
    for field_info in RuxMLConfig.model_fields.values():
        annotation = field_info.annotation
        if annotation is None:
            continue
        # The discriminated-union type aliases reside as Annotated[Union, Field(...)]
        # OR as the union directly when wrapped via Optional. We use the same
        # detector helper to canonicalize.
        try:
            _variants_and_discriminator(annotation)
        except TypeError:
            continue
        expected_unions.add(annotation)

    missing = expected_unions - covered_unions
    missing_names = [u.__name__ if hasattr(u, "__name__") else repr(u) for u in missing]
    assert not missing, (
        f"DISCRIMINATED_TABLES does not cover all top-level discriminated unions "
        f"in RuxMLConfig. Missing: {missing_names}. "
        f"Add the (section_name, union_type) row above and re-run."
    )
