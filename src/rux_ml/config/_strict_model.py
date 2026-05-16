"""Base class for per-layer config models.

Sets `extra="forbid"` so unknown TOML keys raise validation errors instead of
being silently ignored, and `validate_assignment=True` so post-construction
mutations are validated.
"""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Pydantic v2 base with strict semantics for config layers."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,  # mutated by D16 trial-config derivation via model_copy(update=...)
    )
