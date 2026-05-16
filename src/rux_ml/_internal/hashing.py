"""Hashing primitives shared across the workbench.

Per D9 + D13: use mature Python bindings (`xxhash`) for fast non-cryptographic
hashing of bytes; SHA-256 (stdlib) for canonical config / data hashes that
participate in reproducibility provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import xxhash

# 64 KiB matches the GNU coreutils sha256sum default for streaming reads.
_DEFAULT_CHUNK_SIZE = 65_536


def xxh3_64_bytes(data: bytes) -> str:
    """Hex digest of `xxhash.xxh3_64` for an in-memory byte string."""
    return xxhash.xxh3_64(data).hexdigest()


def xxh3_64_file(path: Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    """Streaming xxh3_64 hex digest of a file's bytes (for the bytes_hash component
    of the composite data_hash defined in D9)."""
    h = xxhash.xxh3_64()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Sorted-key, no-whitespace JSON for deterministic hashing.

    `default=str` lets us serialize Path, datetime, etc. uniformly.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_canonical(obj: Any) -> str:
    """SHA-256 hex digest of `canonical_json(obj)`."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()
