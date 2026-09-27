"""Shared pytest fixtures for the rux-ml test suite.

Per-layer fixtures live in each ``tests/<layer>/conftest.py``. This file
holds cross-layer helpers — currently the PR-013 :class:`SeedBag` and
:class:`EnvironmentVersions` test doubles used wherever a test needs to
construct a ``TrialAttrs`` directly (rather than going through the full
trial body that produces them). PR-040 adds the oracle-quarantine fixtures,
which read ``[data.oracle]`` from the repo's ``configs/base.toml`` so no test
restates the namespace or tag-file values.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rux_ml._internal.env import EnvironmentVersions
from rux_ml._internal.seeds import SeedBag, make_seed_bag

if TYPE_CHECKING:
    from rux_ml.config.data import OracleQuarantineConfig

REPO_BASE_TOML = Path(__file__).resolve().parents[1] / "configs" / "base.toml"


@pytest.fixture
def seed_bag() -> SeedBag:
    """Deterministic :class:`SeedBag` for tests (entropy=42, trial_number=0)."""
    return make_seed_bag(master_entropy=42, trial_number=0)


@pytest.fixture
def env_versions() -> EnvironmentVersions:
    """Synthetic :class:`EnvironmentVersions` for tests — no nvidia-smi shell-out.

    Mirrors a CPU-only dev host: ``gpu_model`` / ``driver_version`` are
    ``None``; the required fields carry stable test placeholders so tests
    asserting on TrialAttrs round-trips have known values.
    """
    return EnvironmentVersions(
        xgboost_version="3.2.0",
        cuda_runtime_version="12.9",
        omp_threads=1,
        image_digest="sha256:test",
        gpu_model=None,
        driver_version=None,
    )


def repo_oracle_values() -> dict[str, str]:
    """``[data.oracle]`` from the repo's ``configs/base.toml`` (PR-040) — the single source."""
    return dict(tomllib.loads(REPO_BASE_TOML.read_text())["data"]["oracle"])


def repo_oracle_cfg() -> OracleQuarantineConfig:
    """The repo's :class:`OracleQuarantineConfig`; plain-function twin of ``oracle_cfg``.

    For module-level test helpers that build configs or hash data without
    fixture access (``from ..conftest import repo_oracle_cfg``).
    """
    from rux_ml.config.data import OracleQuarantineConfig  # noqa: PLC0415

    return OracleQuarantineConfig(**repo_oracle_values())


def repo_oracle_toml() -> str:
    """A ``[data.oracle]`` table (repo values) to append to a test-written TOML."""
    values = repo_oracle_values()
    return "\n[data.oracle]\n" + "".join(f'{k} = "{v}"\n' for k, v in values.items())


@pytest.fixture
def oracle_values() -> dict[str, str]:
    return repo_oracle_values()


@pytest.fixture
def oracle_cfg() -> OracleQuarantineConfig:
    return repo_oracle_cfg()


@pytest.fixture
def oracle_toml() -> str:
    return repo_oracle_toml()


# ---------- PR-044: an operator-signed gates file, signed here with a throwaway key ----------

# The four signed keys and program D45 #3's proposed values (test data, not code).
M9_GATE_VALUES: dict[str, float] = {
    "m9_honesty_no_underdeduct_frac_min": 0.90,
    "m9_honesty_overdeduct_max_rel": 0.30,
    "m9_learning_curve_tolerance_rel": 0.05,
    "m9_floor_quantile": 0.90,
}
M9_GATE_IDENTITY = "operator@test"
M9_GATE_NAMESPACE = "test-gates"


def write_signed_gates(directory: Path, thresholds: dict[str, object]) -> dict[str, str]:
    """Write ``gates.yaml`` with ``thresholds`` (program layout), sign it with a fresh
    ed25519 key via ``ssh-keygen -Y sign``, and return the ``[m9.gates]`` settings."""
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    key = directory / "key"
    if not key.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    body = "version: 1\nthresholds:\n" + "".join(
        f"  {k}: {{value: {json.dumps(v)}, unit: ratio, label: bg}}\n"
        for k, v in thresholds.items()
    )
    gates = directory / "gates.yaml"
    gates.write_text(body)
    sig = directory / "gates.yaml.sig"
    sig.unlink(missing_ok=True)
    subprocess.run(
        ["ssh-keygen", "-q", "-Y", "sign", "-f", str(key), "-n", M9_GATE_NAMESPACE, str(gates)],
        check=True,
        capture_output=True,
    )
    signers = directory / "allowed_signers"
    signers.write_text(f"{M9_GATE_IDENTITY} {(directory / 'key.pub').read_text().strip()}\n")
    return {
        "path": str(gates),
        "signature_path": str(sig),
        "allowed_signers": str(signers),
        "identity": M9_GATE_IDENTITY,
        "namespace": M9_GATE_NAMESPACE,
    }


@pytest.fixture
def signed_m9_gates(tmp_path: Path) -> dict[str, str]:
    return write_signed_gates(tmp_path / "gates", dict(M9_GATE_VALUES))


def m9_gates_overrides(gates: dict[str, str]) -> list[str]:
    """``--set`` arguments pointing ``[m9.gates]`` at a signed fixture."""
    out: list[str] = []
    for k, v in gates.items():
        out += ["--set", f"m9.gates.{k}={v}"]
    return out
