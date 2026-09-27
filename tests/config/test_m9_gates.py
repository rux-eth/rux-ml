"""PR-044: the fail-closed reader of the operator-signed M9 keys (program ACCEPTANCE C9).

The four keys live in the rux-capital program's ``config/gates.yaml``, signed by
the operator (``ssh-keygen -Y``). rux-ml verifies the signature itself before it
reads a value, with the signer identity, namespace and allowed-signers file taken
from rux-ml's own config — never from the signed file — and refuses on any doubt.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from rux_ml.config import M9GatesConfig
from rux_ml.config.m9_gates import M9_GATE_KEYS, M9GatesError, load_m9_gates
from tests.conftest import M9_GATE_VALUES, write_signed_gates

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")


def test_reads_the_four_signed_values_and_the_file_sha(signed_m9_gates: dict[str, str]) -> None:
    got = load_m9_gates(M9GatesConfig(**signed_m9_gates))  # type: ignore[arg-type]
    assert set(M9_GATE_KEYS) == set(M9_GATE_VALUES)
    assert got.values == M9_GATE_VALUES
    assert got.sha256 == hashlib.sha256(Path(signed_m9_gates["path"]).read_bytes()).hexdigest()
    assert got.no_underdeduct_frac_min == 0.90
    assert got.overdeduct_max_rel == 0.30


def test_an_edit_after_signing_is_refused(signed_m9_gates: dict[str, str]) -> None:
    p = Path(signed_m9_gates["path"])
    p.write_text(p.read_text().replace("0.3", "0.9"))
    with pytest.raises(M9GatesError, match="signature"):
        load_m9_gates(M9GatesConfig(**signed_m9_gates))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["signature_path", "allowed_signers", "path"])
def test_a_missing_file_is_refused(signed_m9_gates: dict[str, str], field: str) -> None:
    Path(signed_m9_gates[field]).unlink()
    with pytest.raises(M9GatesError):
        load_m9_gates(M9GatesConfig(**signed_m9_gates))  # type: ignore[arg-type]


def test_a_signature_by_another_identity_is_refused(signed_m9_gates: dict[str, str]) -> None:
    cfg = dict(signed_m9_gates, identity="someone@else")
    with pytest.raises(M9GatesError, match="signature"):
        load_m9_gates(M9GatesConfig(**cfg))  # type: ignore[arg-type]


def test_a_missing_key_is_refused_by_name(tmp_path: Path) -> None:
    values = dict(M9_GATE_VALUES)
    del values["m9_floor_quantile"]
    gates = write_signed_gates(tmp_path / "g", values)
    with pytest.raises(M9GatesError, match="m9_floor_quantile"):
        load_m9_gates(M9GatesConfig(**gates))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("m9_honesty_no_underdeduct_frac_min", 1.5),
        ("m9_honesty_no_underdeduct_frac_min", True),
        ("m9_honesty_overdeduct_max_rel", -0.1),
        ("m9_learning_curve_tolerance_rel", "five percent"),
        ("m9_floor_quantile", 0.0),
        ("m9_floor_quantile", None),
    ],
)
def test_a_malformed_value_is_refused(tmp_path: Path, key: str, bad: object) -> None:
    gates = write_signed_gates(tmp_path / "g", {**M9_GATE_VALUES, key: bad})
    with pytest.raises(M9GatesError, match=key):
        load_m9_gates(M9GatesConfig(**gates))  # type: ignore[arg-type]


PROGRAM = Path(__file__).resolve().parents[3] / "rux-capital" / "program" / "config"


@pytest.mark.skipif(not (PROGRAM / "gates.yaml").exists(), reason="program repo not beside rux-ml")
def test_the_real_program_file_today_verifies_but_lacks_the_keys() -> None:
    """The real signed file (read-only): its signature checks out under the program's
    own identity and namespace, and the reader refuses because the operator has not
    yet signed the four M9 keys in (C9: "signed before the first Stage 3 study")."""
    cfg = M9GatesConfig(
        path=PROGRAM / "gates.yaml",
        signature_path=PROGRAM / "gates.yaml.sig",
        allowed_signers=PROGRAM / "allowed_signers",
        identity="operator@rumpy",
        namespace="rumpy-gates",
    )
    try:
        load_m9_gates(cfg)
    except M9GatesError as exc:
        assert "signature" not in str(exc), exc  # the signature itself verifies
        assert "m9_honesty_no_underdeduct_frac_min" in str(exc)
    else:  # once the operator signs the keys in, the reader returns them
        pass


REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize(
    ("problem", "honesty"), [("m9_fill_frac", False), ("m9_markout_bp", True), ("m9_walk_bp", True)]
)
def test_repo_problems_run_the_honesty_test_where_d45_says(problem: str, honesty: bool) -> None:
    """D45 #3: fill fraction -> Brier; markout and walk -> MAE with the honesty test."""
    from rux_ml.config import RuxMLConfig  # noqa: PLC0415

    cfg = RuxMLConfig.from_layers(
        REPO_CONFIGS / "base.toml", problem=problem, problems_dir=REPO_CONFIGS / "problems"
    )
    assert cfg.m9 is not None and cfg.m9.gates is not None
    assert cfg.m9.signed_error_honesty is honesty


@pytest.mark.skipif(not (PROGRAM / "gates.yaml").exists(), reason="program repo not beside rux-ml")
def test_repo_problems_name_the_program_files_own_signer() -> None:
    import yaml  # noqa: PLC0415

    from rux_ml.config import RuxMLConfig  # noqa: PLC0415

    block = yaml.safe_load((PROGRAM / "gates.yaml").read_text())["signature"]
    cfg = RuxMLConfig.from_layers(
        REPO_CONFIGS / "base.toml", problem="m9_walk_bp", problems_dir=REPO_CONFIGS / "problems"
    )
    assert cfg.m9 is not None and cfg.m9.gates is not None
    assert (cfg.m9.gates.identity, cfg.m9.gates.namespace) == (
        block["identity"],
        block["namespace"],
    )
