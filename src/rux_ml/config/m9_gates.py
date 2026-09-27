"""Fail-closed reader of the operator-signed M9 keys (PR-044; program ACCEPTANCE C9).

The rux-capital program keeps four M9 thresholds in its operator-signed
``config/gates.yaml`` (``thresholds.<key>.value``). rux-ml never hardcodes them:
:func:`load_m9_gates` verifies the signature (``ssh-keygen -Y verify`` over the
exact bytes it then parses), reads the four values, range-checks them and returns
them with the file's sha256 for every report. Any doubt refuses with
:class:`M9GatesError` — a missing file, a failed signature, a missing key, a
non-numeric or out-of-range value. There is no override.

The key names are the signed contract's identifiers (program D45 #3, C9), not
tunable values.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import yaml

if TYPE_CHECKING:
    from rux_ml.config.m9 import M9GatesConfig

# key -> (lower, lower_inclusive, upper, upper_inclusive): the domain of the quantity.
M9_GATE_KEYS: dict[str, tuple[float, bool, float, bool]] = {
    # a fraction of rows
    "m9_honesty_no_underdeduct_frac_min": (0.0, False, 1.0, True),
    # a relative over-deduction bound
    "m9_honesty_overdeduct_max_rel": (0.0, True, math.inf, False),
    # a relative improvement tolerance
    "m9_learning_curve_tolerance_rel": (0.0, True, math.inf, False),
    # a quantile
    "m9_floor_quantile": (0.0, False, 1.0, False),
}


class M9GatesError(ValueError):
    """The signed M9 keys could not be read with certainty (a ``ValueError``: exit 2)."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"m9 gates: {detail} — refused")


@dataclass(frozen=True)
class M9Gates:
    values: dict[str, float]
    sha256: str
    path: str

    @property
    def no_underdeduct_frac_min(self) -> float:
        return self.values["m9_honesty_no_underdeduct_frac_min"]

    @property
    def overdeduct_max_rel(self) -> float:
        return self.values["m9_honesty_overdeduct_max_rel"]

    @property
    def learning_curve_tolerance_rel(self) -> float:
        return self.values["m9_learning_curve_tolerance_rel"]

    @property
    def floor_quantile(self) -> float:
        return self.values["m9_floor_quantile"]


def _verify(data: bytes, cfg: M9GatesConfig) -> None:
    for name in ("signature_path", "allowed_signers"):
        f = getattr(cfg, name)
        if not f.is_file():
            msg = f"{name} {f} is not a file"
            raise M9GatesError(msg)
    exe = shutil.which("ssh-keygen")
    if exe is None:
        msg = "ssh-keygen not found, so the signature cannot be verified"
        raise M9GatesError(msg)
    argv = [exe, "-Y", "verify", "-f", str(cfg.allowed_signers), "-I", cfg.identity]
    argv += ["-n", cfg.namespace, "-s", str(cfg.signature_path)]
    r = subprocess.run(argv, input=data, capture_output=True, check=False)
    if r.returncode != 0:
        detail = r.stderr.decode(errors="replace").strip()
        msg = f"signature FAILED for {cfg.path} ({cfg.identity}, {cfg.namespace}): {detail}"
        raise M9GatesError(msg)


def _value(thresholds: dict[str, Any], key: str) -> float:
    entry = thresholds.get(key)
    if not isinstance(entry, dict) or "value" not in entry:
        msg = f"{key} is not in thresholds (not yet signed in?)"
        raise M9GatesError(msg)
    v = entry["value"]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(v, bool) or not isinstance(v, int | float):
        msg = f"{key} value {v!r} is not a number"
        raise M9GatesError(msg)
    lo, lo_in, hi, hi_in = M9_GATE_KEYS[key]
    x = float(v)
    ok = (x >= lo if lo_in else x > lo) and (x <= hi if hi_in else x < hi)
    if not ok or math.isnan(x):
        msg = f"{key} value {x} outside {'[' if lo_in else '('}{lo}, {hi}{']' if hi_in else ')'}"
        raise M9GatesError(msg)
    return x


def load_m9_gates(cfg: M9GatesConfig) -> M9Gates:
    """Verify, parse and range-check the four signed M9 keys; refuse on any doubt."""
    if not cfg.path.is_file():
        msg = f"path {cfg.path} is not a file"
        raise M9GatesError(msg)
    data = cfg.path.read_bytes()
    _verify(data, cfg)
    try:
        doc = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        msg = f"{cfg.path} is not YAML: {exc}"
        raise M9GatesError(msg) from exc
    raw: object = cast("dict[str, Any]", doc).get("thresholds") if isinstance(doc, dict) else None
    if not isinstance(raw, dict):
        msg = f"{cfg.path} has no thresholds mapping"
        raise M9GatesError(msg)
    thresholds = cast("dict[str, Any]", raw)
    missing = [k for k in M9_GATE_KEYS if k not in thresholds]
    if missing:
        msg = f"{missing} not in thresholds (not yet signed in?)"
        raise M9GatesError(msg)
    values = {k: _value(thresholds, k) for k in M9_GATE_KEYS}
    return M9Gates(values=values, sha256=hashlib.sha256(data).hexdigest(), path=str(cfg.path))
