"""PR-047: the bridge's equality test on the REAL training set (program PR-024 A13).

Runs only when pointed at a materialized subtree and its harness manifest:

    RUXML_BRIDGE_SOURCE=<training root>/<set>/walk \\
    RUXML_BRIDGE_MANIFEST=<training root>/<set>/manifest.json \\
    uv run pytest tests/integration/test_training_set_bridge_real.py

It computes rux-ml's sidecar in this environment and asserts the set manifest's file
list + sha256 for the subtree equal the files rux-ml's loader opened, and its copy of
rux-ml's data_hash (``ruxml_sidecars.<subtree>.data_hash``, required — PR-052) equals
this environment's. The recorded run is program C6 / C11 evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rux_ml.data.bridge import check_bridge, training_set_sidecar
from tests.conftest import repo_oracle_cfg

SOURCE = os.environ.get("RUXML_BRIDGE_SOURCE")
MANIFEST = os.environ.get("RUXML_BRIDGE_MANIFEST")


@pytest.mark.integration
@pytest.mark.skipif(not (SOURCE and MANIFEST), reason="set RUXML_BRIDGE_SOURCE and _MANIFEST")
def test_the_real_subtree_equals_its_harness_manifest() -> None:
    assert SOURCE is not None and MANIFEST is not None
    side = training_set_sidecar(Path(SOURCE), oracle=repo_oracle_cfg())
    result = check_bridge(side, json.loads(Path(MANIFEST).read_text()), subtree=Path(SOURCE).name)
    assert result["equal"] is True
