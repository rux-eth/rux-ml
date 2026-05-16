"""Sentinel test that the no-op-subcommand stub harness is empty post-PR-010.

Every verb group's real-body test now lives in its dedicated module:

- ``tests/cli/test_data_subcommands.py`` — PR-004 (data verbs)
- ``tests/cli/test_train_subcommand.py`` — PR-006 (train)
- ``tests/cli/test_tune_subcommands.py`` — PR-007 / PR-008 (tune)
- ``tests/cli/test_runs_subcommands.py`` — PR-009 (runs)
- ``tests/cli/test_registry_subcommands.py`` — PR-010 (registry)

If you add a new stub subcommand later, route its smoke test here.
"""

from __future__ import annotations
