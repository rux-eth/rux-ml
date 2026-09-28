# PR-052: The bridge's equality test reads the `data_hash` the harness actually writes

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** R-14 track, on the rux-capital program lead's instruction of 2026-09-27 (fix PR-051's Q23). Program ACCEPTANCE C6 / C11: "rux-ml's `data_hash` is shown on the real set to cover the file list its loader opened … the harness manifest carries rux-ml's `data_hash` (asserted equal by a test on the real set)". Covered by the operator's standing approval of the R-14 track; the merge is operator-gated. **Stacked on PR-051**, which provides the harness-shaped `c6_set` fixture and `harness_manifest_ref`.

## Research findings

### State Assessment (2026-09-27)

Baselines: rux-ml `pr-051/harness-manifest-id` @ `1fb5e40`; rumpy-harness `training/materialize.py` @ `ed19e0b` (read-only); the real set's `manifest.json` on the desktop (`~/projects/rux-capital/m9-training-v03/98c848d277ab-83a175d4b812/`, one read-only `cat`, 72 days materialized at the time).

- **The real shape.** The top-level keys are `code_dirty, code_sha, config_sha256, days, days_in_window_missing, excluded_days, feature_config_sha256, feature_params, feed_globs, kind, label_config_sha256, label_params, lockbox, read_set, ruxml_sidecars, schema_file_matches, schema_sha256, set, stream_manifests, subtrees, versions, window`. `subtrees.<t>` holds `{bytes, censored, kind, rows, subtree: "<t>/", view, files}`, and each file is `{bytes, bytes_per_row, day, path: "<day>.parquet" (relative to the subtree), rows, sha256}`. `ruxml_sidecars` is `{"fill": null, "walk": null}` because rux-ml has not written a sidecar yet. Once it has, the harness's `_sidecar` lists `{path, sha256, data_hash, rux_ml_git_sha, polars_version}`. **There is no top-level `files` and no `ruxml_data_hash`.**
- **Who writes `ruxml_data_hash`: nothing.** A grep of the harness, the program scripts and rux-ml finds only rux-ml's own reader and tests. The harness's test asserts `"ruxml_data_hash" not in view`.
- **The defect.** PR-047's `check_bridge` read top-level `files` and an optional top-level `ruxml_data_hash`. Against the harness's per-subtree view (top-level `files`, no hash), it passed on the file list alone and exit was 0. Against the set manifest it failed for the wrong reason (no `files`). The new CLI test, run on `c6_set` against PR-051's code, showed the file-list-only pass: `assert 0 == 2`.

**Decisions (routine):**
- `--manifest` / `check_bridge(sidecar, manifest, *, subtree)` take the harness **set manifest**. That is the file that carries rux-ml's `data_hash`, and PR-051's model manifest id is its sha256, so the bridge checks the same file the model cites. The files come from `subtrees.<subtree>.files` (only `path` and `sha256` are compared; the harness's extra keys are ignored). The data_hash comes from `ruxml_sidecars.<subtree>.data_hash`.
- **Fail closed:** an absent or `null` entry, or no `ruxml_sidecars` at all, is refused with "lists no rux-ml data_hash for <t>/". The file list alone never passes. The flow: write the sidecar (no `--manifest`), the harness lists it on its next `write_manifest`, then check.
- **The top-level `ruxml_data_hash` path is removed** (nothing real writes it). A manifest of PR-047's assumed shape is refused, and the refusal says to pass the set's `manifest.json`.
- `listed_data_hash(manifest, subtree)` is shared with PR-051's `harness_manifest_ref`, so the two read the field one way.
- The `c6_set` fixture (`tests/conftest.py::write_harness_manifest`) now writes the real shape: per-file `day / rows / bytes / bytes_per_row`, per-subtree totals, and `ruxml_sidecars.<t>` as `null` or the listed sidecar (a sidecar file written and hashed as the harness's `_sidecar` records it).

### Synthesis / Gate

**Outcome: Confirm.** ARCHITECTURE's bridge section now describes the real shape and the fail-closed rule. Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `rux_ml.data.bridge`: `check_bridge(..., subtree=)` on the set-manifest shape; `listed_data_hash`; module docstring. `rux_ml.cli.data.bridge` passes `subtree = data.source_path.name`; help text.
- Tests: `tests/data/test_bridge.py`: the shape; `test_the_bridge_fails_closed_without_the_harness_listed_data_hash[null|no_entry|no_ruxml_sidecars]`; `test_the_pr047_assumed_shape_is_not_read`; the equal case and the four differences re-expressed. `tests/cli/test_m9_bridge_cli.py`: the real flow on `c6_set`. `tests/cli/test_data_subcommands.py`: the set-manifest shape. `tests/integration/test_training_set_bridge_real.py`: `RUXML_BRIDGE_MANIFEST` = `<set>/manifest.json`, `subtree=`. Fixture: `tests/conftest.py::write_harness_manifest` (the real shape).
- `docs/ARCHITECTURE.md`, `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`, PR-051's Q23.

## Dependencies

PR-051 (stacked; merge #56, then #57, then this).

## Verification criteria

- [x] On `c6_set` through the real CLI: the view (file list only) exits 2, where PR-051's code exited 0 (**failed first**, `assert 0 == 2`). The set manifest before the harness lists the sidecar exits 2 with "no rux-ml data_hash for walk/". A listed `data_hash` that is not rux-ml's exits 2. With rux-ml's listed, it exits 0 and records `{subtree, files, data_hash, equal}`.
- [x] `check_bridge` refuses a `null` / missing entry and a missing `ruxml_sidecars`, and refuses PR-047's assumed shape. These tests failed first: against PR-051's code, with a `TypeError` for the new `subtree=`. The old code had no reading of `ruxml_sidecars` at all.
- [x] PR-051's promote tests pass on the richer fixture shape (the shared `listed_data_hash`).
- [x] Default suite, ruff check, `ruff format --check` on the touched files, basedpyright `src/`: see the commit message.

## Notes — for the program lead

- **Flow:** run the check with `--output` pointed somewhere other than the sidecar the harness listed (e.g. `<set>/<t>.ruxml-check.json`). The CLI writes its record to `--output`, so reusing the listed sidecar path changes that file's sha256 after the harness recorded it in `ruxml_sidecars.<t>.sha256`.
- The recorded real-set run (C6 / C11) waits until the harness lists rux-ml's sidecars. As of this reading both are `null`.
