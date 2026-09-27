# PR-047: The training-set data-hash bridge — rux-ml's sidecar and the equality test

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** Added to the R-14 track by the program lead on 2026-09-26 (operator "go" ≈ 22:05 CDT on program PR-024's amendment A13): "rux-ml computes its own `data_hash` for each training subtree (`fill/`, `walk/`) in rux-ml's environment and records it in a sidecar with the rux-ml commit SHA and the Polars version … the equality test (the harness manifest's file list + sha256 vs the rux-ml loader's opened files and data_hash) is a rux-ml test." Also program D45 #6 / C11 ("rux-ml's `data_hash` is shown on the real set to cover the file list its loader opened"). **Stacked on PR-046**; operator-gated merge.

## Research findings

### State Assessment (2026-09-26)

Baseline: `pr-046/m9-learning-curve` @ `25b77f0`.
- `data_hash = bytes_hash | logical_hash` (`runs/provenance.py`): xxh3 per file over `iter_parquet_files` (sorted `rglob('*.parquet')`) and a Polars struct row hash over the same explicit file list (`data/versioning.py`). The loader is `pl.scan_parquet(source)` on the directory (`data/loaders.py`).
- PR-040's named successor (ii): the hash's file list and the loader's can differ — Polars follows symlinked subdirectories, `rglob` does not; hive keys enter the frame but not the hash.
- **Probe:** `pl.scan_parquet(src, include_file_paths=…)` (polars 1.40.1) names each row's source file — the loader's file set straight from Polars, not re-derived. A tree mixing hive `day=` directories with a plain one makes Polars itself raise `ShapeError` at `collect_schema` (the loader would fail the same way).

**Decisions** (light):
- Opened files = distinct `include_file_paths` values of the loader's own scan options. A zero-row file contributes no row and is not listed (flagged).
- The sidecar refuses a source whose opened and hashed lists differ — the hash must cover what was loaded (successor (ii), for this path).
- The harness manifest shape is assumed (`files: [{path, sha256}]`, optional `ruxml_data_hash`) — program PR-024 not landed (Q19).

### Synthesis / Gate

**Outcome: Confirm.** ARCHITECTURE gains "The training-set data-hash bridge". Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `rux_ml.data.bridge`: `loader_file_set`, `training_set_sidecar`, `check_bridge`, `BridgeError(ValueError)`.
- `rux-ml data bridge --output … [--manifest …]` (source = `data.source_path`; exit 2 on a refusal or a difference; the sidecar records the check).
- `tests/integration/test_training_set_bridge_real.py` — the equality test on the real set, run by env var.

## Dependencies

PR-046 (stack; no code dependency on it).

## Verification criteria

- [x] The loader file set equals the hashed list on a hive `day=` subtree; the sidecar's `data_hash` equals `data_hashes(...)`; files + sha256 equal an independent hashlib listing; Polars version and git SHA recorded (`tests/data/test_bridge.py`).
- [x] The equality test holds on an equal manifest and refuses a missing file, an extra file, a sha256 difference and a `ruxml_data_hash` difference.
- [x] A symlinked subdirectory Polars loads and `rglob` misses is refused by name.
- [x] CLI: writes the sidecar, records an equal manifest check, exits 2 on a sha256 difference (`tests/cli/test_data_subcommands.py`).
- [x] Default suite 569 → 579 passed, 2 skipped (the real-set test skipped without env), 0 failed; ruff clean; basedpyright clean on the new files.
- [ ] The recorded real-set run — after program PR-024 writes `fill/` and `walk/` and their manifests.

## Notes — questions for the program lead

- Q19: the harness manifest's schema (key names, path base — relative to the subtree?), and whether the harness writes rux-ml's `data_hash` back into it (`ruxml_data_hash` assumed).
- Q20: hive keys — if the subtrees are hive-partitioned (`day=…`), the loader adds a `day` column that `data_hash` does not cover (PR-040's noted gap). Harmless while no feature names it; say if the set's layout is hive.
- Q21: a zero-row parquet file in a subtree is listed by the manifest but read by no row; the equality test would refuse it. Will the materializer ever write one (flat days are excluded by C6)?
