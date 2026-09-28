# PR-051: The model manifest records the harness training set's manifest id

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** R-14 track, delegated by the rux-capital program lead on 2026-09-27. Program v0.3 D45 #6 and ACCEPTANCE C11 require that "rux-ml's model manifest carries the harness manifest id and the harness manifest carries rux-ml's `data_hash`". Covered by the operator's standing approval of the R-14 track; the merge is operator-gated. **Stacked on PR-050**, because both change `registry/promote.py`.

## Research findings

### State Assessment (2026-09-27)

Baselines: rux-ml `pr-050/promote-data-hash-equality` @ `671f1f7` (on `dev` @ `15781d7`); the harness materializer, read-only, in the program's PR-024 worktree (rumpy-harness `ed19e0b`, `src/rumpy_harness/training/materialize.py` `write_manifest`).

- **What the harness writes.** `<set>/manifest.json` is the set's provenance manifest. It lists every file with its sha256, the stream manifests, the config / code SHAs, the window, the lockbox and the days, plus `"set": <root.name>` and `ruxml_sidecars: {fill|walk: {path, sha256, data_hash, rux_ml_git_sha, polars_version} | null}`. Beside it sits one view per subtree, `<set>/<t>.manifest.json`, with `{subtree, set_manifest: "manifest.json", set_manifest_sha256, label_config_sha256, feature_config_sha256, code_sha, files: [{path, sha256}]}`. Both are rewritten after every day. **There is no field named "id".** The id the harness itself uses to cite the set manifest is `set_manifest_sha256`, the content hash of `manifest.json`.
- **Where rux-ml loads the set.** `data.source_path` is the subtree (`configs/problems/m9_*.toml`: `…/training_set/fill`), so the view sits at `source_path.parent / f"{source_path.name}.manifest.json"`.
- `ModelManifest` (`registry/manifest.py`, `extra="forbid"`) has no field for it. The promote CLI probe on the real `m9_walk_bp` config and the C6 fixture promoted with exit 0 and recorded nothing.
- PR-047's `check_bridge` reads rux-ml's `data_hash` from the manifest's top-level `ruxml_data_hash` (its assumed shape). The harness writes it into the **set** manifest under `ruxml_sidecars.<t>.data_hash`, and not into the view. Checked against the real view, `check_bridge` therefore never compares the data_hash (Q23, not changed here).

**Decisions (routine):**
- The id is `manifest_id` = the sha256 of the set's `manifest.json`, which is the value the view cites. Promote verifies it against the file, so a set manifest rewritten after its view is refused rather than recorded under a stale id. `set_name`, `subtree`, the view's sha256 and the set manifest's copy of rux-ml's `data_hash` are recorded beside it.
- Refuse when the set manifest's rux-ml `data_hash` (if present) differs from the re-fit's. Otherwise the bundle would cite a manifest that describes other data. With PR-050's check this completes the chain trial → re-fit → harness manifest.
- **An `[m9]` problem refuses without a view** (fail closed: an M9 model trains on a harness set by definition, and C11 needs the id). A non-M9 source records `harness_manifest: null`, so every existing bundle and config is unchanged. The field is optional, so manifests written before this PR still read.
- The check runs before the re-fit, so a refusal costs no fit. The C6 fixture (`c6_set`) now writes the materializer's manifest and views, as the real set has them.
- The view's file name (`<subtree>.manifest.json`) is the harness's file contract, kept as a module constant in `rux_ml.data.bridge` next to PR-047's other harness-shape assumptions. It is not a tunable.

### Synthesis / Gate

**Outcome: Confirm.** ARCHITECTURE gains the id under the bridge section and in the promote flow. Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `rux_ml.registry.manifest.HarnessManifestRef`; `ModelManifest.harness_manifest: HarnessManifestRef | None = None`.
- `rux_ml.data.bridge`: `harness_view_path`, `harness_manifest_ref`, `HarnessManifestError(BridgeError)`.
- `rux_ml.registry.promote._harness_manifest` (before the re-fit), passed into the bundle manifest.
- Tests: `tests/registry/test_promote.py` (`test_promote_records_the_harness_manifest_id[False|True]`, `test_promote_refuses_a_harness_manifest_that_does_not_describe_the_set[stale_view|other_subtree|other_data_hash|no_set]`, `test_promote_records_no_harness_manifest_for_a_plain_source`), `tests/registry/test_manifest.py::test_harness_manifest_round_trips_and_is_optional`, `tests/cli/test_m9_promote_cli.py` (the real `m9_walk_bp` config: trained, refused without the view, promoted with it, id recorded). Fixture: `tests/conftest.py::write_harness_manifest`, used by `c6_set`.
- `docs/ARCHITECTURE.md`, `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`.

## Dependencies

PR-050 (stacked; merge it first).

## Verification criteria

- [x] A model promoted from a harness subtree records `manifest_id` = sha256(`manifest.json`), the set, the subtree and the view's sha256, plus the harness's copy of rux-ml's `data_hash` when present (equal to the model's). These tests failed first: `ImportError`, and the CLI test with `assert 0 == 2` / `promoted: m9_walk_bp@v_…`.
- [x] A stale view, a view of another subtree, a missing set manifest and a differing rux-ml `data_hash` are each refused, and no registry directory is written.
- [x] An `[m9]` problem without a view exits 2 through the real CLI. A plain source records `null`, and a pre-PR-051 manifest reads.
- [x] Default suite, ruff check, `ruff format --check` on the touched files, basedpyright `src/`: see the commit message.

## Notes — questions for the program lead

- **Q22:** confirm that "the harness manifest id" is the sha256 of the set's `manifest.json` (the harness has no explicit id field, and its views cite this hash). Because the set manifest is rewritten when the harness lists rux-ml's sidecar, the id a model records is the one current at promote. The harness side should therefore list the sidecar **before** the promote, so the recorded id is of the manifest that carries rux-ml's `data_hash`.
- **Q23 (PR-047):** `check_bridge` compares a top-level `ruxml_data_hash` that neither of the harness's files has; the harness keeps it at `ruxml_sidecars.<t>.data_hash` in the set manifest. The file lists compare as intended. Either align `check_bridge` with the real shape (a rux-ml PR) or have the view carry the field (the harness's change). Not changed here.
