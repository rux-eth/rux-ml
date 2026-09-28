# PR-050: Promote refuses a re-fit whose `data_hash` differs from the trial's

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** PR-040's named successor (i), delegated by the rux-capital program lead to the R-14 track on 2026-09-27. Program v0.3 D45 #6 / ACCEPTANCE C11: "the promote re-fit's `data_hash` equals the Stage 4 trial's recorded one". Covered by the operator's standing approval of the R-14 track; merge operator-gated.

## Research findings

### State Assessment (2026-09-27)

Baseline: rux-ml `dev` @ `15781d7` (PR-049 merged, #55).

- PR-040's gap is still open. `registry/promote.py` re-reads `cfg.data.source_path` for the re-fit and recomputes `data_hashes(...)` into the bundle manifest (`_build_manifest`), but never compares it with the trial's `user_attrs["data_hash"]` (`TrialAttrs.data_hash`, required, so every promotable trial has one).
- `data.source_path` is hash-elided (D17), so it is not part of `root_cfg_hash`, and `--set data.source_path=…` or a file rewritten in place re-fits on other data.
- **Probe (the new CLI test, run against `dev`'s code):** `registry promote --set data.source_path=<other.parquet>` on a trial recorded on `synth.parquet` printed `promoted: churn_v1@v_2026_09_28_0b0efc` and exited 0.
- The hashes were computed after the re-fit, a second full read of the source (the re-fit's `load_parquet` is the first).
- Every existing promote caller (`tests/registry`, `tests/cli`, `tests/golden`) records the trial's hashes from the same source it promotes, so the check leaves them unchanged.

**Decisions (routine):**
- Compare the composite `data_hash` (`bytes|logical`), the value the trial records. Hash once, **before** the re-fit, so a refusal costs one hash pass and no fit. The bundle carries the same hashes.
- Refuse with `PromoteDataHashError(ValueError)`: the CLI's promote handler maps `ValueError` to exit 2, like every other promote refusal. The message names both hashes, the source and `study#trial`.
- No override flag. PR-040 said "if one is ever needed", and nothing needs one.
- Oracle-derived data is refused by the hash's own quarantine check (PR-040), which now fires before the re-fit rather than inside it. Same error family, same exit code.

### Synthesis / Gate

**Outcome: Confirm.** ARCHITECTURE's promote flow gains the check. Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `rux_ml.registry.promote`: `PromoteDataHashError`; `_refit_data_hashes` (hash + compare, before the re-fit); `_build_manifest` takes the checked hashes.
- Tests: `tests/registry/test_promote.py::test_promote_refuses_a_refit_whose_data_hash_differs_from_the_trials` (the file rewritten in place, and another file), `::test_promote_manifest_data_hash_equals_the_trials`; `tests/cli/test_registry_subcommands.py::test_registry_promote_refuses_a_refit_on_other_data_exit_2`.
- `docs/ARCHITECTURE.md` (promote flow), `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`, PR-040's successor list.

## Dependencies

`dev` @ `15781d7`.

## Verification criteria

- [x] A trial re-fit on other data, whether the file was rewritten in place or `data.source_path` points elsewhere, is refused with `PromoteDataHashError` naming both hashes and the trial, and no registry directory is written. These tests failed first: the module test with an `ImportError`, and the CLI test with `assert 0 == 2` / `promoted: churn_v1@v_…`.
- [x] The equal case promotes as before, and the bundle's `data_hash` equals the trial's `user_attrs["data_hash"]`.
- [x] The oracle and label refusals on the promote path still exit 2 (`tests/cli/test_oracle_quarantine_cli.py`, `tests/cli/test_label_quarantine_cli.py`).
- [x] Default suite, ruff check, `ruff format --check` on the touched files, basedpyright `src/`: see the commit message.

## Notes — for the program lead

- This lands the promote half of C11's "the promote re-fit's `data_hash` equals the Stage 4 trial's". The evidence on the real set, a promoted M9 model, belongs to program PR-028.
