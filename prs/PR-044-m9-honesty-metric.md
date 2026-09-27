# PR-044: M9 honesty metric, scoring rules and the fail-closed reader of the signed keys

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** Ruled by rux-capital program v0.3 D41 (the honesty criterion; R1-g proven — gen-2's criterion verbatim), D45 #3 (the four signed keys; the scoring rules, "convention, config not signed") and ACCEPTANCE C9 ("read through the fail-closed loader, SHA in every report"). R-14 track session, 2026-09-26; **stacked on PR-043**; operator-gated merge.

## Research findings

### State Assessment (2026-09-26)

Baseline: `pr-043/label-as-feature-refusal` @ `e7a61f3`; program @ `79eed8f` (read-only).

**Current state**:
- **The four keys are not in the program's `config/gates.yaml` today** (grep: none). C9 requires them "signed by the operator before the first Stage 3 study". The program's own reader (`scripts/gates/gates_registry.py`) verifies `ssh-keygen -Y verify -f allowed_signers -I operator@rumpy -n rumpy-gates` before parsing, refuses without override, and validates `applies_to ∈ {gate1, lockbox, gate2, gate3, risk}` — **an M9 key would need a new `applies_to` value there** (Q13).
- rux-ml had no metric for a signed-error test; `train` scores only `val` (which early stopping also uses); `test` is held out and scored only by `registry score` after promotion (PR-031 / PR-032).
- PyYAML is locked (6.0.3, via optuna) but was not a direct dependency.

**Stale assumptions / new constraints**:
1. The signer must come from rux-ml's config: a signer named inside the signed file could be swapped with the file.
2. Verify the bytes that are parsed (one read, passed to `ssh-keygen` on stdin) — no check-then-reread window.
3. Stage 3 is default-config (C9) — no hyperparameter selection — so scoring `test` inside the same `train` run selects nothing; Stage 4's pick is made on these results (Q14).

### Research (light — Tier-1)

- *Over-deduction statistic (undefined in the program docs).* Options: (a) net relative bias `(mean(pred) − mean(real)) / |mean(real)|` (chosen — matches gen-2's "+22–25 % conservative over-deduction", a single aggregate of the book); (b) median of per-row relative errors (robust, but undefined at zero realized cost per row, common for walk); (c) mean over over-deducted rows only (ignores net bias). **best-guess-given-constraints**, flagged Q12. The mechanism takes the bound from the signed key either way.
- *Signature verification: shell out to `ssh-keygen -Y verify` (chosen) vs a Python sshsig implementation.* `ssh-keygen` is what the program signs and verifies with (the same tool, identical semantics); a Python reimplementation would be a hand-rolled crypto path. **Proven** by the real-file test: the program's signed file verifies under rux-ml's reader.
- Group D: the real `gates.yaml` + `.sig` + `allowed_signers` read-only — signature verifies, the reader names the missing keys; the configs' identity / namespace equal the file's own `signature:` block (test).

### Synthesis / Gate

**Outcome: Confirm (with the flagged definitions).** ARCHITECTURE gains "M9 honesty metric and the signed keys". No CONSTRAINTS change (the zero-hardcoded rule already covers it). Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `M9GatesConfig` (`[m9.gates]`: path, signature_path, allowed_signers, identity, namespace — all required); `load_m9_gates` → `M9Gates(values, sha256, path)`; `M9GatesError(ValueError)`.
- `[m9] signed_error_honesty` (requires `[m9.gates]`, refused at load otherwise); true for markout / walk, false for fill (D45 #3).
- `signed_error_honesty(pred, realized, *, no_underdeduct_frac_min, overdeduct_max_rel)`.
- `train`: reads the keys before the trial row (exit 2 on doubt) whenever `[m9.gates]` is set; every `[m9]` fit records `oos` on `test` (metric on rows with a realized label, gates sha256, honesty verdict where enabled).
- The M9 configs point `[m9.gates]` at the sibling program checkout (Q11); `pyyaml` a direct dependency.

## Dependencies

PR-043 (stack).

## Architecture section implemented

`docs/ARCHITECTURE.md` "M9 honesty metric and the signed keys" (new).

## Verification criteria

- [x] Reader: returns the four signed values + sha256; refuses an edit after signing, a missing file (each of three), another identity, a missing key (named), a bool / string / null / out-of-domain value (named) — `tests/config/test_m9_gates.py`.
- [x] The real program file: signature verifies, keys absent → refused naming them (skipped when the program checkout is not beside rux-ml).
- [x] Honesty: hand-computed example (3 of 4 rows not under-deducted = 0.75; over-deduction (2.75 − 2.5)/2.5 = 0.1; MAE 0.75; one missing realized dropped and counted); each criterion fails alone; zero mean realized → not evaluable, never a pass; missing prediction / no rows refused — `tests/training/test_honesty.py`.
- [x] CLI on the C6-shaped set: walk records the verdict at the signed thresholds with the gates sha256 and MAE beside it; fill records Brier and no honesty test; an unsigned edit refuses with exit 2 and no trial (`tests/cli/test_m9_train_cli.py`; all three fail without the `train` wiring).
- [x] Default suite 524 → 549 passed, 1 skipped, 0 failed; ruff clean; basedpyright: no new error (the `tests.*conftest` import-resolution errors are the pre-existing pattern).
- [ ] The first real Stage 3 run — needs the operator's signature on the four keys and program PR-024's set.

## Research backing

Program D41, D45 #3, C9; `scripts/gates/gates_registry.py` (the program's reader, the verification pattern mirrored).

## Notes — questions for the program lead

- Q11: the program checkout's path relative to rux-ml on the desktop (configs assume `../rux-capital/program`).
- Q12: the over-deduction statistic — net relative bias (built) vs per-row median vs over-deducted rows only.
- Q13: the four keys' `applies_to` / `unit` / `label` / `review` fields in `gates.yaml` — the program's reader rejects an unknown `applies_to`, so signing them in needs a program-side reader change; rux-ml reads only `value`.
- Q14: the partition of the honesty verdict — built on the one-off `test` partition; Stage 4 picks on it (the truly unseen window is Stage 5's). Confirm, or name another.
- Q15: the fill fraction's honesty — D45 #3 gives it Brier only; C9 asks for "the honesty verdict per target per regime". Is there a fill-fraction honesty rule (e.g. predicted p_fill ≤ realized — over-predicting fills under-deducts cost under D45 #2), or is Brier-vs-floor the verdict?
- Q16: the sign convention of `y__markout_bp` — the honesty test assumes a cost (positive = adverse). And "against the floor's": the floor's predictions are the harness's (PR-026); rux-ml computes the same statistic for any prediction vector — should the set carry the floor's prediction as a column so rux-ml reports both?
