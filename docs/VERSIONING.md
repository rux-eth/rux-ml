# Versioning

How `rux-ml` is versioned, how versions are documented, and what triggers a version change.

This doc is **flat (not version-scoped)** — it describes how versioning itself works, not a snapshot of any one version. Edits supersede in place.

---

## 1. Bump-rule triggers

`rux-ml` uses **ZeroVer pre-1.0** with project-specific rules for what counts as a user-visible change. The user is a solo researcher; the user-facing surface is the CLI (`rux-ml`), the TOML config schemas (`base.toml` + `problems/*.toml` + `studies/*.toml`), the on-disk model-registry bundle layout (`pipeline.skops` + `model.ubj` + `manifest.json`), and the per-trial Optuna study schema (`user_attrs` provenance fields). The bump boundary is **human-decided, never tool-promoted** — every minor bump triggers a design session (§2), so tools that auto-promote patches to minors based on commit messages are forbidden.

| Bump | When | Triggers design session? |
|------|------|--------------------------|
| **MAJOR** (`x → x+1`) | CLI flag/command rename or removal; breaking change to a model-registry bundle layout; breaking change to a TOML config schema (a renamed/removed key, a changed type, a new required field); breaking change to a manifest-JSON field that invalidates existing bundles; breaking change to the Optuna `user_attrs` provenance triple (a missing/renamed field that invalidates existing studies). | Yes — full `PROCEDURE-design-planning.md` session + migration plan. |
| **MINOR** (`0.x → 0.x+1`) | New user-visible CLI verb or flag; new layer factory (`make_*`); new built-in `Trainer`, `Solver`, or `Splitter` family; new study/problem-level config knob; new `manifest.json` field; new `user_attrs` field; **removal of a registered family after a one-release `FutureWarning` window** (per PR-017, anchored on sklearn / NumPy NEP 23 / PyTorch pre-1.0 deprecation conventions). | **Yes** — see §2. |
| **PATCH** (`0.x.y → 0.x.y+1`) | Bug fixes, copy/typo edits, internal refactors with no behavioural change, dependency bumps that don't change output, doc-only edits. | No. |

**Family-removal policy** (per PR-017 / `docs/0.1/DESIGN-log.md` Q6): <!-- rewrite-doc-refs:skip-line --> deprecating a Trainer or Solver family ships a `FutureWarning` from the family's factory in release `0.y`; the family is removed in release `0.(y+1)`. The bundle-loadability break (any `manifest.json` whose `family` matches the removed name is no longer reconstructable via `load_model`) is accepted as a documented side effect of the deprecation period — recorded in `CHANGELOG.md`, **not** elevated to MAJOR. Anchored on sklearn (`RandomizedLasso`/`RandomizedLogisticRegression`: deprecated 0.19 → removed 0.21), NumPy NEP 23 ("≥2 releases / ≥1 year", removal in MINOR), and PyTorch core (stable tier: 2 releases + 180 nightlies, removal in MINOR). pandas' MAJOR-only removal rule was considered and rejected for a single-user workbench where the `FutureWarning` reaches the only user reliably.

**v1.0 criteria:** the workbench is feature-stable enough that registry bundles + Optuna studies could be handed to a future collaborator (or future-me, post-context-loss) and reproduced end-to-end without an undocumented oral tradition. Practically: a first real-dataset run has stress-tested the BEST-GUESS defaults (categorical_low_card_threshold, HPO budgets, trial_timeout_s), the per-problem CV strategy mapping is settled, and a Rust crate has materialized for any profiled Python hot path > 5%.

## 2. Design-planning at every minor/major cut

Every minor or major bump is treated as **starting a fresh project**. Before any implementation PR for the new version lands, run `PROCEDURE-design-planning.md` from Phase 1 — Idea / Decisions / Convergence / Docs — producing a fresh set of PR stubs for the new version's work.

Patch bumps do not require a design session.

Workflow for a minor cut (example: 0.0.x → 0.1.0):

1. Surfacing — a user-visible feature is proposed or a meaningful behavioural change is needed.
2. **Design session** — run `PROCEDURE-design-planning.md` Phases 1-3.
3. **Phase 4 (Docs)** — snapshot the *current* (pre-cut) state into `docs/<current-x.y>/` per §3, then write fresh `docs/<new-x.y>/{DESIGN-log, ROADMAP, RESEARCH-BACKLOG}.md` for the new version's work. Create PR stubs under `prs/` with monotonic numbers per §4. Update root files (`ARCHITECTURE.md`, `CONSTRAINTS.md`, `CONVENTIONS.md`, etc.) in place — they are SSOT, not snapshots.
4. **Migrate cross-references** — re-point `docs/X.md` references to the new `docs/<x.y>/X.md` paths. Use `scripts/rewrite_doc_refs.py` — see §5.
5. **Implement PRs** — each runs `PROCEDURE-pr-research.md` before its implementation.
6. **Cut the version** — bump `pyproject.toml`'s `version` and `src/rux_ml/__init__.py`'s `__version__`, add `## [x.y.z] - YYYY-MM-DD` to `/CHANGELOG.md`, push the merge commit, and tag at that commit (§7).

## 3. Doc-versioning layout (hybrid)

Docs split into **SSOT** (single-source-of-truth — describe the current system; rewritten in place) and **temporal** (append-log or plan-of-record — frozen at each version cut).

| File | Placement | Why |
|------|-----------|-----|
| `docs/ARCHITECTURE.md` | flat | SSOT — describes current system; historical states recoverable from `git log` |
| `docs/CONSTRAINTS.md` | flat | SSOT — non-negotiables apply to current code |
| `docs/CONVENTIONS.md` | flat | SSOT — applies to current codebase |
| `docs/DEPLOYMENT.md` | flat | SSOT operational how-to (container build + image-digest capture) |
| `docs/VERSIONING.md` | flat | This doc — meta-rule, not a snapshot |
| `docs/<x.y>/DESIGN-log.md` | versioned | Temporal append-log; per-version decisions are the canonical record of that era |
| `docs/<x.y>/RESEARCH-BACKLOG.md` | versioned | Tied to in-flight PR decisions for that version |
| `docs/<x.y>/ROADMAP.md` | versioned | Plan-of-record for that version |

**Default for a new doc added mid-version:** flat. Move to version-scoped only if the doc is explicitly temporal (a log, a roadmap, a research backlog).

## 4. PR-file numbering

Flat monotonic across all versions. `prs/PR-NNN.md` files keep their numbers forever; numbers are never reused.

- A PR scoped pre-cut but landing post-cut keeps its number. Use the `Landed-in: vX.Y.Z` header field for version attribution.
- A PR split mid-implementation across a version cut: the original is marked closed/superseded; new work gets fresh monotonic numbers.
- The `Landed-in:` header is required on every numbered PR file. Use `(not yet landed)` for in-flight or dormant PRs, or `superseded by PR-XXX` for replaced ones.

## 5. Cross-reference migration at a version cut

When temporal docs move into a new `docs/<x.y>/` dir, all references to the old paths across the repo must be updated. Run `scripts/rewrite_doc_refs.py` (Python port of the canonical TS migrator); the script holds the mapping table and applies it idempotently across `*.md` and `*.py` files.

Run order:

```
git mv docs/<file>.md docs/<x.y>/<file>.md         # per file moved
uv run python scripts/rewrite_doc_refs.py --dry-run   # review the diff
uv run python scripts/rewrite_doc_refs.py             # apply
git add -A && git commit
```

Equivalently: `make rewrite-doc-refs-dry` and `make rewrite-doc-refs`.

Update the `PATH_REWRITES` table in the script at every subsequent cut. The negative-lookbehind regex on `MOVED_FILE_FIXUPS` keeps the script idempotent — `../prs/` rewrites to `../../prs/` exactly once even if run repeatedly. Self-rewriting is suppressed via `__file__` comparison so the script's own `PATH_REWRITES` literals aren't corrupted on the walk.

Split the migration into a rename-only commit + a content-fixup commit if any moved file's internal links drop below git's 50% rename-similarity threshold (`git log --follow` won't trace renames otherwise). Measure with `git diff --cached -M1 --summary`. At the v0.0 → docs/0.0/ cut (PR-016), `ROADMAP.md` landed at 33% and `RESEARCH-BACKLOG.md` at 49% — both below the default threshold — so the split was mandatory.

No automated link checker watches between cuts. Doc-link rot is detected at the next cut by the script's unmapped-path surface.

## 6. Changelog

`/CHANGELOG.md` at the repo root, hand-written, [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

Sections per release (in order, omit empty sections):

- `Added` — new features
- `Changed` — changes in existing functionality
- `Deprecated` — soon-to-be-removed features
- `Removed` — removed features
- `Fixed` — bug fixes
- `Security` — vulnerability-related changes

Entries are user-facing. Each entry ends with a PR identifier in parentheses (the `prs/PR-NNN-*.md` filename number, not the GitHub PR number — the former is the durable ID, the latter is a side effect of forge state). The `[Unreleased]` section accumulates entries as PRs merge; at a version cut, it's renamed to `[x.y.z] - YYYY-MM-DD` and a fresh `[Unreleased]` opens.

`release-please` and similar tools that auto-promote minors on `feat:` commits are **forbidden** because they bypass the design-session-per-cut rule.

### Entry template

```markdown
## [0.1.0] - YYYY-MM-DD

### Added
- [User-facing feature description] (PR-NN).

### Changed
- [Existing-behavior change description] (PR-NN).

### Fixed
- [Bug-fix description] (PR-NN).

[0.1.0]: https://github.com/rux-eth/rux-ml/compare/v0.0.1...v0.1.0
```

## 7. Tag and deploy alignment

Version source-of-truth files (`pyproject.toml` `version`, `src/rux_ml/__init__.py` `__version__`, the git tag, and `/CHANGELOG.md`) must always be in sync.

**Tag points at the merge commit that ships the version** — never at an arbitrary earlier commit. If `main` advances past the tag with no version bump, that's fine — the tag still points at the merge commit that shipped the version. Re-tagging is only justified if a version bump is missed; prefer a corrective patch PR over re-tagging.

At every cut:

1. Merge the version-cut PR (containing the version-bump + changelog + relevant doc updates).
2. `git tag v<x.y.z>` at the resulting `main` HEAD.
3. `git push --tags`.
