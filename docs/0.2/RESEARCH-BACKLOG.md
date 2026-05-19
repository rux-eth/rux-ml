# Research Backlog — v0.2

Index of v0.2 PRs with research status. Every PR runs `PROCEDURE-pr-research.md` before implementation — this document tracks the state of each PR's research and flags drift risk per the time-decay policy.

The v0.1 research backlog (PR-017 through PR-021) is frozen at [`docs/0.1/RESEARCH-BACKLOG.md`](../0.1/RESEARCH-BACKLOG.md) — all rows are `implementation-cleared`. v0.0 backlog at [`docs/0.0/RESEARCH-BACKLOG.md`](../0.0/RESEARCH-BACKLOG.md).

**Tiers:**
- **Tier 1** — Design-time research exists. Phase 1 (State Assessment) required before implementation; Phases 2-4 may be light if no drift found.
- **Tier 2** — Design-time research is partial or absent. Full 5-phase procedure required before the PR is written in final form.

**Status legend:**
- `design-research ✓` — research done at design time
- `design-research ~` — partial design research (pattern locked, per-instance details open)
- `design-research ✗` — no design-time research
- `state-assessed YYYY-MM-DD` — Phase 1 of `PROCEDURE-pr-research.md` completed
- `fully-researched YYYY-MM-DD` — all 5 phases of `PROCEDURE-pr-research.md` completed
- `implementation-cleared YYYY-MM-DD` — Phase 5 Gate Check passed

---

## Tier 1 — Implementation-Ready

**[PR-022](../../prs/PR-022-workbench-correctness-hygiene.md)** — workbench correctness hygiene. `fully-researched 2026-05-18` + `implementation-cleared 2026-05-18`. Ships under v0.1.1 PATCH (rolled into the next version cut). Research findings cited in the PR file: Q1 base.toml discriminator hygiene (project-specific, no Phase 3 web research) + Q2 eval_set placement convention (Phase 3 web research dispatched 2026-05-18; verdict: real practitioner split with cited findings — Position A is library-blessed default but with documented HPO bias compounding risk).

## Tier 2 — Research-Pending

| PR | Title | Design research | Required research topics |
|----|-------|-----------------|--------------------------|
| [PR-023](../../prs/PR-023-time-aware-cv-for-panel-data.md) | Time-aware CV for panel data | `design-research ~` → **`fully-researched 2026-05-19`** (Phase 5 implementation pending). Resolved: (1) Q1 — time-unit embargo API via polymorphic `embargo_time: int \| str` on `TimeSeriesSplitCV` (sktime + Nixtla + Darts convention). (2) Q2 — workbench `target_horizon_bars: int` + `embargo_pct: float` knobs convert to skfolio `purged_size` / `embargo_size = int(N * pct)` internally; AFML Snippet 7.3 verbatim formula; accept skfolio's symmetric/coarser purge as conservative-correct. (3) Q3 — new `PanelCombinatorialPurgedCV` discriminator variant; folds over unique sorted timestamps; per-asset purge in timestamp units; preserves combinatorial paths via row-index `(timestamp, asset)` decomposition (mlfinlab `StackedCombinatorialPurgedKFold` + Numerai era-CV convention). (4) Q4 — layered tests: per-asset hand-verified index fixtures + property tests (pairwise disjoint, per-asset embargo) + secondary shuffle-null tripwire; panel-fixture pattern designed in-house (best-guess; no production-grade precedent). (5) Q5 — `train_val_test_split` random-shuffle leak → PR-024 (research archived). (6) Q7 — walk-forward retrain helper → out of v0.2 entirely (3 of 4 surveyed delegate to user code). |
| [PR-024](../../prs/PR-024-temporal-train-val-test-split.md) | One-off temporal split | `design-research ~` — pattern locked by PR-023 Phase 3 Q5 archived research (≥3 cited systems: sktime + Darts + AutoGluon TS + Nixtla + mlfinlab all use **two separate functions**, not one knob). | (1) `time_col` validation semantics on stacked panels (sortedness contract; multi-series tail-slicing per sktime/AutoGluon). (2) Goldens for the temporal path. (3) Cross-validation warning when `cv.kind in {"time_series", "cpcv"}` but `data.split_kind == "random"`. (4) State assessment for any drift since PR-023 lands. |
| [PR-025](../../prs/PR-025-cv-eval-set-design-study.md) | Eval_set Position B/C operational design study | `design-research ~` — pattern surveyed by PR-022 Phase 3 archived research (Position A library-blessed; Position B sklearn HistGradientBoosting default; Position C XGBoost's own explicit recommendation). | (1) Quantify the workbench-specific HPO compounding bias by re-running the crypto-h3 baseline under Position A vs Position B vs Position C on the SAME splits. (2) Implementation cost of carving inner val from train fold per `data.split_ratios.val` (a) on all 3 trainer families' eval_set APIs (XGBoost / LightGBM Pattern-A shim / CatBoost). (3) Decision: switch to B / C / stay on A + document tradeoff more loudly. |
| [PR-027](../../prs/PR-027-int64-timestamp-time-unit.md) | Int64-timestamp time-unit fix | `fully-researched 2026-05-19` + `implementation-cleared 2026-05-19` — bug empirically reproduced pre-Phase 1; Phase 3 web research returned CONVENTION verdict for Option A (4-of-6 libraries enforce explicit unit declaration; Nixtla `validate_freq` + Darts RangeIndex `freq` strongest; Option C zero adopters). Phase 4 locked: `time_unit: Literal["ns","us","ms","s"] \| None = None` on `TimeSeriesSplitCV`; pl.Int64 columns require it (else `ValueError`); pl.Datetime columns reject it (else `ValueError`). One sub-decision (Datetime + time_unit set raises) labeled BEST-GUESS — stricter than Nixtla, user-acknowledged. | — (all required research complete; rolled into v0.2.0 at PR-026) |
| [PR-026](../../prs/PR-026-v0-2-0-cut.md) | v0.2.0 version cut | `fully-researched 2026-05-19` + `implementation-cleared 2026-05-19` — pattern inherited from PR-021 + Phase 1 evidence; no external research needed (CHANGELOG split + PR-025 numeric corrections + `docs/0.2/**` OPT_OUT_GLOBS + 8 historical-citation sentinels all locked from local state assessment). | — (all required research complete; tag v0.1.1 at 027c3b0 + v0.2.0 at this PR's merge commit) |

---

## Drift Watch

Per the time-decay policy in `PROCEDURE-pr-research.md`, any PR marked `fully-researched` or `state-assessed` more than the project's staleness threshold before implementation begins must re-run Phase 1 (State Assessment).

**Project staleness threshold:** **60 days** (recorded in [`docs/CONSTRAINTS.md`](../CONSTRAINTS.md); BEST-GUESS, user-acknowledged).

**Currently watching:** the v0.2 design session completed 2026-05-19. Each PR's `fully-researched` date must complete by **2026-07-18** for direct use, or be revalidated. Tier-2 PRs do NOT inherit the design-time staleness window — each PR's `state-assessed` date starts the clock from whenever its full 5-phase research completes.

### Per-PR drift-risk notes

- **PR-023** — skfolio `CombinatorialPurgedCV` source was inspected at 2026-05-19 (`_combinatorial.py` lines 81-466). Re-verify before implementation if skfolio version bumps; the version pinned at `cdbffd2` is the one read. pandas `Timedelta` parsing API (`pd.Timedelta("24h")`) was used as the polymorphic-`embargo_time` parser anchor — stable. AFML purge formula (Snippet 7.1, 7.3) is canonical text, unlikely to drift.
- **PR-024** — sktime + Darts + AutoGluon TS + Nixtla one-off-split conventions are stable (≥3 systems with multi-year track record). Phase 1 should re-check `src/rux_ml/data/splits.py` for any drift since PR-023 ships.
- **PR-025** — eval_set research (PR-022 Phase 3) is stable practitioner-split material. Phase 1 should re-check `src/rux_ml/tuning/objective.py` for any drift; if PR-023 modified `_fold_scores`, that's the surface PR-025 must work against.
- **PR-026** — `scripts/rewrite_doc_refs.py` is workbench-internal; will need a PATH_REWRITES update for `docs/0.1/* → docs/0.2/*`. Sentinel pattern from PR-021 carries forward.

### Untracked-but-watched ambiguities (from DESIGN-log.md A1–A5)

- **A1 (embargo magnitude defaults)** — resolved; users opt into AFML-recommended values per problem.
- **A2 (event-time / dollar bars)** — out of v0.2; revisit if event-bar ingest ever lands.
- **A3 (panel-fixture pattern in-house design)** — revisit if community CV-test conventions evolve (e.g., sktime ships hierarchical splitter tests).
- **A4 (eval_set Position B/C debate)** — owned by PR-025.
- **A5 (one-off temporal split)** — owned by PR-024.

---

## Design References

- [`docs/0.2/DESIGN-log.md`](DESIGN-log.md) — v0.2 design session (D1–D7 decisions + research trail + deferred A1–A5 ambiguities)
- [`docs/0.2/ROADMAP.md`](ROADMAP.md) — v0.2 PR plan
- [`docs/0.1/DESIGN-log.md`](../0.1/DESIGN-log.md) — v0.1 design history (Q1–Q6 + deferred A1–A6); decisions and constraints remain in force unless v0.2 DESIGN-log explicitly supersedes
- [`docs/0.0/DESIGN-log.md`](../0.0/DESIGN-log.md) — v0.0 design history (D1–D17 + PR-015 CV strategy + PR-016 docs-versioning migration)
- [`docs/VERSIONING.md`](../VERSIONING.md) — versioning policy + bump rules + changelog format (flat, meta-rule)
- [`docs/CONSTRAINTS.md`](../CONSTRAINTS.md) — hard rules (carried forward from v0.1; mandatory CONVENTIONS.md note on skfolio purge model added by PR-023)
