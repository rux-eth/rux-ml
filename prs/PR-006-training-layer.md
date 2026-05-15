# PR-006: Training layer

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

---

## Scope

Implement the training layer per D5 + D3 ingest decision rule. After this PR, `rux-ml train` runs a single baseline training end-to-end through data → features → trainer → score, with every reproducibility hash recorded.

- `src/rux_ml/training/protocol.py` — `Trainer(Protocol)` with `fit / predict / predict_proba / best_iteration_` per D5
- `src/rux_ml/training/factory.py` — `make_trainer(cfg: TrainingConfig) -> Trainer`:
  - `cfg.kind == "xgboost"` → `XGBClassifier` or `XGBRegressor` with `device=cfg.device` (`"cuda"` default), `enable_categorical=True`, `tree_method="hist"`, `**cfg.model_kwargs`
  - native-`xgb.train()` path (`cfg.use_native = True`) for QuantileDMatrix(ref=) and ExtMemQuantileDMatrix cases
- `src/rux_ml/training/metrics.py` — metric registry (`auc`, `logloss`, `rmse`, `mae`); metric chosen via `cfg.metric`
- `src/rux_ml/training/ingest.py` — implements the D3 decision rule:
  - estimate `X_bytes` from the materialized features
  - if ≲ `cfg.gpu_in_memory_x_gb_max` → `QuantileDMatrix(device="cuda", tree_method="hist")`
  - else → `ExtMemQuantileDMatrix` via `data.data_iter.build_iter(cfg)` with `cache_host_ratio` from `MemoryConfig`
- CLI body for `rux-ml train` (`src/rux_ml/cli/train.py`):
  - load `RuxMLConfig` (with `--problem` + `--study` resolved to TOML files)
  - run via `study.ask()` + `study.tell()` so it appears in `studies/studies.db` as a 1-trial study (per D7)
  - record per-trial `user_attrs` (config hashes, `data_hash`, `git_sha` — `entropy_hex` lands in PR-013)
  - print final score + storage location
- Tests:
  - Unit: `make_trainer(cfg)` returns the right concrete estimator with the right kwargs
  - Integration: `rux-ml train` on a tiny synthetic dataset succeeds, score within sane range
  - GPU-gated (`@pytest.mark.gpu`): same integration test against `device="cuda"`
  - Decision-rule unit test: `select_ingest(X_bytes, cfg)` returns `QuantileDMatrix` below threshold, `ExtMemQuantileDMatrix` above

NOT in scope: HPO sweeps (PR-007), subprocess isolation (PR-008), full provenance triple — entropy_hex lands in PR-013.

## Dependencies

PR-005.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Training" component row, "Decision Rules: XGBoost ingest path", "Key Abstractions: `Trainer` Protocol", "Data Flow" (one-off run path).

## Verification criteria

- [ ] `Trainer` Protocol type-checks against `XGBClassifier` (no errors)
- [ ] `make_trainer(cfg)` honors `device`, `enable_categorical`, `tree_method`, `**model_kwargs`
- [ ] Ingest decision rule selects correctly given a fake size estimator
- [ ] `rux-ml train --problem <p> --study <s>` runs end-to-end on a synthetic Parquet, records a trial, prints the score
- [ ] Recorded `user_attrs` includes (at minimum) the `*_cfg_hash` set + `data_hash` + `git_sha`
- [ ] GPU-gated integration test passes when CUDA is present and is correctly skipped on CPU-only

## Research backing

Tier 1:

- D5: [XGBoost sklearn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html), [XGBoost callbacks](https://xgboost.readthedocs.io/en/stable/python/callbacks.html)
- D3: [XGBoost ExtMem tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html), [QuantileDMatrix API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- D7: [Optuna `ask`/`tell`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)

State assessment must verify the XGBoost sklearn wrapper still exposes `device`, `enable_categorical`, `eval_set`, `callbacks`, `best_iteration_`.

## Notes

- The "5 % escape hatch" to `xgb.train()` is configurable per D5 — keep it as a clean code path, not a hidden branch deep in the factory.
- Do not start writing the registry yet (PR-010); just record the trial. Promotion is explicit and lives in its own PR.
