# configs/

TOML configuration for the workbench. Layered three-tier per `docs/ARCHITECTURE.md` D17:

```
configs/
├── base.toml                       # always loaded; project-wide defaults
├── problems/
│   ├── churn_v1.toml               # dataset paths, target column, eval metric
│   └── ...
└── studies/
    ├── churn_xgb_wide.toml         # search space, n_trials, sampler, pruner
    └── ...
```

## Loading order

`pydantic-settings` v2 with `TomlConfigSettingsSource(deep_merge=True)` overlays in this order, lowest → highest priority:

1. `base.toml`
2. `problems/<name>.toml`
3. `studies/<name>.toml`
4. `.env` (gitignored)
5. Env vars (`RUXML_*` with `__` for nesting — e.g. `RUXML_TRAINING__LEARNING_RATE`)
6. CLI dot-path overrides via repeatable `--set` (`--set training.learning_rate=0.05`; JSON-parsed values where possible)

## Override semantics

- **Nested tables** (e.g. `[training.xgboost]`) overlay key-by-key via `deep_merge=True`.
- **Lists** replace; they do not concatenate. Document any list-replacement explicitly when adding to `base.toml`.
- **Unknown keys** raise `pydantic.ValidationError` (`extra="forbid"` is set on every Pydantic model). Typos surface as errors, not silent.

## Hashing & elision

Per D17, `*_cfg_hash` (per-layer) and `root_cfg_hash` (full) are computed at trial time and recorded in Optuna `user_attrs`. Non-deterministic fields — paths, timestamps, and runtime-only values like `logs.path` and `studies.storage_url` — are elided before hashing. So is output-neutral guard config: `[data.oracle]` (PR-040) only decides whether ingest refuses, never what a passing run computes. The elision list lives in `src/rux_ml/config/root.py`.

## Oracle quarantine (`[data.oracle]`, PR-040)

`base.toml` sets `[data.oracle] namespace` and `tag_file`, mirrored from the rux-capital harness's `config/harness.toml` `[oracle]`. Every training-set read (`train`, `tune`, `registry promote` / `score`, `data hash` / `version`) refuses a source that has a column or nested field in that namespace (case-insensitive), or that has the tag file in any ancestor directory or inside it. Refused verbs exit 2. **Removing the table refuses all ingest** rather than switching the check off. See `docs/ARCHITECTURE.md` "Oracle quarantine at ingest".

## File status

Actual config schema (Pydantic models) and `base.toml` content land in **PR-002** (config layer skeleton). This README is the contract those files must honor.
