# PR-002: Config layer skeleton

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

---

## Scope

Lay down the configuration architecture from D2 + D17 — schema in code, layered TOML on disk, override precedence wired up.

- `src/rux_ml/config/__init__.py` re-exports `RuxMLConfig` + each per-layer config via `__all__`
- `src/rux_ml/config/root.py` with `RuxMLConfig(BaseSettings)`, `settings_customise_sources` returning `(cli, init, env, dotenv, toml, file_secrets)` ordered, env prefix `RUXML_`, `extra="forbid"`, hash-elision constant `_HASH_ELIDED_FIELDS`
- `src/rux_ml/config/{data,features,training,tuning,runs,registry,memory}.py` each with a Pydantic model containing only the fields required by later PRs (placeholder defaults marked with TOML knob comments)
- `src/rux_ml/config/tuning.py` includes `SearchSpec` tagged-union (`FloatSpec` / `IntSpec` / `CatSpec`) per D16
- TOML loader entry point: `RuxMLConfig.from_layers(base, problem=None, study=None, **cli_overrides)` resolving the file list passed to `TomlConfigSettingsSource(deep_merge=True)`
- `configs/base.toml` with a minimal complete config (every layer has the fields needed by the smoke tests below)
- `configs/README.md` documenting the three-tier composition + override precedence
- Hashing helpers in `src/rux_ml/_internal/hashing.py` (uses `xxhash` for bytes; canonical JSON via `json.dumps(sort_keys=True)`); helpers `cfg_hash(cfg, exclude=...)` and `layer_cfg_hash(cfg, layer)` returning hex strings
- Tests: load `configs/base.toml`, validate, assert overrides via env + dict CLI win in priority order; assert `extra="forbid"` raises on unknown key; assert per-layer + root hashes are stable across reloads of identical TOML

NOT in scope: data/features/training implementations (later PRs), CLI subcommands (PR-003), search-space walker (PR-007).

## Dependencies

PR-001.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Configuration Architecture", "Key Abstractions: `RuxMLConfig` and `SearchSpec`", and the `_HASH_ELIDED_FIELDS` discussion.

## Verification criteria

- [ ] `RuxMLConfig.from_layers("configs/base.toml")` returns a fully-validated config
- [ ] Layered loading: `from_layers(base, problem="dummy", study="dummy")` overlays in order
- [ ] CLI overrides beat env vars; env vars beat study TOML; study beats problem; problem beats base
- [ ] Unknown TOML keys raise `pydantic.ValidationError` (`extra="forbid"`)
- [ ] `RUXML_TRAINING__LEARNING_RATE=0.01` env override works
- [ ] `cfg_hash(cfg)` is deterministic across reloads; differs when a non-elided field changes; identical when only an elided field (path, timestamp) changes
- [ ] `layer_cfg_hash(cfg, "training")` differs only when the `training` layer changes
- [ ] `SearchSpec` discriminated union round-trips through TOML

## Research backing

Tier 1:

- D2: [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/), conflict re Hydra stagnation
- D16: [Pydantic frozen `model_copy`](https://github.com/pydantic/pydantic/discussions/4250), [Hydra Optuna Sweeper precedent](https://hydra.cc/docs/plugins/optuna_sweeper/)
- D17: [pydantic-settings configuration files](https://deepwiki.com/pydantic/pydantic-settings/3.2-configuration-files), [DVC repro](https://dvc.org/doc/command-reference/repro), [MLflow datasets](https://mlflow.org/docs/latest/python_api/mlflow.data.html)

State assessment must verify the `TomlConfigSettingsSource(deep_merge=True)` API is still present and the source-priority hook signature has not changed.

## Notes

- Per D17, lists in TOML *replace* on overlay; document in `configs/README.md` so users don't expect concatenation.
- `data_cfg_hash` in this PR is the **config layer** hash; `data_hash` (dataset content) lands in PR-004. Naming is intentional per `docs/ARCHITECTURE.md` "Cross-Decision Consistency Notes".
