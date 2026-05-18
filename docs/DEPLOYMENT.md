# Deployment

How to "deploy" `rux-ml` and verify the end-to-end flow.

`rux-ml` is a single-user workbench with **no remote deploy target** — no web UI, no server, no published Python package, no model-serving endpoint (see `docs/CONSTRAINTS.md` → No Web UI / No Server). What this doc covers is the only operational concern that *is* deploy-shaped: building the container image, capturing its digest, and pinning that digest into the per-trial provenance triple.

The image digest is recorded per-trial in `user_attrs.image_digest`; the registry's promotion gate rejects trials missing it. Rebuilding the image therefore changes the provenance surface — handle rebuilds deliberately.

---

## Prerequisites

- Docker Engine CE installed on the workbench host (verified on the desktop per memory `reference_workbench_access`).
- NVIDIA Container Toolkit installed and configured (`nvidia-ctk runtime configure --runtime=docker`).
- An RTX 4090 (or any CUDA 12.x-compatible GPU) visible to Docker via `nvidia-smi` inside `nvidia/cuda:12.4.1-base-ubuntu22.04`.
- `make` and `git` available on the host.

## Step 1 — Build the image

```sh
make docker-build
```

This invokes `DOCKER_BUILDKIT=1 docker build -t rux-ml:local .` against the project root's `Dockerfile`, then writes the resulting image ID to `.docker-image-digest`. The Dockerfile pins its base by `@sha256:` per `docs/CONSTRAINTS.md` → Container Digest Pinning; never edit the Dockerfile to use a `:tag` reference.

## Step 2 — Capture the digest

`make docker-build` already writes `.docker-image-digest` to the repo root. The file is gitignored and recreated on every build. Subsequent `rux-ml train` / `rux-ml tune` invocations read this file to populate `user_attrs.image_digest`; a missing or stale `.docker-image-digest` is a hard error at trial start.

If you `docker image rm rux-ml:local` between sessions, re-run `make docker-build` before any training/tuning work — the registry's promotion check rejects trials with a digest that doesn't match a currently-available image.

## Step 3 — Verify the GPU passthrough

```sh
docker compose run --rm rux-ml --version
make docker-shell    # open a bash shell; run nvidia-smi to confirm GPU visibility
```

`docker-compose.yml` declares the GPU reservation block + `mem_limit: 32g` (per PR-012). If `nvidia-smi` inside the container reports no devices, the toolkit is misconfigured.

## Smoke test

After a fresh build, run:

```sh
make test                # default markers (excludes gpu/slow/golden/docker)
RUXML_RUN_DOCKER_TESTS=1 uv run pytest -m docker   # gated docker smokes
uv run pytest -m gpu     # gated GPU determinism smoke (requires CUDA visible)
uv run pytest -m golden  # tolerance-based golden regression
```

All four sets must pass at the current image digest before any training/tuning that will eventually promote to the registry. If the golden test fails on a fresh build, the image diverged from the locked CPU bit-exact contract (see PR-013, PR-014) — diagnose before promoting any further work.

## When to rebuild

Rebuild the image when **any** of these changes:

- The Dockerfile or `docker-compose.yml` (image-defining configuration).
- `pyproject.toml` / `uv.lock` (Python dependency set).
- The CUDA base-image digest (NVIDIA periodically deletes EoL tags; check the watch note in `docs/0.1/RESEARCH-BACKLOG.md` Drift Watch).
- Driver upgrade on the host (mismatched user-mode driver vs container CUDA runtime).

After every rebuild, re-run the smoke test before resuming any tracked study.

## Rollback

There is no remote roll-back. The local roll-back is `git checkout <old-tag> && make docker-build` against the older Dockerfile state. Image-digest-pinned trials remain reproducible from the registry bundle regardless of what `rux-ml:local` currently points at — registry bundles record their build-time `image_digest` and can be inspected with `rux-ml registry list --bundle <id>`.

## Notes

- This doc is **flat (SSOT)** — describes how to deploy *the current version*. Historical deploy procedures are recoverable from `git log`.
- Cross-link in `CLAUDE.md` Design References (done in PR-016).
- "Deployment" for a published Python package or a server-mode workbench is explicitly out of scope per the No Web UI / No Server constraint. A v1.0 or later cut that introduces either would need this doc rewritten in place per `docs/VERSIONING.md` §3.
