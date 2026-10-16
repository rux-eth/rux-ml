"""PR-063 Q1: what each thread pool holds before any fit, per import order (Linux only).

Harness for the PR-063 Q1 measurement; the receipt goes to ``prs/PR-063-q1-threads-<date>.json``
(the PR-025 convention: harness under ``scripts/``, receipt beside the PR file).

Three arms, each a fresh interpreter spawned with the four pinned env vars (and
``GOTO_NUM_THREADS``) stripped from the environment:

- ``cli_today``: ``import rux_ml.cli`` then ``pin_threads`` (the in-process verbs' order:
  ``cli/__init__.py:26`` loads every verb, the pin runs after the cfg build).
- ``trial_child``: the trial child's order (``_internal/trial_runner.py:86-111``):
  ``from rux_ml.config import RuxMLConfig`` -> ``pin_threads`` -> the post-pin imports.
- ``env_at_start``: the ``trial_child`` order with the four values in the child's env
  at start (the Q2 spawn-env condition at ``tuning/isolation.py:123``).

Each child reads, before any fit: ``/proc/self/task`` (count, ``comm``, CPU ticks), then
``threadpoolctl.threadpool_info()``, an idle window, ``polars.thread_pool_size()`` (which
builds Polars' pool), then ``import lightgbm`` and ``threadpool_info()`` again. An audit
hook records the thread count at every ``import`` event, so each rise is attributed to
the import in flight when it happened.

Measurement only: threadpoolctl reads the pools and never sets them (D10 rejects it as
the pinning mechanism, not as an instrument). The pinned values come from the base
layer's ``[memory]`` table through ``pin_threads`` itself, so no value is restated here.
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import tomllib
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PINNED = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS")
STRIPPED = (*PINNED, "GOTO_NUM_THREADS")
ARMS = ("cli_today", "trial_child", "env_at_start")
POOL_LIB_RE = re.compile(r"(gomp|openblas|libomp|iomp|mkl)", re.IGNORECASE)
MAPS_FIELDS_WITH_PATH = 6  # /proc/<pid>/maps: address perms offset dev inode pathname
VERSION_PKGS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "xgboost",
    "polars",
    "pyarrow",
    "pandas",
    "lightgbm",
    "catboost",
    "threadpoolctl",
)


def _memory_table(config: Path) -> dict[str, Any]:
    with config.open("rb") as fh:
        return dict(tomllib.load(fh).get("memory", {}))


def _threads() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tid in os.listdir("/proc/self/task"):
        base = Path("/proc/self/task", tid)
        try:
            comm = (base / "comm").read_text().strip()
            stat = (base / "stat").read_text()
        except FileNotFoundError:
            continue
        fields = stat[stat.rindex(")") + 2 :].split()
        # stat fields 14 (utime) and 15 (stime); the slice starts at field 3.
        out.append({"tid": int(tid), "comm": comm, "cpu_ticks": int(fields[11]) + int(fields[12])})
    return out


def _snap_summary(snap: list[dict[str, Any]]) -> dict[str, Any]:
    tick_ms = 1000.0 / os.sysconf("SC_CLK_TCK")
    main = os.getpid()
    return {
        "count": len(snap),
        "comm": dict(Counter(t["comm"] for t in snap)),
        "cpu_ms_non_main": sum(t["cpu_ticks"] for t in snap if t["tid"] != main) * tick_ms,
    }


def _pool_libs() -> list[str]:
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=MAPS_FIELDS_WITH_PATH - 1)
        if len(parts) == MAPS_FIELDS_WITH_PATH and POOL_LIB_RE.search(Path(parts[5]).name):
            paths.add(parts[5])
    return sorted(paths)


def _pools(info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = ("user_api", "internal_api", "filepath", "num_threads", "threading_layer", "version")
    return [{k: p.get(k) for k in keep} for p in info]


def _child(arm: str, config: Path, idle_s: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    rises: list[dict[str, Any]] = []
    state: dict[str, Any] = {"threads": len(os.listdir("/proc/self/task")), "last": "<start>"}

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "import":
            return
        n = len(os.listdir("/proc/self/task"))
        if n != state["threads"]:
            rises.append({"during_import_of": state["last"], "from": state["threads"], "to": n})
            state["threads"] = n
        state["last"] = args[0]

    sys.addaudithook(hook)
    env_at_child_start = {k: os.environ.get(k) for k in PINNED}

    # The arm's own import order, exactly as the code path runs it.
    import rux_ml._internal.trial_runner  # noqa: PLC0415

    if arm == "cli_today":
        import rux_ml.cli  # noqa: F401, PLC0415
    else:
        from rux_ml.config import RuxMLConfig  # noqa: F401, PLC0415
    from rux_ml._internal.env import pin_threads  # noqa: PLC0415
    from rux_ml.config.memory import MemoryConfig  # noqa: PLC0415

    pin_threads(MemoryConfig.model_validate(_memory_table(config)))
    if arm != "cli_today":
        from rux_ml._internal.seeds import make_seed_bag  # noqa: F401, PLC0415
        from rux_ml.training import optuna_direction  # noqa: F401, PLC0415
        from rux_ml.tuning import (  # noqa: F401, PLC0415
            build_objective,
            create_or_load,
            make_pruner,
            make_sampler,
        )
    import_wall_s = time.perf_counter() - t0

    snap_a = _threads()  # before any pool read, before any fit
    from threadpoolctl import threadpool_info  # noqa: PLC0415

    pools_a = _pools(threadpool_info())
    libs_a = _pool_libs()
    time.sleep(idle_s)
    snap_idle = _threads()

    import polars as pl  # noqa: PLC0415

    polars_pool = pl.thread_pool_size()
    snap_b = _threads()

    import lightgbm  # noqa: F401, PLC0415

    pools_c = _pools(threadpool_info())
    libs_c = _pool_libs()
    snap_c = _threads()

    openblas = [p for p in pools_a if p["internal_api"] == "openblas"]
    openblas_workers = sum(max(int(p["num_threads"]) - 1, 0) for p in openblas)
    a = _snap_summary(snap_a)
    idle = _snap_summary(snap_idle)
    status = Path("/proc/self/status").read_text()
    rss = {k: v.strip() for k, _, v in (ln.partition(":") for ln in status.splitlines())}
    return {
        "arm": arm,
        "env_at_child_start": env_at_child_start,
        "env_after_pin": {k: os.environ.get(k) for k in PINNED},
        "import_wall_s": round(import_wall_s, 3),
        "threads_pre_fit": a,
        "threads_after_idle": idle,
        "idle_s": idle_s,
        "idle_cpu_ms_non_main": round(idle["cpu_ms_non_main"] - a["cpu_ms_non_main"], 1),
        "pools_pre_fit": pools_a,
        "pool_libs_mapped_pre_fit": libs_a,
        "attribution_pre_fit": {
            "main": 1,
            "openblas_workers": openblas_workers,
            "residual": a["count"] - 1 - openblas_workers,
        },
        "import_rises": rises,
        "polars_thread_pool_size": polars_pool,
        "threads_after_polars_init": _snap_summary(snap_b),
        "pools_after_lightgbm": pools_c,
        "pool_libs_mapped_after_lightgbm": libs_c,
        "threads_after_lightgbm": _snap_summary(snap_c),
        "vm_rss": rss.get("VmRSS"),
        "vm_size": rss.get("VmSize"),
    }


def _pool_str(pools: list[dict[str, Any]], api: str) -> str:
    sel = [p for p in pools if p["internal_api"] == api]
    return ", ".join(f"{Path(p['filepath']).name}={p['num_threads']}" for p in sel) or "-"


def _parent(args: argparse.Namespace) -> int:
    if not sys.platform.startswith("linux"):
        sys.stderr.write("Linux only: libgomp / OpenBLAS size at load there; macOS differs.\n")
        return 2
    base_env = {k: v for k, v in os.environ.items() if k not in STRIPPED}
    from rux_ml._internal.env import pin_threads  # noqa: PLC0415
    from rux_ml.config.memory import MemoryConfig  # noqa: PLC0415

    memory = MemoryConfig.model_validate(_memory_table(args.config))
    pin_threads(memory)  # the values the trial child would pin, by the code's own mapping
    pinned = {k: os.environ[k] for k in PINNED}
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    results: list[dict[str, Any]] = []
    for run in range(args.runs):
        for arm in args.arms:
            env = dict(base_env)
            if arm == "env_at_start":
                env.update(pinned)
            argv = [sys.executable, str(Path(__file__).resolve()), "--child", arm]
            argv += ["--config", str(args.config), "--idle-s", str(args.idle_s)]
            proc = subprocess.run(
                argv,
                env=env,
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=args.timeout_s,
                check=False,
            )
            row: dict[str, Any] = {"arm": arm, "run": run, "rc": proc.returncode}
            if proc.returncode == 0:
                row["data"] = json.loads(proc.stdout.strip().splitlines()[-1])
            else:
                row["stderr_tail"] = proc.stderr[-2000:]
            results.append(row)
            d = row.get("data")
            line = f"{arm:13s} run {run} rc {proc.returncode}"
            if d:
                line += (
                    f" | pre-fit {d['threads_pre_fit']['count']}"
                    f" | gomp {_pool_str(d['pools_pre_fit'], 'openmp')}"
                    f" | openblas {_pool_str(d['pools_pre_fit'], 'openblas')}"
                    f" | residual {d['attribution_pre_fit']['residual']}"
                    f" | polars {d['polars_thread_pool_size']}"
                    f" -> {d['threads_after_polars_init']['count']}"
                    f" | +lgbm gomp {_pool_str(d['pools_after_lightgbm'], 'openmp')}"
                    f" -> {d['threads_after_lightgbm']['count']}"
                )
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    ok = [r for r in results if r["rc"] == 0]
    medians = {
        arm: statistics.median(r["data"]["threads_pre_fit"]["count"] for r in ok if r["arm"] == arm)
        for arm in args.arms
        if any(r["arm"] == arm for r in ok)
    }
    versions = {}
    for pkg in VERSION_PKGS:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    receipt = {
        "pr": "PR-063",
        "question": "Q1",
        "git_sha": sha,
        "uname": " ".join(platform.uname()),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)),
        "versions": versions,
        "config": str(args.config),
        "memory": memory.model_dump(),
        "pinned_env": pinned,
        "stripped_env": list(STRIPPED),
        "runs": args.runs,
        "arms": list(args.arms),
        "median_threads_pre_fit": medians,
        "results": results,
    }
    args.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(f"median pre-fit threads: {medians}\nreceipt: {args.out}\n")
    return 0 if len(ok) == len(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", choices=ARMS, help="internal: run one arm in this interpreter")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "base.toml")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--idle-s", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=Path("pr063-q1-threads.json"))
    args = ap.parse_args()
    if args.child:
        sys.stdout.write(json.dumps(_child(args.child, args.config, args.idle_s)) + "\n")
        return 0
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
