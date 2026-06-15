"""DVC-inspired data versioning — content-addressed cache, .dvc pointers, pipeline DAG.

Problem this solves:
    git can version code but not large data files. Committing a 1 GB CSV bloats
    the repo and makes every clone slow. DVC fixes this with three ideas:

    1. Content-addressed cache — data is stored by its md5 hash, identical files
       share one cache entry, changed files get a new entry. Same principle as
       git objects and Docker layers.

    2. .dvc pointer files — tiny JSON files (md5 + size + path) replace the
       actual data in the git tree. git commits the pointer; DVC restores data.

    3. Pipeline stages (dvc.yaml / dvc.lock) — each stage records the md5 of
       its dep files. reproduce() skips a stage if nothing it depends on changed;
       only stale stages re-run. This is the "smart Make for ML" idea.

What this file builds (no DVC binary required, numpy only):
    DVCCache    — content-addressed store, aa/bb… shard layout
    DVCPointer  — .dvc file: write JSON {md5, size, path} and load it back
    DVCStage    — one pipeline step: cmd string, dep paths, out paths, callable fn
    DVCPipeline — DAG runner: status() detects stale stages, reproduce() re-runs them
    Demo        — 3-stage pipeline on a generated dataset (v1 then v2):
                    raw.csv → normalize → normalized.csv
                              split     → train.csv, test.csv
                              featurize → features.csv
                  Shows full run, up-to-date check, v2 dataset → partial re-run.

Run:
    pip install numpy
    python starter.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Content-addressed cache
# ─────────────────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DVCCache:
    """Stores files by md5. Layout mirrors DVC: cache_root/ab/cdef123…"""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:]

    def put(self, src: Path) -> str:
        """Copy src into the cache; return its md5."""
        digest = _md5(src)
        dst = self._cache_path(digest)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return digest

    def checkout(self, digest: str, dst: Path) -> None:
        """Restore a cached file to dst path."""
        cached = self._cache_path(digest)
        if not cached.exists():
            raise FileNotFoundError(f"cache miss: {digest}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dst)

    def has(self, digest: str) -> bool:
        return self._cache_path(digest).exists()


# ─────────────────────────────────────────────────────────────────────────────
# .dvc pointer files
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DVCPointer:
    """Mirrors a real .dvc file: just the md5, size, and original path."""
    path: str
    md5: str
    size: int

    def save(self, pointer_path: Path) -> None:
        pointer_path.write_text(
            json.dumps({"md5": self.md5, "size": self.size, "path": self.path}, indent=2)
        )

    @classmethod
    def load(cls, pointer_path: Path) -> "DVCPointer":
        d = json.loads(pointer_path.read_text())
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DVCStage:
    """One pipeline step.

    cmd   — display string (what dvc.yaml would contain)
    deps  — input file paths
    outs  — output file paths
    fn    — callable(deps, outs) that actually runs the transformation
    """
    name: str
    cmd: str
    deps: list[Path]
    outs: list[Path]
    fn: Callable[[list[Path], list[Path]], None]
    _dep_hashes: dict[str, str] = field(default_factory=dict)   # locked hashes
    _out_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class StageStatus:
    name: str
    state: str   # "changed" | "up-to-date" | "no-cache"
    changed_deps: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

class DVCPipeline:
    """Ordered list of stages with status checking and incremental reproduction."""

    def __init__(self, stages: list[DVCStage], cache: DVCCache) -> None:
        self.stages = stages
        self.cache = cache

    def _dep_hash(self, stage: DVCStage) -> dict[str, str]:
        return {str(p): _md5(p) for p in stage.deps if p.exists()}

    def status(self) -> list[StageStatus]:
        results: list[StageStatus] = []
        for stage in self.stages:
            if not stage._dep_hashes:
                results.append(StageStatus(name=stage.name, state="no-cache"))
                continue
            current = self._dep_hash(stage)
            changed = [k for k, v in current.items() if stage._dep_hashes.get(k) != v]
            state = "changed" if changed else "up-to-date"
            results.append(StageStatus(name=stage.name, state=state, changed_deps=changed))
        return results

    def reproduce(self, force: bool = False) -> RunReport:
        t0 = time.perf_counter()
        report = RunReport()
        # Propagate staleness: if a stage runs, downstream stages are forced.
        force_downstream = False
        for stage in self.stages:
            current = self._dep_hash(stage)
            stale = (
                force
                or force_downstream
                or not stage._dep_hashes
                or any(stage._dep_hashes.get(k) != v for k, v in current.items())
            )
            if stale:
                print(f"  running  [{stage.name}]  {stage.cmd}")
                stage.fn(stage.deps, stage.outs)
                # Cache outputs and lock hashes
                stage._dep_hashes = current
                stage._out_hashes = {
                    str(p): self.cache.put(p) for p in stage.outs if p.exists()
                }
                report.ran.append(stage.name)
                force_downstream = True   # invalidate everything after
            else:
                print(f"  skipped  [{stage.name}]  (up-to-date)")
                report.skipped.append(stage.name)
                force_downstream = False
        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        return report

    def checkout(self, stage_name: str) -> None:
        """Restore cached outputs of a stage (for switching dataset versions)."""
        stage = next(s for s in self.stages if s.name == stage_name)
        for p, digest in stage._out_hashes.items():
            self.cache.checkout(digest, Path(p))
            print(f"  checkout [{stage_name}] → {Path(p).name} (md5={digest[:8]}…)")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generator + pipeline stage functions
# ─────────────────────────────────────────────────────────────────────────────

def make_raw_csv(path: Path, n: int = 800, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)
    header = "x1,x2,x3,x4,label"
    rows = [",".join(str(round(v, 6)) for v in [*row, int(lbl)]) for row, lbl in zip(X, y)]
    path.write_text(header + "\n" + "\n".join(rows))


def _normalize(deps: list[Path], outs: list[Path]) -> None:
    lines = deps[0].read_text().splitlines()
    header = lines[0]
    data = np.array([[float(x) for x in r.split(",")] for r in lines[1:]])
    cols = data[:, :-1]
    lo, hi = cols.min(axis=0), cols.max(axis=0)
    cols_norm = (cols - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    data[:, :-1] = cols_norm
    rows = [",".join(str(round(v, 8)) for v in row) for row in data]
    outs[0].write_text(header + "\n" + "\n".join(rows))


def _split(deps: list[Path], outs: list[Path]) -> None:
    rng = np.random.default_rng(0)
    lines = deps[0].read_text().splitlines()
    header, rows = lines[0], lines[1:]
    idx = rng.permutation(len(rows))
    cut = int(0.8 * len(rows))
    train = [rows[i] for i in idx[:cut]]
    test = [rows[i] for i in idx[cut:]]
    outs[0].write_text(header + "\n" + "\n".join(train))
    outs[1].write_text(header + "\n" + "\n".join(test))


def _featurize(deps: list[Path], outs: list[Path]) -> None:
    lines = deps[0].read_text().splitlines()
    data = np.array([[float(x) for x in r.split(",")] for r in lines[1:]])
    x1, x2 = data[:, 0], data[:, 1]
    # Append x1*x2 and x1^2 as polynomial features
    extra = np.column_stack([x1 * x2, x1 ** 2])
    augmented = np.hstack([data[:, :-1], extra, data[:, -1:]])
    header = "x1,x2,x3,x4,x1x2,x1sq,label"
    rows = [",".join(str(round(v, 8)) for v in row) for row in augmented]
    outs[0].write_text(header + "\n" + "\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path("/tmp/dvc_demo")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    data_dir = root / "data"
    data_dir.mkdir()
    cache = DVCCache(root / ".dvc" / "cache")

    raw      = data_dir / "raw.csv"
    norm     = data_dir / "normalized.csv"
    train    = data_dir / "train.csv"
    test     = data_dir / "test.csv"
    features = data_dir / "features.csv"

    stages = [
        DVCStage(
            name="normalize",
            cmd="python normalize.py --input raw.csv --output normalized.csv",
            deps=[raw], outs=[norm], fn=_normalize,
        ),
        DVCStage(
            name="split",
            cmd="python split.py --input normalized.csv --train train.csv --test test.csv",
            deps=[norm], outs=[train, test], fn=_split,
        ),
        DVCStage(
            name="featurize",
            cmd="python featurize.py --input train.csv --output features.csv",
            deps=[train], outs=[features], fn=_featurize,
        ),
    ]
    pipeline = DVCPipeline(stages, cache)

    # ── V1: generate dataset, first full run ─────────────────────────────────
    print("=" * 64)
    print("DATASET v1 — first reproduce (all stages stale)")
    print("=" * 64)
    make_raw_csv(raw, n=800, seed=42)
    raw_ptr_v1 = DVCPointer(path="data/raw.csv", md5=cache.put(raw), size=raw.stat().st_size)
    raw_ptr_v1.save(root / "raw.csv.dvc")

    report = pipeline.reproduce()
    print(f"\nRan    : {report.ran}")
    print(f"Skipped: {report.skipped}")
    print(f"Time   : {report.elapsed_ms:.1f} ms")

    n_train = len(train.read_text().splitlines()) - 1
    n_test  = len(test.read_text().splitlines()) - 1
    n_feat  = len(features.read_text().splitlines()[0].split(","))
    print(f"\nOutput : train={n_train} rows, test={n_test} rows, features={n_feat} cols")

    # ── status: nothing changed ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("STATUS — no changes since last run")
    print("=" * 64)
    for s in pipeline.status():
        mark = "✓" if s.state == "up-to-date" else "!"
        print(f"  [{mark}] {s.name:<12} {s.state}")

    # ── Reproduce again: all stages should be skipped ─────────────────────────
    print("\n" + "=" * 64)
    print("REPRODUCE — should skip everything")
    print("=" * 64)
    report2 = pipeline.reproduce()
    print(f"\nRan    : {report2.ran}")
    print(f"Skipped: {report2.skipped}")

    # ── V2: change the dataset, reproduce should re-run all stages ───────────
    print("\n" + "=" * 64)
    print("DATASET v2 — larger dataset (n=1200, new seed); partial re-run")
    print("=" * 64)
    make_raw_csv(raw, n=1200, seed=99)
    raw_ptr_v2 = DVCPointer(path="data/raw.csv", md5=cache.put(raw), size=raw.stat().st_size)
    raw_ptr_v2.save(root / "raw.csv.dvc")
    print(f"Pointer v1 md5: {raw_ptr_v1.md5[:12]}…")
    print(f"Pointer v2 md5: {raw_ptr_v2.md5[:12]}…  (new content = new hash)")
    print()

    for s in pipeline.status():
        mark = "!" if s.state != "up-to-date" else "✓"
        suffix = f"  ← changed deps: {s.changed_deps}" if s.changed_deps else ""
        print(f"  [{mark}] {s.name:<12} {s.state}{suffix}")

    print()
    report3 = pipeline.reproduce()
    n_train2 = len(train.read_text().splitlines()) - 1
    n_test2  = len(test.read_text().splitlines()) - 1
    print(f"\nRan    : {report3.ran}")
    print(f"Skipped: {report3.skipped}")
    print(f"Output : train={n_train2} rows, test={n_test2} rows  (grew with dataset)")

    # ── Restore v1 from cache and verify ─────────────────────────────────────
    print("\n" + "=" * 64)
    print("CHECKOUT — restore v1 dataset from cache")
    print("=" * 64)
    cache.checkout(raw_ptr_v1.md5, raw)
    restored_rows = len(raw.read_text().splitlines()) - 1
    print(f"  restored raw.csv: {restored_rows} rows (v1 had 800)")
    print(f"  pointer md5 matches cache: {cache.has(raw_ptr_v1.md5)}")

    # Show cache layout
    print("\n" + "=" * 64)
    print("CACHE LAYOUT (first 5 entries)")
    print("=" * 64)
    entries = sorted(cache.root.rglob("*") )
    files = [e for e in entries if e.is_file()]
    for f in files[:5]:
        rel = f.relative_to(cache.root)
        size_kb = f.stat().st_size / 1024
        print(f"  .dvc/cache/{rel}  ({size_kb:.1f} KB)")
    print(f"  … {len(files)} total cache entries")

    print("\nDone. DVC store at:", cache.root)


if __name__ == "__main__":
    main()
