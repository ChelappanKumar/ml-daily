# Learning log — DVC data versioning

What worked:
- Content-addressed storage is the core insight. Hash the file contents with
  md5, store at `.dvc/cache/ab/cdef1234…` (first 2 chars = subdir). Two
  identical datasets share one cache entry; a single-byte change creates a new
  entry. This is exactly how git objects and Docker layers work — DVC just
  applies the same idea to large binary files. The pointer file (.dvc) in git
  is tiny (< 1 KB); the actual data stays outside git entirely.
- Downstream staleness propagation was the tricky part. `status()` checks
  whether dep hashes match locked hashes — but it can only see the current
  state, not what a stage will produce after running. The fix: once a stage
  runs, set `force_downstream = True` so every subsequent stage re-runs
  regardless of what status() reported. Without this, split would skip even
  though normalize just rewrote its input.
- The `.dvc` pointer file format is intentionally minimal: `{md5, size, path}`.
  Size is redundant (derivable from cache) but lets you detect truncated files
  without reading the whole thing. Real DVC adds `nfiles` for directory
  tracking, but single-file versioning only needs three fields.

What surprised me:
- The status check before reproduce showed `split` and `featurize` as
  "up-to-date" even though they were about to re-run (because normalize was
  stale). This is not a bug — it's the correct answer given current on-disk
  state. The reproduce logic overrides status with `force_downstream` once a
  stage actually runs. Real DVC behaves identically: `dvc status` can lag
  behind `dvc repro` in multi-stage pipelines.
- md5 is fast enough for small files but DVC ships with xxhash as its default
  since 2.x because md5 is ~3× slower on large files. The cache layout is
  identical regardless of hash function — only the hash length changes.
- Restoring v1 from cache is instantaneous (one file copy) even after v2 ran
  and overwrote the file. The cache is write-once — you never mutate a cached
  entry, only add new ones. Rollback is free.

What I'd try next:
- Directory tracking: DVC's `.dir` cache format stores a JSON manifest
  `[{md5, relpath}, ...]` for every file in a directory. Implement
  `DVCCache.put_dir(dir_path)` returning a `.dir` hash, with each member file
  cached individually. This is how DVC handles ImageNet-sized datasets.
- dvc.lock serialisation: write/read the locked dep+out hashes to YAML so the
  pipeline survives process restarts. Right now all state is in memory.
- Remote storage backend: replace `shutil.copy2` with `boto3.upload_file` /
  `download_file` to simulate `dvc push` / `dvc pull`. The cache abstraction
  is already the right interface — only the I/O changes.
- Compare xxhash vs md5 throughput on large (>100 MB) CSV files to understand
  when the hash choice actually matters in practice.
