# Learning log — CI for ML with GitHub Actions

What worked:
- Splitting the workflow into TWO jobs (`tests` for code correctness, `sanity`
  for the model quality gate) makes the failure signal precise. If only `sanity`
  is red, I know it's a model regression, not a typo. Adding `needs: tests` on
  the sanity job means I don't waste compute training a model when the code
  doesn't even pass lint.
- Putting the pipeline in a single `build_pipeline()` function (instead of
  inline in main) was the biggest CI win. The unit test `test_pipeline_has_expected_steps`
  and the training script and (a future) inference service all import the exact
  same artifact. Drift between train and serve disappears.
- `concurrency: cancel-in-progress: true` on the same ref was important — without
  it, every push to a busy PR branch stacks up another ML run. With it, only
  the latest commit's run survives.
- `actions/setup-python@v5` with `cache: pip` cut cold-start install time from
  ~90s to ~15s after the first run. Free and worth it.

What surprised me:
- GitHub Actions doesn't fail a step just because pytest emitted warnings —
  it only fails on non-zero exit. So the model-regression guard had to use
  `sys.exit(1)` explicitly, not just `print("FAILED")` and continue.
- `paths:` filters on PR triggers are evaluated against the diff, not the
  whole tree. Initially had `paths: ["pipelines/**"]` and was confused why
  the workflow didn't fire when I edited only the workflow YAML — needed to
  add `.github/workflows/ml-ci.yml` to the path list explicitly.
- `multi_class="auto"` in `LogisticRegression` will be deprecated in newer
  sklearn versions. Worth leaving the kwarg explicit so the deprecation
  warning is visible in CI logs, then removing it later.

What I'd try next:
- Cache the trained model as a workflow artifact and load it in a downstream
  "inference smoke test" job. Catches serialization bugs (joblib version,
  numpy dtype) that unit tests miss.
- Add a baseline-comparison step: store the latest main-branch metrics as a
  release asset, compare the PR's metrics against it, post a comment with
  the delta. This is the real "model regression catcher" — fixed thresholds
  drift over time and become meaningless.
- Run the sanity job on a matrix of sklearn versions (1.4, 1.5, 1.6) so
  pinning bumps don't surprise us in production.
- Move the heavier sanity training onto a self-hosted runner once it gets
  past ~5 minutes — free GitHub runners are fine until they're not.
