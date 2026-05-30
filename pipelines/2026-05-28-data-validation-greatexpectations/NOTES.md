# Learning log — Data validation with Great Expectations

What worked:
- Modeling each expectation as a class with a single `validate(df) -> ExpectationResult`
  method keeps the abstraction small and lets the suite runner stay dumb (just a list-comp).
- Separating `critical` from `warning` severity meant warnings show up in the report
  but don't halt the pipeline — useful for things like "amount usually under $10k" where
  occasional outliers are fine but a sudden cluster is worth noticing.
- Returning a structured `SuiteReport` (not just a bool) made the JSON log
  immediately useful for an observability stack — could pipe straight into BigQuery.

What surprised me:
- The hardest part isn't writing checks, it's deciding `mostly` thresholds. Strict
  `null_rate == 0` breaks too often in production; `mostly=0.95` ignores real regressions.
  Real teams seem to tune these per-column based on observed historical distribution.
- `pandas.Series.str.match` only anchors at the start, not the end. Need `^...$`
  explicitly or the regex check silently passes garbage. Caught this writing test batches.

What I'd try next:
- Add distribution-shift expectations: KS test on a numeric column vs. a reference
  snapshot. Halt if drift > threshold.
- Wire this into a tiny Prefect/Airflow DAG so the validation step is a first-class
  task, not buried inside the model training script.
- Swap the hand-rolled version for actual `great_expectations` and compare ergonomics —
  guessing the real library wins on data docs + checkpoints but is heavier to set up.
