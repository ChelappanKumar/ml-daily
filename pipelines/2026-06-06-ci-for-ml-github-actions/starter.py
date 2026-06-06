"""ML pipeline + sanity-check test, designed to run under CI.

Problem this solves:
    Most ML teams catch model regressions weeks late — a hyperparameter change,
    a sklearn version bump, or a data preprocessor edit silently degrades
    accuracy and nobody notices until production metrics tank. CI for ML adds
    fast, deterministic guardrails to PR: train a tiny model on a fixed dataset,
    assert minimum quality, fail the build if it drops.

What's in this file:
    1. `build_pipeline()` — sklearn Pipeline with imputation + scaling + LogReg.
       Same code is used by training, batch inference, and the CI sanity test.
    2. `train_and_evaluate()` — deterministic train on the iris dataset with a
       fixed seed, returns metrics dict.
    3. A CI gate: assert accuracy >= ACC_THRESHOLD and macro-F1 >= F1_THRESHOLD,
       else exit non-zero (so GitHub Actions marks the run as failed).
    4. Three pytest-style unit tests (`test_*`) exercising pipeline shape,
       prediction determinism, and quality threshold. These run under
       `pytest starter.py`.

Companion files:
    - `.github/workflows/ml-ci.yml` (at repo root) — the actual workflow.
    - `NOTES.md` — what I learned wiring this up.

Run locally:
    pip install scikit-learn pytest
    python starter.py              # runs the CI gate
    pytest starter.py -v           # runs the unit tests
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# CI thresholds — if either drops below, the build fails.
# These are deliberately conservative so a real regression trips them but
# normal seed/floating-point jitter doesn't.
ACC_THRESHOLD = 0.90
F1_THRESHOLD = 0.90
SEED = 42


@dataclass
class TrainResult:
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int


def build_pipeline() -> Pipeline:
    """Single source of truth for the model. Used by train, infer, AND tests.

    Putting the pipeline in one function (instead of inline in main) is the
    biggest win for CI — the unit test exercises the exact same artifact that
    serves predictions.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=SEED, multi_class="auto")),
    ])


def train_and_evaluate(seed: int = SEED) -> TrainResult:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y,
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return TrainResult(
        accuracy=float(accuracy_score(y_test, preds)),
        macro_f1=float(f1_score(y_test, preds, average="macro")),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
    )


# ---------- CI gate ----------

def ci_gate() -> int:
    result = train_and_evaluate()
    print(json.dumps(asdict(result), indent=2))

    failures: list[str] = []
    if result.accuracy < ACC_THRESHOLD:
        failures.append(f"accuracy {result.accuracy:.4f} < {ACC_THRESHOLD}")
    if result.macro_f1 < F1_THRESHOLD:
        failures.append(f"macro_f1 {result.macro_f1:.4f} < {F1_THRESHOLD}")

    if failures:
        print("\nCI gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nCI gate PASSED (acc={result.accuracy:.4f}, f1={result.macro_f1:.4f})")
    return 0


# ---------- pytest-style tests ----------
# Run with: pytest starter.py -v

def test_pipeline_has_expected_steps() -> None:
    """Schema test — fail loudly if someone reorders or removes a step."""
    pipe = build_pipeline()
    names = [name for name, _ in pipe.steps]
    assert names == ["impute", "scale", "clf"], f"unexpected pipeline shape: {names}"


def test_predictions_are_deterministic() -> None:
    """Same seed twice -> identical predictions. Catches non-determinism bugs
    introduced by e.g. unset `random_state` or env-dependent randomness."""
    a = train_and_evaluate(seed=SEED)
    b = train_and_evaluate(seed=SEED)
    assert a.accuracy == b.accuracy
    assert a.macro_f1 == b.macro_f1


def test_meets_quality_threshold() -> None:
    """The actual model-regression guard."""
    result = train_and_evaluate()
    assert result.accuracy >= ACC_THRESHOLD, f"accuracy regressed: {result.accuracy:.4f}"
    assert result.macro_f1 >= F1_THRESHOLD, f"macro_f1 regressed: {result.macro_f1:.4f}"


def test_handles_missing_values() -> None:
    """If preprocessor regresses, this fails loudly instead of crashing
    at inference time on a NaN-bearing row."""
    pipe = build_pipeline()
    X, y = load_iris(return_X_y=True)
    pipe.fit(X, y)
    # Inject NaN into a held-out row; predict should still produce a class.
    row = X[0].copy().astype(float)
    row[1] = np.nan
    pred = pipe.predict(row.reshape(1, -1))
    assert pred.shape == (1,)
    assert int(pred[0]) in {0, 1, 2}


if __name__ == "__main__":
    sys.exit(ci_gate())
