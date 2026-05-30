"""Data validation pipeline — Great Expectations-style, dependency-light.

Problem this solves:
    ML pipelines silently break when input data drifts (new categorical values,
    null spikes, distribution shifts, schema changes). The fix is to declare
    your data expectations explicitly, run them on every input batch, and
    halt the pipeline when an expectation fails — surfacing the problem at the
    data layer instead of at the model's output where it's much harder to debug.

This file implements the core ideas from Great Expectations from scratch so the
mechanics are visible:
    1. An `Expectation` base class with `validate(df) -> ExpectationResult`.
    2. Concrete expectations (not_null, in_set, between, regex, unique, schema).
    3. A `Suite` that runs a list of expectations and produces a structured report.
    4. A `validate_or_raise` entrypoint that integrates into a training/inference
       pipeline — pass = continue; any critical failure = raise + structured log.

Why not just import great_expectations?
    Real projects should — `pip install great-expectations`. This file is
    deliberately minimal so the concepts are concrete.

Run:
    python starter.py
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd


# ---------- result objects ----------

@dataclass
class ExpectationResult:
    expectation: str
    column: str | None
    success: bool
    observed: dict[str, Any] = field(default_factory=dict)
    severity: str = "critical"  # critical | warning


@dataclass
class SuiteReport:
    total: int
    passed: int
    failed: int
    critical_failures: int
    results: list[ExpectationResult]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["results"] = [asdict(r) for r in self.results]
        return d


# ---------- base ----------

class Expectation(ABC):
    severity: str = "critical"

    def __init__(self, column: str | None = None, severity: str | None = None):
        self.column = column
        if severity is not None:
            self.severity = severity

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ExpectationResult: ...

    def _result(self, success: bool, **observed: Any) -> ExpectationResult:
        return ExpectationResult(
            expectation=type(self).__name__,
            column=self.column,
            success=success,
            observed=observed,
            severity=self.severity,
        )


# ---------- expectations ----------

class ExpectColumnsToExist(Expectation):
    """Schema check: every named column must be present."""

    def __init__(self, columns: list[str], severity: str = "critical"):
        super().__init__(column=None, severity=severity)
        self.columns = columns

    def validate(self, df: pd.DataFrame) -> ExpectationResult:
        missing = [c for c in self.columns if c not in df.columns]
        return self._result(success=not missing, missing=missing, expected=self.columns)


class ExpectColumnValuesToNotBeNull(Expectation):
    def __init__(self, column: str, mostly: float = 1.0, severity: str = "critical"):
        super().__init__(column=column, severity=severity)
        assert 0.0 <= mostly <= 1.0
        self.mostly = mostly

    def validate(self, df: pd.DataFrame) -> ExpectationResult:
        if self.column not in df.columns:
            return self._result(success=False, error="column_missing")
        null_rate = df[self.column].isna().mean()
        non_null_rate = 1.0 - null_rate
        return self._result(
            success=non_null_rate >= self.mostly,
            null_rate=round(float(null_rate), 4),
            threshold=self.mostly,
        )


class ExpectColumnValuesToBeInSet(Expectation):
    def __init__(self, column: str, value_set: Iterable[Any], severity: str = "critical"):
        super().__init__(column=column, severity=severity)
        self.value_set = set(value_set)

    def validate(self, df: pd.DataFrame) -> ExpectationResult:
        if self.column not in df.columns:
            return self._result(success=False, error="column_missing")
        present = set(df[self.column].dropna().unique())
        unexpected = sorted(present - self.value_set, key=str)
        return self._result(
            success=not unexpected,
            unexpected=unexpected[:10],
            unexpected_count=len(unexpected),
        )


class ExpectColumnValuesToBeBetween(Expectation):
    def __init__(self, column: str, min_value: float, max_value: float, severity: str = "critical"):
        super().__init__(column=column, severity=severity)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, df: pd.DataFrame) -> ExpectationResult:
        if self.column not in df.columns:
            return self._result(success=False, error="column_missing")
        s = pd.to_numeric(df[self.column], errors="coerce").dropna()
        out_of_range = int(((s < self.min_value) | (s > self.max_value)).sum())
        return self._result(
            success=out_of_range == 0,
            out_of_range=out_of_range,
            min_seen=float(s.min()) if len(s) else None,
            max_seen=float(s.max()) if len(s) else None,
        )


class ExpectColumnValuesToMatchRegex(Expectation):
    def __init__(self, column: str, pattern: str, severity: str = "critical"):
        super().__init__(column=column, severity=severity)
        self.pattern = re.compile(pattern)

    def validate(self, df: pd.DataFrame) -> ExpectationResult:
        if self.column not in df.columns:
            return self._result(success=False, error="column_missing")
        s = df[self.column].dropna().astype(str)
        bad = s[~s.str.match(self.pattern)]
        return self._result(success=bad.empty, mismatches=int(bad.shape[0]), examples=bad.head(3).tolist())


class ExpectColumnValuesToBeUnique(Expectation):
    def validate(self, df: pd.DataFrame) -> ExpectationResult:
        if self.column not in df.columns:
            return self._result(success=False, error="column_missing")
        dup_count = int(df[self.column].duplicated().sum())
        return self._result(success=dup_count == 0, duplicate_count=dup_count)


# ---------- suite ----------

class Suite:
    """A named collection of expectations that produces a structured report."""

    def __init__(self, name: str, expectations: list[Expectation]):
        self.name = name
        self.expectations = expectations

    def validate(self, df: pd.DataFrame) -> SuiteReport:
        results = [exp.validate(df) for exp in self.expectations]
        passed = sum(r.success for r in results)
        failed = len(results) - passed
        critical = sum(1 for r in results if not r.success and r.severity == "critical")
        return SuiteReport(
            total=len(results), passed=passed, failed=failed,
            critical_failures=critical, results=results,
        )


class DataValidationError(Exception):
    """Raised when critical expectations fail. The pipeline should halt."""

    def __init__(self, report: SuiteReport):
        self.report = report
        super().__init__(f"{report.critical_failures} critical expectation(s) failed out of {report.total}")


def validate_or_raise(df: pd.DataFrame, suite: Suite, *, log: bool = True) -> SuiteReport:
    """Pipeline-integration entrypoint. Halts on critical failure."""
    report = suite.validate(df)
    if log:
        print(json.dumps({"suite": suite.name, "report": report.to_dict()}, indent=2, default=str))
    if report.critical_failures > 0:
        raise DataValidationError(report)
    return report


# ---------- demo ----------

def _good_batch() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":  [101, 102, 103, 104, 105],
        "email":    ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com"],
        "age":      [23, 45, 31, 27, 52],
        "country":  ["US", "IN", "US", "DE", "IN"],
        "amount":   [12.5, 99.0, 3.14, 7.0, 21.0],
    })


def _bad_batch() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":  [201, 202, 202, 204, None],                  # duplicate + null
        "email":    ["a@x.com", "bad-email", "c@x.com", "d@x.com", "e@x.com"],  # regex fail
        "age":      [23, 45, 31, 999, 52],                       # out of range
        "country":  ["US", "MARS", "US", "DE", "IN"],            # unexpected value
        "amount":   [12.5, 99.0, 3.14, 7.0, 21.0],
    })


def build_suite() -> Suite:
    return Suite("user_events", [
        ExpectColumnsToExist(["user_id", "email", "age", "country", "amount"]),
        ExpectColumnValuesToNotBeNull("user_id"),
        ExpectColumnValuesToBeUnique("user_id"),
        ExpectColumnValuesToMatchRegex("email", r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
        ExpectColumnValuesToBeBetween("age", 0, 120),
        ExpectColumnValuesToBeInSet("country", {"US", "IN", "DE", "FR", "UK"}),
        ExpectColumnValuesToBeBetween("amount", 0.0, 10_000.0, severity="warning"),
    ])


def main() -> None:
    suite = build_suite()

    print("=== Validating GOOD batch — should pass ===")
    validate_or_raise(_good_batch(), suite)

    print("\n=== Validating BAD batch — should raise ===")
    try:
        validate_or_raise(_bad_batch(), suite)
    except DataValidationError as e:
        print(f"\nCaught DataValidationError: {e}")
        print("Pipeline halted before bad data hit the model. This is the win.")


if __name__ == "__main__":
    main()
