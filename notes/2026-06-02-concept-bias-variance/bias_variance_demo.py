"""Bias / variance decomposition — measured, not hand-waved.

Problem this solves:
    The textbook formula `E[(y - ŷ)²] = Bias² + Var + σ²` is easy to write down
    and easy to misunderstand. This script makes it concrete: pick a known
    generating function, sweep model complexity, and *measure* bias and
    variance from many bootstrap fits. The numbers and the U-shaped test
    curve match the textbook story.

What it does:
    1. Synthesize y = sin(1.5πx) + N(0, 0.2²).
    2. For each polynomial degree d in [1..15]:
        - Draw `n_trials` independent training sets of size `n_train`.
        - Fit a degree-d polynomial to each.
        - Predict on a fixed test grid.
        - Bias²(x) = (mean prediction over trials - true f(x))²
        - Var(x)   = variance of prediction at x over trials
        - Train MSE and test MSE averaged over trials.
    3. Plot the four curves (train MSE, test MSE, bias², variance) on one chart.
    4. Print a sortable table so the optimal degree is obvious.

Run:
    pip install numpy scikit-learn matplotlib
    python bias_variance_demo.py
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

RNG = np.random.default_rng(42)
NOISE_STD = 0.2
N_TRAIN = 30
N_TEST = 200
N_TRIALS = 200
DEGREES = list(range(1, 16))


def true_fn(x: np.ndarray) -> np.ndarray:
    return np.sin(1.5 * np.pi * x)


def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = RNG.uniform(0.0, 1.0, size=n)
    y = true_fn(x) + RNG.normal(0.0, NOISE_STD, size=n)
    return x.reshape(-1, 1), y


def fit_polynomial(degree: int, x: np.ndarray, y: np.ndarray):
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    model.fit(x, y)
    return model


def evaluate_degree(degree: int) -> dict[str, float]:
    """Run `N_TRIALS` bootstrap fits at this degree; return summary statistics."""
    x_test = np.linspace(0.0, 1.0, N_TEST).reshape(-1, 1)
    f_true = true_fn(x_test.ravel())

    preds = np.empty((N_TRIALS, N_TEST))
    train_mses = np.empty(N_TRIALS)

    for t in range(N_TRIALS):
        x_tr, y_tr = sample(N_TRAIN)
        model = fit_polynomial(degree, x_tr, y_tr)
        preds[t] = model.predict(x_test)
        train_mses[t] = float(np.mean((model.predict(x_tr) - y_tr) ** 2))

    mean_pred = preds.mean(axis=0)
    bias_sq = float(np.mean((mean_pred - f_true) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    # Test MSE on the noisy test draws would add σ²; here we compare to f_true,
    # which separates reducible error cleanly. To get textbook E[test MSE],
    # add NOISE_STD**2.
    test_mse_reducible = float(np.mean((preds - f_true) ** 2))
    test_mse_total = test_mse_reducible + NOISE_STD ** 2
    return {
        "degree": degree,
        "train_mse": float(train_mses.mean()),
        "test_mse": test_mse_total,
        "bias_sq": bias_sq,
        "variance": variance,
        "irreducible": NOISE_STD ** 2,
    }


def print_table(rows: list[dict[str, float]]) -> None:
    headers = ["degree", "train_mse", "test_mse", "bias_sq", "variance", "irreducible"]
    widths = [max(len(h), 10) for h in headers]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    best = min(rows, key=lambda r: r["test_mse"])
    for r in rows:
        marker = "  <-- best" if r is best else ""
        cells = [str(r["degree"]).ljust(widths[0])] + [
            f"{r[h]:.4f}".ljust(widths[i]) for i, h in enumerate(headers[1:], 1)
        ]
        print("  ".join(cells) + marker)


def plot(rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt  # imported lazily so the script works headless
    except ImportError:
        print("matplotlib not installed — skipping plot. `pip install matplotlib` to enable.")
        return

    degrees = [r["degree"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, [r["train_mse"] for r in rows], "o-", label="train MSE")
    ax.plot(degrees, [r["test_mse"] for r in rows], "o-", label="test MSE")
    ax.plot(degrees, [r["bias_sq"] for r in rows], "o--", label="bias²")
    ax.plot(degrees, [r["variance"] for r in rows], "o--", label="variance")
    ax.axhline(NOISE_STD ** 2, ls=":", color="gray", label=f"σ² = {NOISE_STD ** 2:.2f}")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("error")
    ax.set_yscale("log")
    ax.set_title(f"Bias-variance sweep on y = sin(1.5πx) + ε   (n_train={N_TRAIN}, trials={N_TRIALS})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "bias_variance_curve.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot to {out}")


def main() -> None:
    print(f"Sweeping degrees {DEGREES[0]}..{DEGREES[-1]} with {N_TRIALS} trials each, n_train={N_TRAIN}.")
    rows = [evaluate_degree(d) for d in DEGREES]
    print()
    print_table(rows)
    plot(rows)

    best = min(rows, key=lambda r: r["test_mse"])
    print(
        f"\nBest degree by test MSE: {best['degree']}  "
        f"(test_mse={best['test_mse']:.4f}, bias²={best['bias_sq']:.4f}, var={best['variance']:.4f})"
    )
    print("Sanity: train MSE should be monotonically non-increasing in degree, and the test curve should U-shape.")


if __name__ == "__main__":
    main()
