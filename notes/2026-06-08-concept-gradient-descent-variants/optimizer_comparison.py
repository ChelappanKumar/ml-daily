"""SGD vs SGD+momentum vs Adam vs AdamW — head-to-head, from-scratch optimizers.

Problem this solves:
    Most people read the Adam paper, accept that "Adam works better", and move
    on without ever implementing the three update rules side by side. The
    actual differences (per-parameter step sizes, bias correction, decoupled
    weight decay) become muscle-memory when you write them in 6 lines each.
    This file does exactly that — no `torch.optim`, all four optimizers
    implemented from primitives so you can see every term.

What's here:
    1. `Optimizer` ABC with `step(params, grads, params_data)`.
    2. Four concrete optimizers: SGD, SGD-momentum, Adam, AdamW. Each is
       ~10 lines, mirroring the math in the README.
    3. A 2-layer MLP trained from scratch (manual forward + backward — no
       autograd) on a noisy 1D regression task `y = sin(2x) + ε`.
    4. Side-by-side run: same model, same data, same seed — only the
       optimizer differs. Prints per-optimizer train+test loss curves and
       a final ranking. With `weight_decay > 0` the gap between Adam and
       AdamW shows up clearly.

Run:
    pip install numpy
    python optimizer_comparison.py
"""
from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# ---------- problem setup ----------

SEED = 7
N_TRAIN = 200
N_TEST = 200
NOISE_STD = 0.15
HIDDEN = 32
N_STEPS = 600
BATCH_SIZE = 32
LR = 1e-2
WD = 1e-2


def make_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_tr = rng.uniform(-3.0, 3.0, size=(N_TRAIN, 1))
    y_tr = np.sin(2.0 * x_tr) + rng.normal(0.0, NOISE_STD, size=(N_TRAIN, 1))
    x_te = rng.uniform(-3.0, 3.0, size=(N_TEST, 1))
    y_te = np.sin(2.0 * x_te)  # noise-free test (clean target)
    return x_tr, y_tr, x_te, y_te


# ---------- optimizers (from primitives) ----------

class Optimizer(ABC):
    """Update params in-place given grads. State is per-parameter, keyed by id()."""

    def __init__(self, lr: float, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.t = 0
        self.state: dict[int, dict[str, np.ndarray]] = {}

    def _slot(self, param: np.ndarray, key: str) -> np.ndarray:
        s = self.state.setdefault(id(param), {})
        if key not in s:
            s[key] = np.zeros_like(param)
        return s[key]

    @abstractmethod
    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None: ...


class SGD(Optimizer):
    """Plain SGD with optional coupled L2 weight decay (the 'wrong' way)."""

    def step(self, params, grads):
        self.t += 1
        for p, g in zip(params, grads):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            p -= self.lr * g


class SGDMomentum(Optimizer):
    """SGD with classical momentum (β=0.9). v_t = β·v_{t-1} + g_t; θ -= lr·v_t."""

    def __init__(self, lr: float, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.momentum = momentum

    def step(self, params, grads):
        self.t += 1
        for p, g in zip(params, grads):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            v = self._slot(p, "v")
            v *= self.momentum
            v += g
            p -= self.lr * v


class Adam(Optimizer):
    """Adam with COUPLED L2 weight decay (i.e. wd folded into the gradient).
    This is the failure mode AdamW fixes."""

    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def step(self, params, grads):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for p, g in zip(params, grads):
            if self.weight_decay > 0:
                # L2-style: fold wd into gradient BEFORE adaptive scaling.
                # This is what makes the regularization wrong.
                g = g + self.weight_decay * p
            m = self._slot(p, "m")
            v = self._slot(p, "v")
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with DECOUPLED weight decay. The fix: apply wd directly to params,
    NOT to the gradient. This is what every modern transformer pretrain uses."""

    def step(self, params, grads):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for p, g in zip(params, grads):
            m = self._slot(p, "m")
            v = self._slot(p, "v")
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            # Adam step + decoupled weight-decay term.
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p)


# ---------- 2-layer MLP (forward + manual backward) ----------

@dataclass
class MLP:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    cache: dict = field(default_factory=dict)

    @classmethod
    def init(cls, in_dim: int, hidden: int, out_dim: int, rng: np.random.Generator) -> "MLP":
        # He init for the ReLU layer.
        W1 = rng.standard_normal((in_dim, hidden)) * math.sqrt(2.0 / in_dim)
        b1 = np.zeros((hidden,))
        W2 = rng.standard_normal((hidden, out_dim)) * math.sqrt(2.0 / hidden)
        b2 = np.zeros((out_dim,))
        return cls(W1=W1, b1=b1, W2=W2, b2=b2)

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        y = h1 @ self.W2 + self.b2
        self.cache = {"x": x, "z1": z1, "h1": h1}
        return y

    def backward(self, dy: np.ndarray) -> list[np.ndarray]:
        x, z1, h1 = self.cache["x"], self.cache["z1"], self.cache["h1"]
        n = x.shape[0]
        dW2 = h1.T @ dy / n
        db2 = dy.mean(axis=0)
        dh1 = dy @ self.W2.T
        dz1 = dh1 * (z1 > 0)
        dW1 = x.T @ dz1 / n
        db1 = dz1.mean(axis=0)
        return [dW1, db1, dW2, db2]


# ---------- training loop ----------

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def train_one(opt: Optimizer, name: str, x_tr: np.ndarray, y_tr: np.ndarray,
              x_te: np.ndarray, y_te: np.ndarray, rng: np.random.Generator) -> dict:
    # Same init seed for every optimizer so the comparison is honest.
    init_rng = np.random.default_rng(SEED)
    model = MLP.init(in_dim=1, hidden=HIDDEN, out_dim=1, rng=init_rng)

    train_curve: list[float] = []
    test_curve: list[float] = []
    for step in range(1, N_STEPS + 1):
        idx = rng.integers(0, x_tr.shape[0], size=BATCH_SIZE)
        xb, yb = x_tr[idx], y_tr[idx]

        y_pred = model.forward(xb)
        loss = mse_loss(y_pred, yb)
        dy = (y_pred - yb) * 2.0  # d(mse)/d(y_pred), n-mean done inside backward

        grads = model.backward(dy)
        opt.step(model.params(), grads)

        if step % 50 == 0 or step == 1:
            train_curve.append(loss)
            test_curve.append(mse_loss(model.forward(x_te), y_te))

    final_train = mse_loss(model.forward(x_tr), y_tr)
    final_test = mse_loss(model.forward(x_te), y_te)
    return {
        "name": name,
        "train_curve": train_curve,
        "test_curve": test_curve,
        "final_train": final_train,
        "final_test": final_test,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    x_tr, y_tr, x_te, y_te = make_data(rng)

    configs = [
        ("SGD",          SGD(lr=LR, weight_decay=WD)),
        ("SGD+momentum", SGDMomentum(lr=LR, momentum=0.9, weight_decay=WD)),
        ("Adam",         Adam(lr=LR, weight_decay=WD)),
        ("AdamW",        AdamW(lr=LR, weight_decay=WD)),
    ]

    results = []
    for name, opt in configs:
        # Fresh RNG for batch sampling so every optimizer sees the same batches.
        batch_rng = np.random.default_rng(SEED + 1)
        results.append(train_one(opt, name, x_tr, y_tr, x_te, y_te, batch_rng))

    # Print loss curves side-by-side at each checkpoint.
    checkpoints = list(range(1, N_STEPS + 1, 50))[: len(results[0]["train_curve"])]
    header = f"{'step':>6}  " + "  ".join(f"{r['name']:>22}" for r in results)
    print(header)
    print("-" * len(header))
    for i, step in enumerate(checkpoints):
        cells = [f"train={r['train_curve'][i]:.4f}/test={r['test_curve'][i]:.4f}" for r in results]
        print(f"{step:>6}  " + "  ".join(f"{c:>22}" for c in cells))

    print("\nFinal results (sorted by test loss):")
    ranked = sorted(results, key=lambda r: r["final_test"])
    for r in ranked:
        print(f"  {r['name']:<14}  train={r['final_train']:.4f}  test={r['final_test']:.4f}")

    # Highlight the headline contrast.
    adam = next(r for r in results if r["name"] == "Adam")
    adamw = next(r for r in results if r["name"] == "AdamW")
    delta = adam["final_test"] - adamw["final_test"]
    print(f"\nHeadline: AdamW test loss is {delta:+.4f} lower than Adam at wd={WD}.")
    print("With wd=0 the two converge; the gap opens as weight decay grows.")


if __name__ == "__main__":
    main()
