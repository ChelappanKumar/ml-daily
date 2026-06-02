# Concept — bias / variance tradeoff

**Goal:** Explain bias and variance with a single worked example — polynomial regression with a degree sweep — and back the intuition with measured numbers.

## Summary

A model's expected squared error on unseen data decomposes into three pieces: **bias²** (how far the average prediction is from the truth across many training sets), **variance** (how much the prediction wobbles between training sets), and **irreducible noise** (whatever the labels can't tell you). Underfit models — too simple — have high bias and low variance. Overfit models — too flexible — have low bias and high variance. The "sweet spot" minimises the sum. The polynomial regression degree sweep below makes this measurable: as degree goes from 1 → 15, training error keeps falling but test error U-shapes, with the trough sitting at the degree that matches the true generating process.

## Key ideas

- **Bias** is *systematic* error. A linear model trying to fit `y = sin(x)` is biased no matter how much data you give it — the hypothesis class can't represent the truth.
- **Variance** is *sensitivity to the training sample*. A degree-15 polynomial fitted on 30 points will draw a wildly different curve if you swap 3 of those points. The hypothesis is so flexible it tracks the noise.
- **The decomposition is exact** for squared loss: `E[(y - ŷ)²] = Bias² + Var + σ²`. It estimates by fitting the same model class on many bootstrap samples of the training data and looking at the spread of predictions at each test point.
- **More data shrinks variance, not bias.** If your model is biased (wrong hypothesis class), throwing data at it won't fix it — you need a more expressive model. If your model is high-variance, more data flattens out the noise-tracking.
- **Regularization trades variance for bias on purpose.** Ridge / Lasso shrink coefficients toward zero, accepting a bit more bias to dramatically cut variance.
- **For modern deep nets, the classic U-curve mostly disappears** — see "double descent" (Belkin et al. 2019). Beyond a certain over-parameterization threshold, test error starts dropping again. This doesn't break the decomposition, it changes what "model complexity" means.

## Worked example — see `bias_variance_demo.py`

True function: `y = sin(1.5 π x) + ε`, with `ε ~ N(0, 0.2²)`. We sweep polynomial degrees 1 → 15 and for each degree:
1. Draw 200 random training sets of size n=30.
2. Fit a polynomial of that degree to each training set.
3. Predict on a fixed grid of held-out test points.
4. Compute bias² and variance at each test point, average over the grid.
5. Plot — train MSE keeps dropping, test MSE U-shapes, bias² drops and variance climbs as degree grows. Crossover happens around degree 3-5.

Run:
```
pip install numpy scikit-learn matplotlib
python bias_variance_demo.py
```
Outputs `bias_variance_curve.png` plus a numerical table you can read in the terminal.

## Open questions

- For my own work: when I'm tuning XGBoost depth/n_estimators on a Kaggle-style table, am I really navigating bias-variance or am I just chasing leak signal in CV? The decomposition assumes the train/test draws are iid — most real problems aren't quite.
- Double descent in regression: when does it actually appear with non-deep models? The classical setting + `degree >> n` shows it (interpolating overparameterized polynomials), but my intuition is weak.
- For neural net regularization — is dropout closer to "variance reduction" (averaging many subnetworks) or "implicit bias" (favouring solutions with redundant features)? Probably both, but the framing changes how I'd diagnose a failing model.

## References

- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, §2.9
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
- Belkin et al. 2019 — *Reconciling modern machine-learning practice and the bias-variance trade-off* (the double-descent paper) — https://arxiv.org/abs/1812.11118
