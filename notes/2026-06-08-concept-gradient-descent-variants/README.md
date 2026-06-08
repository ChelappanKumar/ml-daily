# Concept — SGD vs Adam vs AdamW

**Goal:** Write the three update rules side by side, name when each is preferred, and back the choice with a working comparison on a small noisy regression task.

## Summary

All three optimizers move parameters in the negative-gradient direction; what changes is **how the step size is computed**. SGD (with momentum) uses a single global learning rate plus an EMA of past gradients. Adam adapts the learning rate **per parameter** by dividing the momentum estimate by an EMA of squared gradients — so noisy or sparse parameters get smaller effective steps and stable parameters get bigger ones. AdamW is Adam with one specific fix: it **decouples weight decay** from the gradient calculation, applying it directly to the parameters. The fix matters in practice — vanilla Adam with `weight_decay > 0` does the wrong thing (decays parameters by an amount that interacts with the adaptive learning rate), and AdamW recovers SGD-with-L2-style regularization while keeping Adam's per-parameter step sizes.

## The three update rules, side by side

Let `g_t` = gradient at step `t`, `θ_t` = parameter, `lr` = learning rate, `λ` = weight decay.

**SGD with momentum:**
```
v_t = β · v_{t-1} + g_t
θ_t = θ_{t-1} − lr · v_t       # plus optional weight decay: θ_t -= lr · λ · θ_{t-1}
```

**Adam:**
```
m_t = β1 · m_{t-1} + (1 − β1) · g_t                # 1st moment (momentum)
v_t = β2 · v_{t-1} + (1 − β2) · g_t²              # 2nd moment (variance)
m̂_t = m_t / (1 − β1^t)                            # bias correction
v̂_t = v_t / (1 − β2^t)
θ_t = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)
```
With L2 regularization the bug is that `λ · θ` gets added to `g_t` *before* the adaptive scaling, so the effective decay strength varies per parameter.

**AdamW:**
```
# same m̂_t, v̂_t as Adam, but:
θ_t = θ_{t-1} − lr · ( m̂_t / (√v̂_t + ε)  +  λ · θ_{t-1} )
```
Weight decay is now an extra term applied to the parameter directly, NOT folded into the gradient. This is what the original Loshchilov & Hutter 2017 paper added.

## When each shines

| Optimizer | Sweet spot | Failure mode |
|---|---|---|
| **SGD + momentum** | Convex / well-conditioned problems, computer vision with ResNet-style architectures, when you have time to tune `lr`. Better final-test generalization on many benchmarks. | Slow to converge on sparse-gradient problems (recommendation, NLP). Sensitive to `lr` choice. |
| **Adam** | Default starting optimizer for transformers, RNNs, anything with sparse/noisy gradients. Robust to learning-rate choice. | With `weight_decay > 0`, the regularization is wrong (use AdamW). Sometimes generalizes worse than SGD on CNNs. |
| **AdamW** | Modern transformer training. The pretraining recipe for BERT, GPT-2 onwards, Llama all use AdamW. | Same speed/memory cost as Adam, no real downside. Just use this for new work. |

## Practical recipe

- **Default for transformers**: AdamW, `lr=3e-4` (small) / `5e-5` (fine-tune), `β1=0.9`, `β2=0.95-0.999`, `weight_decay=0.1` for pretraining / `0.01` for fine-tune.
- **Default for vision**: SGD + momentum=0.9, `lr=0.1` with cosine schedule, `weight_decay=5e-4`.
- **Universal**: warmup the LR for the first 1-5% of training, then cosine decay. Helps Adam-family especially — early gradients are noisy and the variance estimate `v_t` is bad.

## Worked example — see `optimizer_comparison.py`

Trains a 2-layer MLP on a noisy 1D regression task with each of {SGD, SGD+momentum, Adam, AdamW}. Plots loss curves on a shared chart. With weight decay deliberately set high (`1e-2`), Adam's regularization underperforms AdamW's clearly — the difference the original paper made concrete.

## Open questions to revisit

- How sensitive is AdamW's "better generalization" to the weight decay value? My intuition is that for `wd=0` AdamW and Adam are identical, and the gap opens as `wd` grows. Measure.
- Why does SGD-momentum often beat AdamW on ImageNet but lose on transformers? Common answers: BN interaction, sharper minima, scale invariance. Read the Smith & Le 2018 paper.
- For RLHF / preference fine-tuning, why does the community sometimes prefer plain SGD over AdamW? Suspect it's because the loss landscape is much flatter and momentum-only steps are easier to control.

## References

- Kingma & Ba 2014 — *Adam: A Method for Stochastic Optimization* — https://arxiv.org/abs/1412.6980
- Loshchilov & Hutter 2017 — *Decoupled Weight Decay Regularization* (AdamW) — https://arxiv.org/abs/1711.05101
- Wilson et al. 2017 — *The Marginal Value of Adaptive Gradient Methods in Machine Learning* (SGD beats Adam on vision) — https://arxiv.org/abs/1705.08292
- Smith & Le 2018 — *A Bayesian Perspective on Generalization and Stochastic Gradient Descent* — https://arxiv.org/abs/1710.06451
