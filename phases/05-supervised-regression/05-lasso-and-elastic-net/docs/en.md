# Lasso and Elastic Net

> Ridge shrinks every weight. Lasso shrinks some of them all the way to zero, and in doing so picks your features for you.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 02 (Calculus and Optimization: convexity), Phase 04 (ML Fundamentals: overfitting & regularization, cross-validation), Lesson 05.04 (Ridge Regression)
**Time:** ~75 minutes

## Learning Objectives

- Write the lasso and elastic net objectives and explain why the L1 penalty produces exactly-zero weights
- Derive the soft-thresholding operator and implement lasso and elastic net with cyclic coordinate descent
- Trace the lasso path as `alpha` varies and compute `alpha_max`, the smallest value that zeros every weight
- Explain why elastic net handles groups of correlated features better than lasso, and match scikit-learn's `Lasso`, `ElasticNet` and `LassoCV`

## The Problem

You have 80 samples and 30 candidate features, but only 5 of them actually influence the target. OLS uses all 30 and fits noise with the other 25. Ridge shrinks those 25 toward zero but never to zero, so the model still depends on every input: you still need to collect all 30 measurements in production and you cannot tell anyone which 5 matter.

You want a method that fits and selects at the same time. Lasso (Least Absolute Shrinkage and Selection Operator, Tibshirani 1996) does exactly this by swapping the squared penalty for an absolute value.

## The Concept

### The objectives (scikit-learn's scaling)

```
lasso:        (1/(2n)) ||y - X w - b||^2 + alpha * sum(|w_j|)

elastic net:  (1/(2n)) ||y - X w - b||^2 + alpha * l1_ratio * sum(|w_j|)
                                         + (alpha/2) * (1 - l1_ratio) * sum(w_j^2)
```

Note the `1/(2n)`: unlike ridge in sklearn, the lasso loss is averaged over samples, so lasso `alpha` values are on a different scale from ridge ones. `l1_ratio = 1` is pure lasso, `l1_ratio = 0` is pure ridge.

### Why L1 gives exact zeros

Picture the constrained form `||w||_1 <= t` in two dimensions. The L1 ball is a diamond with corners on the axes; the L2 ball is a circle.

```
        L1 (lasso)                    L2 (ridge)
            w2                            w2
            |   RSS contours              |   RSS contours
           /\    ( ( o ) )               _|_    ( ( o ) )
          /  \  /                       / | \  /
     ----<----*---- w1             ----|--+--*-- w1
          \  /                          \_|_/
           \/                             |
    first contact at a corner      first contact anywhere on the circle
    -> w2 = 0                      -> both nonzero
```

The expanding RSS ellipse is most likely to first touch the diamond at a corner, where some coordinates are exactly zero. The circle has no corners.

The same thing from calculus: the slope of `alpha * |w|` is `+-alpha` no matter how small `w` is, so there is a constant force toward zero that wins whenever the data's pull on `w_j` is weaker than `alpha`. The slope of `alpha * w^2` is `2 alpha w`, which vanishes near zero, so ridge never quite gets there.

### Coordinate descent and soft-thresholding

`|w|` is not differentiable at 0, so there is no normal equation. But if you fix every weight except `w_j`, the problem in `w_j` alone is a one-dimensional quadratic plus `alpha |w_j|`, which has a closed-form solution.

Let `r` be the residual with feature `j`'s contribution added back, `rho_j = x_j . r / n`, and `z_j = ||x_j||^2 / n`. Then

```
w_j = S(rho_j, alpha * l1_ratio) / (z_j + alpha * (1 - l1_ratio))

S(v, t) = sign(v) * max(|v| - t, 0)        soft-thresholding
```

```
S(v, t)
   |        /
   |       /
---+--|--|/------ v      flat zero on [-t, t], then shifted by t
     -t   t
```

Cycle through `j = 1..p`, update each weight, keep the residual up to date incrementally, and repeat until no weight changes more than a tolerance. Each sweep is `O(np)`. Because the objective is convex, this converges to the global minimum.

### The lasso path and alpha_max

For `alpha` large enough every weight is zero. The threshold is

```
alpha_max = max_j |x_j . (y - y_mean)| / n          (x_j centered)
```

Below it, features enter the model one by one as `alpha` decreases. Plotting coefficients against `alpha` gives the lasso path. Solvers compute the whole path cheaply by warm-starting each `alpha` from the previous solution.

### Where lasso struggles: correlated groups

If three features are near-copies, lasso tends to pick one arbitrarily and zero the others. Which one it picks can change with a tiny perturbation of the data. Elastic net adds a bit of ridge, which makes the objective strictly convex and spreads the weight across the group (the grouping effect). It is also more stable when `p > n`, where pure lasso can select at most `n` features.

## Build It

### Step 1: Soft-thresholding

```python
def soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)
```

```
input            [-2.  -0.5  0.   0.3  1.2]
S(input, 0.5)    [-1.5 -0.   0.   0.   0.7]
```

Small values are snapped to zero; large ones are pulled toward zero by exactly `0.5`.

### Step 2: Coordinate descent for elastic net

```python
def elastic_net_coordinate_descent(features, targets, alpha, l1_ratio, max_iter=1000, tol=1e-8):
    n_samples, n_features = features.shape
    x_mean = features.mean(axis=0)
    y_mean = targets.mean()
    centered_x = features - x_mean
    centered_y = targets - y_mean
    column_norms = (centered_x ** 2).sum(axis=0) / n_samples
    weights = np.zeros(n_features)
    residual = centered_y.copy()
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1 - l1_ratio)
    for _ in range(max_iter):
        largest_change = 0.0
        for j in range(n_features):
            old_weight = weights[j]
            residual += centered_x[:, j] * old_weight
            correlation = centered_x[:, j] @ residual / n_samples
            weights[j] = soft_threshold(correlation, l1_penalty) / (column_norms[j] + l2_penalty)
            residual -= centered_x[:, j] * weights[j]
            largest_change = max(largest_change, abs(weights[j] - old_weight))
        if largest_change < tol:
            break
    bias = y_mean - x_mean @ weights
    return weights, bias
```

Lasso is the special case `l1_ratio = 1.0`. As with ridge, centering keeps the intercept out of the penalty.

### Step 3: Walk the lasso path

The data: 30 standard-normal features, true weights `[4, -3, 2, 1.5, -1, 0, ..., 0]`, noise std 1.

```
 alpha  nonzero  test MSE  first 6 weights
 0.001       30     1.711  [ 3.93 -3.22  1.75  1.36 -1.11 -0.03]
  0.05       18     1.339  [ 3.87 -3.08  1.7   1.31 -1.01  0.  ]
   0.1       14     1.270  [ 3.81 -3.    1.66  1.28 -0.96  0.  ]
   0.3        6     1.849  [ 3.59 -2.79  1.43  1.1  -0.74  0.  ]
   0.6        5     4.034  [ 3.25 -2.45  1.02  0.76 -0.35  0.  ]
   1.0        4     8.637  [ 2.82 -1.98  0.47  0.35 -0.    0.  ]
   2.0        2    17.647  [ 1.88 -0.9   0.    0.   -0.   -0.  ]

alpha_max (every weight is zero above this): 3.959
```

Near-zero `alpha` is almost OLS: 30 nonzero weights and the worst test error among small alphas. Around `alpha = 0.1` you get the best test error with 14 features. By `alpha = 0.6` exactly the 5 true features survive, but their weights are shrunk too much (the L1 bias), so test error rises. Prediction-optimal and selection-optimal `alpha` are often not the same.

### Step 4: Correlated twins

```python
base = twin_rng.normal(0, 1, 200)
twins = np.column_stack([base + twin_rng.normal(0, 0.01, 200) for _ in range(3)])
twin_targets = 3 * base + twin_rng.normal(0, 0.5, 200)
```

```
lasso       weights on 3 near-copies: [2.952 0.    0.   ]
elastic net weights on 3 near-copies: [0.991 0.983 0.98 ]
```

Lasso puts everything on the first copy. Elastic net with `l1_ratio = 0.3` shares the effect across the group.

## Use It

```python
from sklearn.linear_model import ElasticNet, Lasso, LassoCV

sk_lasso = Lasso(alpha=0.1, tol=1e-10, max_iter=100000).fit(features, targets)
sk_enet = ElasticNet(alpha=0.1, l1_ratio=0.5, tol=1e-10, max_iter=100000).fit(features, targets)
lasso_cv = LassoCV(cv=5, random_state=0).fit(features, targets)
```

```
Lasso(alpha=0.1) max |ours - sklearn| = 5.60e-09
ElasticNet(0.1, 0.5) max |ours - sklearn| = 5.01e-09
LassoCV alpha 0.0455, selected features [0, 1, 2, 3, 4, 6, 7, ...]
LassoCV test R^2 0.9577
```

sklearn uses the same coordinate descent, so the coefficients match to 1e-8. `LassoCV` computes the full path with warm starts and picks the `alpha` with the lowest CV error. Notice it keeps 19 features: CV optimizes prediction, and a few noise features with tiny weights cost almost nothing. If you want a sparser model, pick a larger `alpha` from `lasso_cv.alphas_` (the one-standard-error rule), or use `ElasticNetCV` to tune `l1_ratio` too. As with ridge, standardize features first; here they already are.

## Exercises

1. **Easy.** Rerun Step 3 with 300 training samples instead of 80. At `alpha = 0.1`, how many noise features survive now? Explain.
2. **Medium.** Implement a path solver: compute `alpha_max`, build 50 log-spaced alphas down to `alpha_max / 1000`, and warm-start each fit from the previous weights. Plot the path and compare total runtime with cold starts.
3. **Hard.** Implement the "relaxed lasso": use lasso only to select features, then refit plain OLS on the selected set. Compare test MSE with lasso at `alpha = 0.6`, where selection is perfect but weights are over-shrunk.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Lasso | "L1 regularization" | Least squares plus `alpha * sum(|w_j|)`, which sets some weights exactly to zero |
| Elastic net | "L1 plus L2" | A weighted mix of lasso and ridge penalties controlled by `l1_ratio` |
| Sparsity | "Feature selection" | Most weights are exactly zero, so the model uses only a subset of features |
| Soft-thresholding | "Shrink and snap" | `S(v, t) = sign(v) max(|v| - t, 0)`, the exact minimizer of the 1D L1 problem |
| Coordinate descent | "One weight at a time" | Optimizes each coordinate in turn with the others fixed, cycling until convergence |
| Lasso path | "Regularization path" | The coefficients as a function of `alpha`, from all zero at `alpha_max` down to OLS |
| alpha_max | "Where everything is zero" | `max_j |x_j . y_centered| / n`, the smallest penalty that zeros every weight |
| Grouping effect | "Elastic net shares weight" | Correlated features receive similar coefficients under elastic net instead of one being picked |

## Further Reading

- [An Introduction to Statistical Learning, Chapter 6.2](https://www.statlearning.com/) - lasso geometry and comparison with ridge
- [Hastie, Tibshirani, Wainwright: Statistical Learning with Sparsity](https://hastie.su.domains/StatLearnSparsity/) - free book, coordinate descent and the lasso path in depth
- [StatQuest: Regularization Part 2, Lasso Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) - why lasso can reach zero and ridge cannot
- [scikit-learn: Lasso and Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html#lasso) - objectives, `LassoCV`, `ElasticNetCV`, and path functions
