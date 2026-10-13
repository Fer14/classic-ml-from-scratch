# Interpretability

> A model you cannot explain is a model you cannot debug, defend or trust. Ask it questions from the outside.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 07 (Trees and Ensembles: Gradient Boosting), Phase 04 (ML Fundamentals: Metrics), Phase 03 (Probability and Statistics: Expectation & Variance)
**Time:** ~75 minutes

## Learning Objectives

- Implement permutation importance and explain what "drop in score when a feature is shuffled" measures
- Implement partial dependence and ICE curves and explain why an averaged curve can hide interactions
- Compute exact Shapley values by enumerating all feature coalitions and verify the efficiency property
- Use `permutation_importance` and `partial_dependence` from `sklearn.inspection` and know what the `shap` library adds

## The Problem

Your gradient boosting model scores R^2 = 0.95. A stakeholder asks three questions:

1. Which features does the model rely on? (global importance)
2. How does the prediction change as feature X goes up? (global effect)
3. Why did it predict 7.9 for this particular row? (local explanation)

The model is 300 trees deep with no coefficients to read. Impurity-based `feature_importances_` exist for trees, but they are computed on training data and are biased toward high-cardinality features. You need model-agnostic tools that only use `predict`.

In this lesson the ground truth is known, so you can check every answer:

```
y = 2*linear + 2*sin(2*sine) + 1.5*inter_a*inter_b + noise_term
```

The feature called `noise` is not used at all.

## The Concept

### Permutation importance

Shuffle one column in the test set. This breaks its relationship with y while keeping its distribution. If the score drops a lot, the model relied on that feature.

```
importance_j = score(X_test, y_test) - score(X_test with column j shuffled, y_test)
```

Repeat the shuffle several times and report mean and standard deviation. Compute it on held-out data: on training data, an overfit model looks like it depends on noise.

Caveat: when two features are strongly correlated, shuffling one creates unrealistic rows and the model can partially use the other one, so importance is split or distorted.

### Partial dependence and ICE

Partial dependence answers "what does the model predict, on average, if I set feature j to value v for everybody?"

```
PD_j(v) = (1/n) * sum_i f(x_i with x_ij replaced by v)
```

An ICE (individual conditional expectation) curve is the same thing for a single row, without averaging. PD is the average of the ICE curves.

```
 prediction                      prediction
     |        /  ICE (inter_b>1)     |
     |      /                        |
     |----/------- PD (flat!)        |  PD of a purely additive
     |  /  \                         |  feature: all ICE curves
     |/      \  ICE (inter_b<-1)     |  are parallel
     +-------------> inter_a         +-------------> x
```

For `inter_a`, the effect is +slope when `inter_b > 0` and -slope when `inter_b < 0`. On average they cancel and the PD is flat, but the feature is clearly important. Always look at ICE curves before concluding a feature does nothing.

### Shapley values

From cooperative game theory: the features are players, the prediction is the payout. The Shapley value of feature j is its average marginal contribution over all orders in which features could be added:

```
phi_j = sum over S not containing j of
        |S|! (p - |S| - 1)! / p!  *  [ v(S + {j}) - v(S) ]
```

The value of a coalition S is the model's expected prediction when the features in S are fixed to the row's values and the others are drawn from a background dataset:

```
v(S) = mean over background rows b of f(x_S, b_notS)
```

Shapley values are the unique attribution satisfying efficiency, symmetry, dummy and additivity. Efficiency is the one you can check:

```
v(empty) + sum_j phi_j = f(x)
```

The cost is 2^p coalitions, fine for p = 5 (32 coalitions), impossible for p = 50. The `shap` library approximates it (KernelSHAP) or computes it exactly in polynomial time for trees (TreeSHAP).

For a linear model with independent features, Shapley values have a closed form: `phi_j = w_j * (x_j - mean(x_j))`. That is your correctness test.

### The key insight

These tools explain the model, not the world. A feature can be important to the model because it is a proxy for something else, and a feature can matter in reality while the model ignores it. Interpretability tells you what the model learned; causality needs different tools.

## Build It

### Step 1: Permutation importance

```python
def permutation_importances(predict, X, y, n_repeats=10, seed=0):
    rng = np.random.default_rng(seed)
    baseline = r2(y, predict(X))
    importances = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for repeat in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, j] = rng.permutation(X_permuted[:, j])
            importances[j, repeat] = baseline - r2(y, predict(X_permuted))
    return importances.mean(axis=1), importances.std(axis=1)
```

```
  linear   drop in R^2 = 0.905 +/- 0.031
  inter_b  drop in R^2 = 0.581 +/- 0.024
  inter_a  drop in R^2 = 0.572 +/- 0.032
  sine     drop in R^2 = 0.301 +/- 0.024
  noise    drop in R^2 = -0.001 +/- 0.001
```

The unused feature is correctly at zero, and both interaction features are important.

### Step 2: Partial dependence and ICE

```python
def partial_dependence_curve(predict, X, feature, grid):
    averages = []
    for value in grid:
        X_fixed = X.copy()
        X_fixed[:, feature] = value
        averages.append(predict(X_fixed).mean())
    return np.array(averages)
```

`ice_curves` is the same loop without the `.mean()`.

```
  grid:       -1.80  -1.20  -0.60  -0.00  +0.60  +1.20  +1.80
  linear      -3.56  -2.64  -1.17  +0.31  +1.13  +2.23  +3.37
  sine        +0.99  -1.41  -1.82  -0.30  +1.82  +1.34  -0.70
  inter_a     -0.05  +0.08  +0.01  +0.02  +0.01  -0.10  -0.26
  noise       -0.06  +0.01  +0.04  -0.01  -0.04  +0.00  -0.08
    ICE slopes of inter_a when inter_b < -1: [-1.21 -1.8  -1.75]
    ICE slopes of inter_a when inter_b > +1: [2.34 2.05 2.1 ]
```

`linear` recovers a slope near 2, `sine` traces the sine wave, and `inter_a` looks flat on average while the ICE curves show opposite slopes. Permutation importance said `inter_a` matters; the PD alone would have said it does not.

### Step 3: Exact Shapley values

```python
def coalition_value(predict, x, background, coalition):
    X_mixed = background.copy()
    columns = list(coalition)
    X_mixed[:, columns] = x[columns]
    return predict(X_mixed).mean()


def exact_shapley_values(predict, x, background):
    n_features = len(x)
    values = {}
    for size in range(n_features + 1):
        for coalition in itertools.combinations(range(n_features), size):
            values[coalition] = coalition_value(predict, x, background, coalition)
    phi = np.zeros(n_features)
    for j in range(n_features):
        others = [k for k in range(n_features) if k != j]
        for size in range(n_features):
            weight = math.factorial(size) * math.factorial(n_features - size - 1) / math.factorial(n_features)
            for coalition in itertools.combinations(others, size):
                with_j = tuple(sorted(coalition + (j,)))
                phi[j] += weight * (values[with_j] - values[coalition])
    return phi, values[()]
```

Each coalition value is computed once and cached, so the cost is 32 model calls on 200 background rows.

```
  explaining x = [1.5 0.8 1.5 1.5 0. ]
  base value E[f(X)] = +0.205
  linear   phi = +2.691
  sine     phi = +1.788
  inter_a  phi = +1.363
  inter_b  phi = +1.905
  noise    phi = -0.035
  base + sum(phi) = +7.916   model prediction = +7.916
```

Efficiency holds exactly. The interaction term `1.5 * 1.5 * 1.5 = 3.4` is split between `inter_a` and `inter_b` (about 3.3 together); the split is not perfectly equal because the boosted model is only an approximation of the symmetric true function.

## Use It

```python
from sklearn.inspection import partial_dependence, permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, scoring="r2")
pd_result = partial_dependence(model, X_train, features=[1], method="brute", custom_values={1: grid})
```

```
  permutation_importance linear   0.914 (scratch 0.905)
  permutation_importance inter_a  0.579 (scratch 0.572)
  partial_dependence(sine) max abs diff vs scratch: 0.00e+00
  linear model: exact Shapley  = +2.814 +0.338 -0.130 +0.050 +0.001
  linear model: w*(x - mean)   = +2.814 +0.338 -0.130 +0.050 +0.001
```

Permutation importances differ only by shuffle randomness; partial dependence matches exactly; and your Shapley code reproduces the linear-model closed form. `PartialDependenceDisplay.from_estimator(model, X, features, kind="both")` plots PD with ICE curves overlaid.

For Shapley values in practice, the `shap` package (not installed in this course) offers `shap.TreeExplainer(model).shap_values(X)` for tree ensembles, XGBoost and LightGBM, computing exact values in polynomial time, and `shap.KernelExplainer(predict, background)` for any model by sampling coalitions. Its plots (waterfall, beeswarm, dependence) are built on exactly the quantities you computed.

## Exercises

1. **Easy.** Compute permutation importance on the training set instead of the test set. Then train a deliberately overfit model (`max_leaf_nodes=None, min_samples_leaf=1`) and compare. Does `noise` gain importance on training data?
2. **Medium.** Add a sixth feature that is a noisy copy of `linear` (`X[:, 0] + 0.1 * noise`). Retrain and recompute permutation importance and Shapley values. How is credit shared between the two copies?
3. **Hard.** Implement a sampling approximation: for M random permutations of the features, add features one at a time and record each one's marginal contribution. Show that with M = 200 your estimate is within 0.05 of the exact values, and count model calls versus the exact method as p grows.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Permutation importance | "Shuffle and see what breaks" | Drop in a held-out score when one feature's column is randomly permuted |
| Partial dependence | "The average effect of a feature" | Mean model prediction when a feature is set to a fixed value for all rows |
| ICE curve | "Partial dependence per row" | The prediction for a single row as one feature is varied; PD is their average |
| Shapley value | "Fair credit for each feature" | Average marginal contribution of a feature over all coalitions, weighted by |S|!(p-|S|-1)!/p! |
| Efficiency | "SHAP values add up" | Base value plus the sum of Shapley values equals the model's prediction for that row |
| Background data | "The reference distribution" | Rows used to fill in features outside a coalition; defines the base value |
| Model-agnostic | "Works for any model" | Uses only predictions, never the model's internals |

## Further Reading

- [Christoph Molnar, Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) -- free book with chapters on PDP, ICE, permutation importance and SHAP
- [Lundberg and Lee, A Unified Approach to Interpreting Model Predictions (NeurIPS 2017)](https://arxiv.org/abs/1705.07874) -- the SHAP paper
- [scikit-learn: Inspection](https://scikit-learn.org/stable/inspection.html) -- permutation importance and partial dependence, including the pitfalls of impurity-based importances
- [Breiman, Random Forests (Machine Learning 2001)](https://link.springer.com/article/10.1023/A:1010933404324) -- introduced permutation-based variable importance
