# Ridge Regression

> Add a small price on large weights and an unstable model becomes a stable one.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra: eigenvalues & eigenvectors), Phase 04 (ML Fundamentals: overfitting & regularization, cross-validation), Lessons 05.01-05.03
**Time:** ~60 minutes

## Learning Objectives

- Derive the ridge closed form `w = (X^T X + alpha I)^(-1) X^T y` and implement it in NumPy
- Explain why ridge stabilizes coefficients when features are collinear, using the SVD shrinkage factors `s^2 / (s^2 + alpha)`
- Explain why you center the data and leave the intercept unpenalized, and why you must standardize features first
- Select `alpha` with k-fold cross-validation and match scikit-learn's `Ridge` and `RidgeCV`

## The Problem

You are predicting a target from ten features. Two of them, `a` and `b`, are almost the same measurement (say, temperature from two sensors on the same wall). The truth is `y = 3a + 3b + noise`.

OLS gives `w_a = 1.49, w_b = 3.76`. Collect a different sample and you might get `w_a = 5.1, w_b = 0.9`. The sum `w_a + w_b` is well determined by the data, but how it splits between two near-copies is not, so OLS picks a split driven by noise. With more correlated features or fewer samples, this variance makes predictions on new data unreliable and makes any interpretation of individual coefficients meaningless.

Ridge regression fixes this with one extra term in the loss.

## The Concept

### The penalized loss

```
L(w, b) = ||y - X w - b||^2 + alpha * ||w||^2
```

The second term is the squared L2 norm of the weights. `alpha >= 0` sets the tradeoff: `alpha = 0` is OLS, `alpha -> infinity` drives every weight to 0 and predicts the mean.

### The closed form

Center `X` and `y` first (subtract column means). Then the intercept separates out and the gradient of the penalized loss is

```
-2 X^T (y - X w) + 2 alpha w = 0
=>  (X^T X + alpha I) w = X^T y
=>  w = (X^T X + alpha I)^(-1) X^T y
b = y_mean - x_mean . w
```

Adding `alpha I` to `X^T X` raises every eigenvalue by `alpha`. A nearly singular matrix (tiny eigenvalue) becomes comfortably invertible. This is the original motivation for ridge (Hoerl and Kennard, 1970).

### The intercept is not penalized

You want the model to be free to shift up or down; shrinking `b` toward zero would just bias every prediction toward 0. Centering makes this clean: on centered data the optimal intercept is 0, so you solve for `w` alone and recover `b` afterward.

### Standardize first

The penalty `alpha * sum(w_j^2)` treats all weights equally. If one feature is in meters and another in millimeters, the millimeter feature needs a weight 1000 times smaller for the same effect and gets penalized 10^6 times less. Standardize features so the penalty is fair.

### The SVD view: shrinkage by direction

Write the centered `X = U S V^T`. Then

```
w_ols   = V diag(1 / s_j)               U^T y
w_ridge = V diag(s_j / (s_j^2 + alpha)) U^T y
```

Compared with OLS, each principal direction `j` gets multiplied by

```
shrink_j = s_j^2 / (s_j^2 + alpha)
```

- Directions with large singular values (lots of spread in the data) get `shrink ~ 1`: they are trusted.
- Directions with tiny singular values (like `a - b` for two near-copies) get `shrink ~ 0`: the data says almost nothing there, so ridge does not guess.

```
direction:   a + b           noise_1 ... noise_8      a - b
s_j^2:       large           moderate                 tiny
shrink:      0.93            0.77 - 0.90              0.016
```

That is exactly why ridge splits weight evenly between `a` and `b`: their difference lives in the direction that gets zeroed.

### Bias-variance and the constraint view

Ridge trades a little bias (weights pulled toward 0) for a large drop in variance. Equivalently, it is OLS constrained to `||w||^2 <= t`: the solution is where the elliptical RSS contours first touch a circle around the origin. In the next lesson the circle becomes a diamond, which changes everything.

### Choosing alpha

`alpha` is a hyperparameter. Try a log-spaced grid (`10^-3` to `10^3`) and pick the value with the lowest k-fold cross-validated error.

## Build It

### Step 1: Collinear data

```python
def make_collinear_data(n_samples, rng):
    base = rng.normal(0, 1, n_samples)
    feature_a = base + rng.normal(0, 0.05, n_samples)
    feature_b = base + rng.normal(0, 0.05, n_samples)
    noise_features = rng.normal(0, 1, (n_samples, 8))
    features = np.column_stack([feature_a, feature_b, noise_features])
    targets = 3 * feature_a + 3 * feature_b + rng.normal(0, 1.0, n_samples)
    return features, targets
```

60 training rows, 500 test rows, all standardized with training statistics. The condition number of `X^T X` is about 790.

### Step 2: Ridge in closed form

```python
def fit_ridge(features, targets, alpha):
    n_features = features.shape[1]
    x_mean = features.mean(axis=0)
    y_mean = targets.mean()
    centered_x = features - x_mean
    centered_y = targets - y_mean
    weights = np.linalg.solve(centered_x.T @ centered_x + alpha * np.eye(n_features), centered_x.T @ centered_y)
    bias = y_mean - x_mean @ weights
    return weights, bias
```

```
   alpha      w_a      w_b    ||w||  test MSE
     0.0    1.487    3.756    4.049     1.126
     0.1    1.910    3.329    3.847     1.120
     1.0    2.435    2.761    3.691     1.133
    10.0    2.385    2.422    3.411     1.493
   100.0    1.405    1.408    2.019     8.731
  1000.0    0.280    0.280    0.407    27.600
```

As `alpha` grows, `w_a` and `w_b` converge toward each other, which is the stable, sensible answer. Test MSE barely moves for small `alpha` because the sum `w_a + w_b` was already right; the gain here is stable, interpretable coefficients. Too much `alpha` crushes the real signal and test MSE explodes: that is underfitting.

### Step 3: Verify the SVD formula

```python
def fit_ridge_svd(features, targets, alpha):
    x_mean = features.mean(axis=0)
    y_mean = targets.mean()
    u, singular_values, vt = np.linalg.svd(features - x_mean, full_matrices=False)
    shrink = singular_values / (singular_values ** 2 + alpha)
    weights = vt.T @ (shrink * (u.T @ (targets - y_mean)))
    return weights, y_mean - x_mean @ weights, singular_values
```

Both solutions agree. At `alpha = 10` the shrink factors are `0.93 ... 0.77` for the nine well-supported directions and `0.016` for the last one, the `a - b` direction.

### Step 4: Cross-validate alpha

```python
def cross_validate_alpha(features, targets, alphas, n_folds, rng):
    folds = kfold_indices(len(targets), n_folds, rng)
    scores = []
    for alpha in alphas:
        fold_errors = []
        for fold in folds:
            train_mask = np.ones(len(targets), dtype=bool)
            train_mask[fold] = False
            weights, bias = fit_ridge(features[train_mask], targets[train_mask], alpha)
            fold_errors.append(mse(features[fold], targets[fold], weights, bias))
        scores.append(np.mean(fold_errors))
    return np.array(scores)
```

Over 25 values from `10^-3` to `10^3`, 5-fold CV picks `alpha = 1.778`, with test MSE `1.151`.

## Use It

```python
from sklearn.linear_model import Ridge, RidgeCV

ridge = Ridge(alpha=10.0).fit(train_s, targets)
ridge_cv = RidgeCV(alphas=alphas).fit(train_s, targets)
```

```
Ridge(alpha=10) matches ours: True
RidgeCV picked alpha 1.778, test R^2 0.9655
```

sklearn's `Ridge` minimizes the same `||y - Xw||^2 + alpha ||w||^2` with an unpenalized intercept, so `alpha` means the same thing as in your code. `RidgeCV` by default uses an efficient leave-one-out formula instead of k-fold; here it lands on the same `alpha`. In a real project, put `StandardScaler` and `Ridge` in one `Pipeline` so scaling is fit inside each CV fold.

## Exercises

1. **Easy.** Plot each of the 10 coefficients as a function of `alpha` on a log x-axis (the "ridge trace"). Which ones shrink first?
2. **Medium.** Implement ridge with gradient descent (Lesson 02) by adding `2 * alpha * w` to the gradient. Confirm it matches `fit_ridge`. How does `alpha` change the largest stable learning rate?
3. **Hard.** Show numerically that ridge equals OLS on an augmented dataset: append `sqrt(alpha) * I` as extra rows to `X` and zeros to `y`. Then derive why this works.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Ridge regression | "L2 regularization" | Least squares plus `alpha * ||w||^2`, solved by `(X^T X + alpha I) w = X^T y` |
| Multicollinearity | "Correlated features" | Some features are nearly linear combinations of others, making `X^T X` ill-conditioned |
| Shrinkage | "Pulling weights to zero" | Ridge multiplies each principal direction by `s^2 / (s^2 + alpha)`, never exactly zero |
| Regularization strength | "Lambda" or "alpha" | Hyperparameter balancing data fit against weight size |
| Unpenalized intercept | "Don't shrink the bias" | The intercept is excluded from the penalty, implemented by centering the data |
| Ridge trace | "Coefficient path" | Plot of each coefficient as `alpha` varies |
| Effective degrees of freedom | "How complex the model really is" | `sum(s_j^2 / (s_j^2 + alpha))`, which falls from `p` to 0 as `alpha` grows |

## Further Reading

- [An Introduction to Statistical Learning, Chapter 6.2](https://www.statlearning.com/) - ridge and lasso side by side with the constraint geometry
- [The Elements of Statistical Learning, Section 3.4](https://hastie.su.domains/ElemStatLearn/) - the SVD derivation and effective degrees of freedom
- [StatQuest: Regularization Part 1, Ridge Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) - intuitive walkthrough
- [scikit-learn: Ridge regression and classification](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) - `Ridge`, `RidgeCV` and solver options
