# Polynomial Regression

> Curves from straight-line math: transform the inputs, keep the model linear in its weights.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 04 (ML Fundamentals: overfitting & regularization, bias-variance), Lesson 05.01 (Linear Regression: The Closed Form)
**Time:** ~45 minutes

## Learning Objectives

- Build a polynomial design matrix (Vandermonde matrix) and fit it with ordinary least squares
- Explain why polynomial regression is still linear regression
- Choose the polynomial degree with a validation set and recognize underfitting and overfitting from the error table
- Explain why high-degree polynomials need input scaling and why their coefficients explode

## The Problem

Your data is not a straight line. Fuel consumption vs speed is U-shaped, sales vs temperature rises and then falls, a sensor's response saturates. A linear model on `x` will miss the shape no matter how much data you give it: that is pure bias.

You could invent a new nonlinear algorithm. Or you could notice that a curve like `w0 + w1 x + w2 x^2 + w3 x^3` is still a weighted sum, just of different inputs. Create columns `x, x^2, x^3`, hand them to the OLS solver from Lesson 01, and you have a curve fitter. The hard part is no longer fitting: it is deciding how flexible the curve should be.

## The Concept

### Linear in the parameters

A model is "linear" in the regression sense when the prediction is a linear combination of the parameters:

```
y_hat = w0 * 1 + w1 * x + w2 * x^2 + ... + wd * x^d
```

The features are nonlinear in `x`, but every `w` enters linearly. So the loss is still a convex quadratic in `w`, and the normal equation still gives the exact minimum.

### The Vandermonde matrix

For degree `d` and inputs `x_1..x_n`:

```
    [1  x1  x1^2 ... x1^d]
X = [1  x2  x2^2 ... x2^d]
    [:   :    :        : ]
    [1  xn  xn^2 ... xn^d]
```

With several input features, "polynomial features" also includes interaction terms like `x1 * x2`. The number of columns grows as `C(p + d, d)`, so degree 3 on 20 features already gives 1771 columns.

### Degree is a capacity knob

```
degree 0-1        degree ~ truth          degree >> truth
underfit          good fit                overfit
high bias         balanced                high variance

  .  .  .            .  .  .                .  .  .
 ________          /‾‾‾\                   /\/\  /\
  .  .  .         /     \__               /    \/  \/
                   .  .  .                 .  .  .
```

Training error always goes down as you add degree (more columns can only help OLS fit the training set). Validation error goes down, flattens, then rises. Pick the degree at the bottom of the validation curve. This is the bias-variance tradeoff from Phase 04, made concrete with a single integer.

### Coefficient explosion and conditioning

When the polynomial wiggles through noisy points, it does so by setting huge positive and negative coefficients that almost cancel. That is a symptom of overfitting, and the next lesson (ridge) attacks it directly by penalizing large weights.

There is also a numeric problem. If `x` ranges up to 100, `x^12` is `10^24` while `x^0` is 1. The columns of `X` differ by 24 orders of magnitude and `X^T X` becomes numerically singular. Two cures:

- Rescale `x` to roughly `[-1, 1]` before taking powers (done here by dividing by 3).
- Standardize the generated columns (what the sklearn pipeline does).

### Beyond polynomials

Polynomials behave badly far from the data (they shoot off to infinity). Splines, which are piecewise low-degree polynomials joined smoothly, are often a better basis. The trick is the same: build columns, then run linear regression. sklearn's `SplineTransformer` drops into the same pipeline.

## Build It

### Step 1: A nonlinear truth

```python
def true_function(x):
    return np.sin(1.5 * x) + 0.3 * x


def make_data(n_samples, rng):
    x = rng.uniform(-3, 3, n_samples)
    y = true_function(x) + rng.normal(0, 0.3, n_samples)
    return x, y
```

You train on only 30 points and validate on 200. Small training sets make overfitting visible.

### Step 2: Polynomial features and OLS

```python
def polynomial_features(x, degree):
    return np.column_stack([x ** power for power in range(degree + 1)])


def fit_polynomial(x, y, degree):
    design = polynomial_features(x / 3.0, degree)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def predict_polynomial(x, coefficients):
    degree = len(coefficients) - 1
    return polynomial_features(x / 3.0, degree) @ coefficients
```

`polynomial_features(np.array([1.0, 2.0, 3.0]), 3)` prints the Vandermonde matrix with columns `1, x, x^2, x^3`. Dividing by 3 maps the input range to `[-1, 1]` so every power stays between -1 and 1.

### Step 3: Sweep the degree

```
degree  train MSE  valid MSE  max |coef|
     0     0.6894     0.7721         0.0
     1     0.4361     0.5228         0.9
     2     0.4357     0.5233         0.9
     3     0.1266     0.1697         3.3
     5     0.0543     0.0881        10.5
     7     0.0542     0.0882        10.3
     9     0.0514     0.1105        35.9
    12     0.0487     0.1417       437.7
    15     0.0458     0.1971      6822.9
```

Read it top to bottom:

- Degrees 0-2 underfit: both errors are high. Degree 2 adds nothing because the truth is roughly odd-symmetric (`sin` plus a line), so even powers barely help.
- Degree 5 is the sweet spot: validation MSE `0.088`, right at the noise variance `0.3^2 = 0.09`. You cannot do better than the noise.
- Beyond 9, training error keeps falling while validation error more than doubles, and the largest coefficient grows from 10 to 6800. That is overfitting.

### Step 4: Plot it

```python
figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for axis, degree in zip(axes, [1, best_degree, 15]):
    axis.scatter(train_x, train_y, s=15, color="black", label="train")
    axis.plot(grid, true_function(grid), "--", color="gray", label="truth")
    axis.plot(grid, predict_polynomial(grid, results[degree][0]), color="tab:blue", label=f"degree {degree}")
```

The script saves `polynomial_fits.png` next to itself. Degree 1 is a straight line through a wave. Degree 15 chases individual points and swings wildly near the edges, where there is little data to constrain it.

## Use It

In scikit-learn you compose a transformer and a regressor in a pipeline, so the same feature expansion is applied at fit and predict time.

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

model = make_pipeline(
    PolynomialFeatures(degree=best_degree, include_bias=False),
    StandardScaler(),
    LinearRegression(),
)
model.fit(train_x.reshape(-1, 1), train_y)
```

```
sklearn degree 5 valid MSE 0.0881  (ours 0.0881)
```

`include_bias=False` drops the ones column because `LinearRegression` fits its own intercept. sklearn expects a 2D feature array, hence `reshape(-1, 1)`. In practice you would pick the degree with `GridSearchCV` over `polynomialfeatures__degree`, using cross-validation instead of a single validation set.

## Exercises

1. **Easy.** Increase the training set from 30 to 300 points and rerun the sweep. What happens to the gap between train and valid MSE at degree 15? Why?
2. **Medium.** Remove the `/ 3.0` scaling and rerun with `x` in `[-30, 30]`. At which degree do the coefficients become garbage? Print `np.linalg.cond(design)` for each degree.
3. **Hard.** Replace the single validation set with 5-fold cross-validation written from scratch (Phase 04). Plot mean and standard deviation of CV error vs degree, and apply the "one standard error rule": pick the smallest degree whose CV error is within one standard error of the minimum.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Polynomial regression | "Curve fitting" | Linear regression on the expanded features `1, x, x^2, ..., x^d` |
| Linear in parameters | "Still linear" | The prediction is a linear combination of weights, even if features are nonlinear in `x` |
| Vandermonde matrix | "Polynomial design matrix" | Matrix whose columns are successive powers of the input |
| Basis expansion | "Feature engineering" | Replacing `x` with a set of functions of `x` and fitting a linear model on them |
| Degree | "How curvy" | Highest power used; controls model capacity |
| Underfitting | "Too simple" | Both training and validation error are high because the model cannot represent the pattern |
| Overfitting | "Memorizing" | Training error is low but validation error is high because the model fits noise |
| Interaction term | "Feature cross" | Product of two features, such as `x1 * x2`, included by `PolynomialFeatures` |

## Further Reading

- [An Introduction to Statistical Learning, Chapter 7](https://www.statlearning.com/) - polynomial regression, step functions and splines
- [scikit-learn: Polynomial and Spline Interpolation example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html) - polynomials vs `SplineTransformer`
- [scikit-learn: Underfitting vs. Overfitting example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html) - the same degree sweep with cross-validation
- [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) - the tradeoff that drives degree selection
