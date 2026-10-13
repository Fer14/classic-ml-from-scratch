# Linear Regression: The Closed Form

> One matrix equation, solved once, gives you the best possible straight-line model. No loops, no learning rate.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra: matrix multiplication, linear systems & inverse), Phase 02 (Calculus and Optimization: partial derivatives & gradients), Phase 04 (ML Fundamentals: loss functions, train validation test)
**Time:** ~60 minutes

## Learning Objectives

- Write a linear model in matrix form `y_hat = X beta` with an intercept column
- Derive the normal equation `X^T X beta = X^T y` by setting the gradient of the squared error to zero
- Implement ordinary least squares in NumPy with `np.linalg.solve` and `np.linalg.lstsq`, and explain why you never call `inv`
- Interpret the fitted coefficients and evaluate the model on held-out data with R^2 and RMSE

## The Problem

You have 200 houses. For each one you know its size in square meters, its number of bedrooms and its age, plus the price it sold for. A new listing appears and you want a price estimate that is defensible: you want to be able to say "each extra square meter adds about 2.5k, each year of age removes about 0.8k".

A linear model gives you exactly that kind of answer. The question is how to choose the numbers. In Phase 04 you learned that training means minimizing a loss. For a linear model with squared error, you do not need to search for the minimum step by step: calculus and linear algebra hand it to you in closed form.

This lesson is the foundation of the whole phase. Gradient descent, polynomial features, ridge and lasso are all variations on the system of equations you will solve here.

## The Concept

### The model in matrix form

With `p` features, a linear model predicts

```
y_hat = b + w1*x1 + w2*x2 + ... + wp*xp
```

Stack your `n` samples into a matrix and prepend a column of ones. The ones column turns the intercept `b` into just another weight:

```
        [1  x11  x12 ... x1p]          [b ]
    X = [1  x21  x22 ... x2p]   beta = [w1]       y_hat = X @ beta
        [:   :    :       : ]          [: ]
        [1  xn1  xn2 ... xnp]          [wp]
```

`X` is `n x (p+1)`, `beta` is `(p+1)`, and `y_hat` is a vector of `n` predictions.

### The loss

Ordinary least squares (OLS) picks the `beta` that minimizes the residual sum of squares:

```
RSS(beta) = ||y - X beta||^2 = (y - X beta)^T (y - X beta)
```

This is a quadratic bowl in `beta` (convex, Phase 02). Its minimum is where the gradient is zero.

### Deriving the normal equation

Expand and differentiate:

```
RSS = y^T y - 2 beta^T X^T y + beta^T X^T X beta
dRSS/dbeta = -2 X^T y + 2 X^T X beta = 0
=>  X^T X beta = X^T y
```

That is the **normal equation**. If `X^T X` is invertible (no feature is an exact linear combination of the others and `n > p`), the unique solution is

```
beta = (X^T X)^(-1) X^T y
```

### The geometric picture

`X beta` can only reach vectors in the column space of `X`. OLS finds the point in that space closest to `y`, which is the orthogonal projection. The residual `r = y - X beta` is perpendicular to every column of `X`:

```
          y
          |\
          | \  r = y - X beta   (perpendicular to the plane)
          |  \
  --------+---*-----------  column space of X
              X beta
```

That is literally what `X^T r = 0` says, and it is the same equation as the normal equation rearranged. With an intercept column, one consequence is that residuals always sum to zero.

### Why not invert the matrix

The formula has an inverse in it, but in code you should not compute one. Explicit inverses are slower and lose precision when `X^T X` is ill-conditioned (features nearly collinear). Two better options:

- `np.linalg.solve(X.T @ X, X.T @ y)` solves the linear system directly with an LU factorization.
- `np.linalg.lstsq(X, y)` never forms `X^T X` at all. It uses an SVD of `X`, which squares the condition number less and also handles rank-deficient `X`.

Cost: forming `X^T X` is `O(n p^2)` and solving it is `O(p^3)`. That is instant for hundreds of features and becomes painful only at tens of thousands, which is where the next lesson's gradient descent takes over.

### Simple regression as a special case

With one feature the normal equation collapses to the formula you may have seen in a statistics course:

```
slope     = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)  = cov(x, y) / var(x)
intercept = y_mean - slope * x_mean
```

The fitted line always passes through `(x_mean, y_mean)`.

### Measuring the fit

```
R^2  = 1 - SS_res / SS_tot      SS_res = sum((y - y_hat)^2), SS_tot = sum((y - y_mean)^2)
RMSE = sqrt(mean((y - y_hat)^2))
```

R^2 is the fraction of variance explained, unitless. RMSE is in the units of `y` and tells you a typical error size. Always report them on data the model did not see.

## Build It

### Step 1: Synthetic houses with known truth

Generating data from known coefficients lets you check that the algorithm recovers them.

```python
def make_house_data(n_samples, rng):
    size_m2 = rng.uniform(40, 220, n_samples)
    bedrooms = rng.integers(1, 6, n_samples).astype(float)
    age_years = rng.uniform(0, 60, n_samples)
    noise = rng.normal(0, 15, n_samples)
    price_k = 30 + 2.5 * size_m2 + 12 * bedrooms - 0.8 * age_years + noise
    features = np.column_stack([size_m2, bedrooms, age_years])
    return features, price_k
```

You split the first 160 rows for training and the last 40 for testing.

### Step 2: One feature with the textbook formula

```python
def fit_simple_line(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return intercept, slope
```

Using size alone gives `price_k = 50.963 + 2.412 * size_m2`. The intercept is inflated because the bedrooms effect is being absorbed into it.

### Step 3: All features with the normal equation

```python
def add_intercept_column(features):
    ones = np.ones((features.shape[0], 1))
    return np.hstack([ones, features])


def fit_normal_equation(features, targets):
    design = add_intercept_column(features)
    gram = design.T @ design
    moment = design.T @ targets
    return np.linalg.solve(gram, moment)
```

Output:

```
 intercept: learned   31.799   true  30.00
   size_m2: learned    2.479   true   2.50
  bedrooms: learned   11.589   true  12.00
 age_years: learned   -0.779   true  -0.80
```

The estimates are close to the truth but not exact: with only 160 noisy samples, each coefficient has sampling error. Lesson 06 shows how to quantify it.

### Step 4: The safer solver

```python
def fit_lstsq(features, targets):
    design = add_intercept_column(features)
    coefficients, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    return coefficients
```

On this well-conditioned problem both agree to `3e-13`. On collinear data they would not, and `lstsq` is the one to trust.

### Step 5: Evaluate and check orthogonality

```python
def predict(features, coefficients):
    return add_intercept_column(features) @ coefficients


def r_squared(targets, predictions):
    residual_sum = np.sum((targets - predictions) ** 2)
    total_sum = np.sum((targets - targets.mean()) ** 2)
    return 1 - residual_sum / total_sum
```

Test R^2 is `0.9876` and test RMSE is `16.16`, close to the noise level of 15. That is the best you can hope for: the remaining error is noise no model can predict. Finally, `design.T @ residuals` prints `[0, 0, 0, 0]`, confirming the projection picture.

## Use It

scikit-learn's `LinearRegression` solves the same least squares problem (via `scipy.linalg.lstsq`) and handles the intercept for you, so you pass raw features without a ones column.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(train_x, train_y)
print(model.intercept_, model.coef_)
print(model.score(test_x, test_y))
same = np.allclose(np.r_[model.intercept_, model.coef_], beta)
```

The output is `matches from-scratch solution: True`. `score` returns R^2. For p-values and confidence intervals on coefficients you would reach for `statsmodels`, but Lesson 06 builds those by hand.

## Exercises

1. **Easy.** Add a fourth feature that is pure noise, `rng.normal(0, 1, n)`. Refit. How large is its coefficient, and does test RMSE improve or get worse?
2. **Medium.** Add a feature `size_m2 * 2 + 1` (an exact linear function of another column). What does `np.linalg.solve` do? What does `np.linalg.lstsq` return, and what is its reported rank?
3. **Hard.** Implement OLS with a QR decomposition: `Q, R = np.linalg.qr(X)` then solve `R beta = Q^T y` by back substitution written by hand. Compare the result and the condition number of `R` against `X^T X`.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Ordinary least squares | "Fit a line" | Choose `beta` that minimizes the sum of squared residuals `||y - X beta||^2` |
| Design matrix | "The X matrix" | The `n x (p+1)` matrix of features with a leading column of ones for the intercept |
| Normal equation | "The closed form" | `X^T X beta = X^T y`, obtained by setting the gradient of RSS to zero |
| Residual | "The error" | `y_i - y_hat_i` for one sample; OLS makes the residual vector orthogonal to every column of `X` |
| Projection | "The geometry of OLS" | `X beta` is the closest point to `y` within the column space of `X` |
| Condition number | "How unstable the solve is" | Ratio of largest to smallest singular value; large values amplify rounding and noise |
| R^2 | "Variance explained" | `1 - SS_res / SS_tot`; 1 is perfect, 0 equals predicting the mean |
| RMSE | "Typical error" | Square root of mean squared error, in the units of the target |

## Further Reading

- [An Introduction to Statistical Learning, Chapter 3](https://www.statlearning.com/) - linear regression with a focus on interpretation and inference
- [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf) - derives the normal equation with matrix calculus
- [3Blue1Brown, Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - the column space and projection intuition behind least squares
- [scikit-learn: Ordinary Least Squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) - API and complexity notes
