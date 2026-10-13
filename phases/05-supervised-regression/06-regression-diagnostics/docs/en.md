# Regression Diagnostics

> A high R^2 does not mean your model is right. The residuals tell you whether to trust it.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 03 (Probability and Statistics: hypothesis testing), Lesson 05.01 (Linear Regression: The Closed Form), Lesson 05.04 (Ridge Regression)
**Time:** ~60 minutes

## Learning Objectives

- Compute R^2 and adjusted R^2 and explain why adding useless features always raises R^2 but not adjusted R^2
- Compute coefficient standard errors, t-statistics and p-values for OLS from the matrix `sigma^2 (X^T X)^(-1)`
- Detect influential points with leverage (hat matrix diagonal) and Cook's distance, and multicollinearity with the variance inflation factor (VIF)
- Read the three standard diagnostic plots: residuals vs fitted, normal Q-Q, residuals vs leverage

## The Problem

You fit a house-price model and get R^2 = 0.94. Good news? Look at the coefficients: each square meter is worth 1.08k and each extra room is worth 29.6k. The data was generated with 2.0k per square meter and 5k per room. The model fits well and is badly wrong.

The cause is a single mistyped row: a 190 m^2 house recorded at 120k. One point out of 121 has dragged the size coefficient in half and pushed its effect onto the correlated `rooms` column. Nothing in `model.score()` tells you that. Diagnostics do.

Whenever you will interpret coefficients, report uncertainty, or deploy a linear model where errors cost money, you run these checks.

## The Concept

### Assumptions behind OLS inference

The OLS coefficients are always the least squares fit. But the standard errors and p-values you compute from them assume:

1. **Linearity:** `E[y | x] = X beta`.
2. **Independence:** errors of different samples are independent.
3. **Homoscedasticity:** errors have constant variance `sigma^2`.
4. **Normality:** errors are normally distributed (matters mostly for small `n`).

Plus two practical concerns that are not assumptions but break things anyway: influential outliers and multicollinearity.

### R^2 and adjusted R^2

```
R^2     = 1 - SS_res / SS_tot
adj R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
```

R^2 can only go up when you add a column, even a column of random noise. Adjusted R^2 charges for each predictor and goes down if the new feature does not pull its weight. It is a quick sanity check, not a replacement for validation error.

### Uncertainty of the coefficients

Under the assumptions above, the OLS estimator has covariance

```
Cov(beta_hat) = sigma^2 (X^T X)^(-1)
sigma^2_hat  = SS_res / (n - p - 1)
SE(beta_j)   = sqrt(sigma^2_hat * [(X^T X)^(-1)]_jj)
t_j          = beta_j / SE(beta_j)        ~ t distribution with n - p - 1 df
```

The p-value tests `H0: beta_j = 0` (Phase 03). Note the `(X^T X)^(-1)` term: collinear features make it large, which inflates standard errors.

### Leverage: unusual inputs

The hat matrix `H = X (X^T X)^(-1) X^T` maps `y` to `y_hat = H y`. Its diagonal `h_ii` is the leverage of sample `i`: how much its own target pulls its own prediction.

```
0 <= h_ii <= 1        sum(h_ii) = p + 1        average = (p + 1) / n
```

High leverage means the input `x_i` is far from the bulk of the data (here, a big house with few rooms). High leverage alone is not bad: it is a point with the power to move the fit.

### Cook's distance: influence

Influence combines leverage with a large residual:

```
D_i = (e_i^2 / ((p + 1) * sigma^2_hat)) * h_ii / (1 - h_ii)^2
```

It measures how much all fitted values change if you delete point `i`. Common flags: `D_i > 4/n` (look at it), `D_i > 1` (it is driving the model).

```
                 low residual        high residual
low leverage     fine                outlier in y, little influence
high leverage    fine, good anchor   INFLUENTIAL: check it
```

### Variance inflation factor

Regress feature `j` on all the other features and get `R^2_j`. Then

```
VIF_j = 1 / (1 - R^2_j)
```

`VIF = 1` means feature `j` is uncorrelated with the others. `VIF = 10` means its coefficient variance is 10 times what it would be without collinearity. Rules of thumb: above 5, take note; above 10, act (drop, combine, or use ridge from Lesson 04).

### The three plots

- **Residuals vs fitted:** you want a shapeless horizontal band. A curve means missing nonlinearity; a funnel means heteroscedasticity.
- **Normal Q-Q:** sorted residuals vs normal quantiles. Points on the line mean normal errors; bent tails mean heavy tails or outliers.
- **Residuals vs leverage:** points in the far right with large residuals are the influential ones.

## Build It

### Step 1: Data with a planted bad row

```python
def make_data(n_samples, rng):
    size_m2 = rng.uniform(40, 200, n_samples)
    rooms = np.round(size_m2 / 30 + rng.normal(0, 0.4, n_samples))
    age_years = rng.uniform(0, 50, n_samples)
    price_k = 20 + 2.0 * size_m2 + 5 * rooms - 0.6 * age_years + rng.normal(0, 12, n_samples)
    features = np.column_stack([size_m2, rooms, age_years])
    features = np.vstack([features, [[190.0, 3.0, 5.0]]])
    price_k = np.append(price_k, 120.0)
    return features, price_k
```

`rooms` is derived from `size_m2`, so they are correlated. Row 120 is a large house with too few rooms (high leverage) and an absurdly low price (large residual).

### Step 2: R^2 vs adjusted R^2

```python
def r2_and_adjusted(targets, predictions, n_predictors):
    n_samples = len(targets)
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
    adjusted = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1)
    return r2, adjusted
```

```
R^2 0.9438   adjusted R^2 0.9424
with 10 pure-noise features: R^2 0.9484   adjusted R^2 0.9421
```

Ten random columns raised R^2 and lowered adjusted R^2.

### Step 3: The coefficient table

```python
def coefficient_table(features, targets, beta):
    design = design_matrix(features)
    n_samples, n_params = design.shape
    residuals = targets - design @ beta
    sigma2 = residuals @ residuals / (n_samples - n_params)
    standard_errors = np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
    t_values = beta / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), n_samples - n_params)
    return standard_errors, t_values, p_values
```

```
      term     coef  std err       t        p
 intercept   18.676    6.896    2.71   0.0078
   size_m2    1.084    0.140    7.74   0.0000
     rooms   29.600    3.930    7.53   0.0000
 age_years   -0.137    0.152   -0.90   0.3695
```

Everything looks "significant", yet `size_m2` and `rooms` are far from the truth and `age_years`, a real effect, now looks like noise. Small p-values do not mean correct coefficients when the assumptions fail.

### Step 4: Residual checks

```
mean residual -1.20e-13
Shapiro-Wilk normality test p = 0.0000
residual std, low vs high predictions: 16.93 vs 29.88
```

Mean zero is guaranteed by the intercept and tells you nothing. Normality is rejected and the spread is uneven: both are symptoms of the one huge residual.

### Step 5: Leverage and Cook's distance

```python
def leverage(features):
    design = design_matrix(features)
    hat_matrix = design @ np.linalg.inv(design.T @ design) @ design.T
    return np.diag(hat_matrix)


def cooks_distance(residuals, leverages, n_params):
    sigma2 = np.sum(residuals ** 2) / (len(residuals) - n_params)
    return residuals ** 2 / (n_params * sigma2) * leverages / (1 - leverages) ** 2
```

```
sum of leverages = 4.000 (equals number of parameters 4)
row 120: leverage 0.348 residual  -192.80 Cook's D 12.504
row 106: leverage 0.057 residual    48.54 Cook's D 0.062
coefficients without row 120: [19.804  2.04   3.574 -0.496]
```

Row 120 has 10 times the average leverage (`4/121 = 0.033`) and a Cook's distance of 12.5. Dropping it brings `size_m2` back to 2.04 and `rooms` to 3.6. In real work you do not silently delete it: you go find out why it is wrong (data entry? a different market?), then decide.

### Step 6: VIF

```python
def variance_inflation_factors(features):
    factors = []
    for j in range(features.shape[1]):
        others = np.delete(features, j, axis=1)
        beta = fit_ols(others, features[:, j])
        predictions = design_matrix(others) @ beta
        r2, _ = r2_and_adjusted(features[:, j], predictions, others.shape[1])
        factors.append(1 / (1 - r2))
    return np.array(factors)
```

```
VIF    size_m2:   8.77
VIF      rooms:   8.69
VIF  age_years:   1.03
```

Size and rooms are strongly collinear, which is why the outlier could trade weight between them so easily. Building the hat matrix is `O(n^2)` memory: fine here, but for large `n` compute `h_ii` as the row sums of `Q * Q` from a QR decomposition of `X`.

### Step 7: Plots

The script saves `diagnostics.png` with residuals vs fitted, a Q-Q plot from `scipy.stats.probplot`, and residuals vs leverage colored by Cook's distance. Row 120 sits alone in every panel.

## Use It

scikit-learn focuses on prediction and does not report standard errors. You get the fit from sklearn and the statistics from SciPy (or `statsmodels`, whose `OLS(...).fit().summary()` prints this whole lesson in one table, and whose `get_influence()` gives leverage and Cook's distance).

```python
from scipy import stats
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(features, prices)
slope_test = stats.linregress(features[:, 0], prices)
stats.probplot(residuals, dist="norm", plot=axes[1])
```

```
sklearn R^2 0.9438, same coefficients: True
scipy linregress on size only: slope 2.067, p = 1.43e-65
```

`linregress` gives slope, intercept, R, p-value and standard error for one predictor. sklearn's `PredictionErrorDisplay` draws residual-vs-predicted plots for any regressor.

## Exercises

1. **Easy.** Remove row 120, rerun the whole script, and compare the coefficient table, the Shapiro-Wilk p-value and the residual spread with the version including it.
2. **Medium.** Generate data where noise std is proportional to `size_m2`. Show the funnel in residuals vs fitted. Fit again with `log(price)` as the target and compare the plot.
3. **Hard.** Implement leave-one-out deletion directly: refit OLS `n` times, each time dropping one row, and compute `sum((y_hat - y_hat_without_i)^2) / ((p + 1) sigma^2)`. Confirm it equals the Cook's distance formula for every row.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Residual plot | "Check the residuals" | Residuals vs fitted values; structure in it means a violated assumption |
| Adjusted R^2 | "R^2 that penalizes features" | `1 - (1 - R^2)(n - 1)/(n - p - 1)`, which can decrease when a useless feature is added |
| Standard error | "Uncertainty of a coefficient" | `sqrt(sigma^2 [(X^T X)^(-1)]_jj)`, the standard deviation of the estimate under the OLS assumptions |
| Homoscedasticity | "Constant variance" | The error variance does not depend on `x`; its violation is heteroscedasticity |
| Leverage | "Unusual x" | Diagonal of the hat matrix `h_ii`; how far `x_i` is from the rest and how much it can move its own fit |
| Cook's distance | "Influence" | How much all predictions change when sample `i` is removed |
| VIF | "Collinearity score" | `1 / (1 - R^2_j)` from regressing feature `j` on the others |
| Q-Q plot | "Normality check" | Sorted residuals against theoretical normal quantiles; a straight line means normal errors |

## Further Reading

- [An Introduction to Statistical Learning, Section 3.3.3](https://www.statlearning.com/) - potential problems: nonlinearity, heteroscedasticity, outliers, high leverage, collinearity
- [statsmodels: Regression Diagnostics](https://www.statsmodels.org/stable/diagnostic.html) - production implementations of every check in this lesson
- [StatQuest: Linear Regression, Clearly Explained](https://www.youtube.com/watch?v=7ArmBVF2dCs) - R^2, F-tests and p-values for regression
- [scikit-learn: Visualizations with Display Objects](https://scikit-learn.org/stable/visualizations.html) - `PredictionErrorDisplay` for residual plots
