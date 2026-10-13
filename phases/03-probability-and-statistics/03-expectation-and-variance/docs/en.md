# Expectation and Variance

> The mean tells you where a distribution sits. The variance tells you how far it spreads. Most of statistics is these two numbers.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra), Phase 03 Lesson 02 (Random Variables and Distributions)
**Time:** ~60 minutes

## Learning Objectives

- Compute expectation and variance for discrete and continuous random variables from their PMF or PDF
- Use linearity of expectation and the scaling rules for variance, and know when Var(X + Y) = Var(X) + Var(Y) fails
- Explain why the sample variance divides by n - 1 and verify it by simulation
- Compute covariance, correlation, and a covariance matrix, and match them against `np.cov` and `np.corrcoef`

## The Problem

MSE is an expectation: the average squared error. The bias-variance decomposition (Phase 04) splits that expectation into pieces. Standardizing features means subtracting the mean and dividing by the standard deviation. PCA (Phase 09) is an eigen-decomposition of the covariance matrix. Random forests work because averaging reduces variance.

Every one of those ideas uses the same handful of rules about E[X] and Var(X). If you know them, you can predict what happens when you scale a feature, add noise, or average models, without running a single experiment.

## The Concept

### Expectation

The expectation is the probability-weighted average of the values:

```
Discrete:    E[X] = sum_k  k * P(X = k)
Continuous:  E[X] = ∫ x f(x) dx
```

More generally, for any function g:

```
E[g(X)] = sum_k g(k) P(X = k)      or      ∫ g(x) f(x) dx
```

A fair die has E[X] = (1 + 2 + ... + 6) / 6 = 3.5. No single roll is 3.5; it is the long-run average.

### Linearity

Expectation is linear, always, even when X and Y are dependent:

```
E[aX + b] = a E[X] + b
E[X + Y]  = E[X] + E[Y]
```

This is why the average loss over a dataset can be split into per-sample terms, and why the expected number of heads in n flips is simply n p.

### Variance and Standard Deviation

Variance is the expected squared distance from the mean:

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
SD(X)  = sqrt(Var(X))
```

The standard deviation is in the same units as X; the variance is in squared units.

Scaling rules:

```
Var(aX + b) = a^2 Var(X)                 shifting does nothing, scaling squares
Var(X + Y)  = Var(X) + Var(Y) + 2 Cov(X, Y)
            = Var(X) + Var(Y)            only if X, Y are uncorrelated
```

Var(X + X) = Var(2X) = 4 Var(X), not 2 Var(X). Adding perfectly correlated copies does not average anything out. Adding independent copies does, which is the whole idea behind bagging.

### Sample Estimates and Bessel's Correction

You rarely know the distribution, so you estimate from data:

```
sample mean:      x̄ = (1/n) sum x_i
sample variance:  s^2 = (1/(n-1)) sum (x_i - x̄)^2
```

Why n - 1? The deviations are measured from x̄, which was fitted to the same data, so they are systematically a bit too small. On average, dividing by n gives (n - 1)/n times the true variance. Dividing by n - 1 removes that bias. With n = 5 the difference is 20%; with n = 10,000 it is negligible.

In NumPy, `np.var` divides by n by default (`ddof=0`). Pass `ddof=1` for the unbiased version. pandas `.var()` uses `ddof=1` by default. This mismatch catches everyone at least once.

### Covariance and Correlation

Covariance measures whether two variables move together:

```
Cov(X, Y) = E[(X - E[X]) (Y - E[Y])]
Corr(X, Y) = Cov(X, Y) / (SD(X) SD(Y))       in [-1, 1]
```

Correlation is covariance with the units removed. It only measures **linear** association. If U is uniform on [-1, 1], then U and U^2 are completely dependent, yet their correlation is 0.

For a data matrix X with n rows and d features, center each column and compute:

```
Σ = (1/(n-1)) X_c^T X_c          (d x d, symmetric, variances on the diagonal)
```

This is the linear algebra from Phase 01 doing statistics.

## Build It

### Step 1: Discrete expectation and variance

```python
def discrete_expectation(values, probabilities, function=lambda x: x):
    return sum(function(v) * p for v, p in zip(values, probabilities))


def discrete_variance(values, probabilities):
    mean = discrete_expectation(values, probabilities)
    return discrete_expectation(values, probabilities, lambda x: (x - mean) ** 2)
```

```
fair   E[X]=3.5000  Var=2.9167  E[X^2]-E[X]^2=2.9167
loaded E[X]=4.5000  Var=3.2500  E[X^2]-E[X]^2=3.2500
```

Both formulas for variance agree.

### Step 2: Continuous expectation by integration

```python
def continuous_expectation(pdf, grid, function=lambda x: x):
    return np.trapezoid(function(grid) * pdf(grid), grid)
```

For Exponential(rate = 2) this returns E[X] = 0.5 and Var = 0.25, matching the formulas 1/λ and 1/λ^2.

### Step 3: Linearity and scaling, checked by simulation

```
E[2X + 5]       = 10.9970  vs 2E[X]+5 = 10.9970
Var(2X + 5)     = 16.0882  vs 4Var(X) = 16.0882
Var(X + Y) indep= 5.0177  vs Var(X)+Var(Y) = 5.0154
Var(X + X)      = 16.0882  vs 2Var(X) = 8.0441  (not independent)
```

The first two are identities and match to every digit. The independent sum matches up to sampling noise. The last line is the trap.

### Step 4: Biased vs unbiased sample variance

```python
def sample_variance(data, ddof=0):
    mean = sample_mean(data)
    return sum((x - mean) ** 2 for x in data) / (len(data) - ddof)
```

Draw 20,000 samples of size 5 from a normal with variance 4 and average the estimates:

```
Average of divide-by-n     estimates: 3.2022  (theory 3.20)
Average of divide-by-(n-1) estimates: 4.0027
```

### Step 5: Covariance, correlation, covariance matrix

```python
def covariance(x, y):
    mean_x, mean_y = sample_mean(x), sample_mean(y)
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / (len(x) - 1)


def correlation(x, y):
    return covariance(x, y) / np.sqrt(sample_variance(x, ddof=1) * sample_variance(y, ddof=1))


def covariance_matrix(data):
    centered = data - data.mean(axis=0)
    return centered.T @ centered / (len(data) - 1)
```

```
corr(hours, score)= 0.9315
corr(hours, shoe) = -0.0313
corr(u, u^2)      = -0.0259  (dependent but uncorrelated)
```

## Use It

```python
np.var(exam_score)
np.var(exam_score, ddof=1)
np.cov(data, rowvar=False)
np.corrcoef(hours_studied, exam_score)[0, 1]
stats.expon(scale=0.5).mean(), stats.expon(scale=0.5).var()
stats.describe(exam_score)
```

All match your scratch versions. Two things to remember: `np.cov` treats rows as variables unless you pass `rowvar=False`, and it uses `ddof=1` while `np.var` uses `ddof=0`. Every scipy distribution object exposes `.mean()`, `.var()`, and `.std()` computed from the exact formulas, so you can check any derivation in one line.

## Exercises

1. (Easy) Compute E[X] and Var(X) for a Binomial(20, 0.3) by summing over its PMF, and compare with n p and n p (1 - p).
2. (Medium) Average k independent Normal(0, 1) samples for k = 1, 2, 4, ..., 64, repeated 10,000 times. Show empirically that the variance of the average is 1/k. Then repeat with samples that have pairwise correlation 0.5 and observe that the variance stops shrinking. Relate this to why random forests decorrelate their trees.
3. (Hard) Prove that E[(1/n) sum (x_i - x̄)^2] = (n-1)/n σ^2 by expanding the square and using Var(x̄) = σ^2/n. Check each intermediate quantity numerically.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Expectation | "The average" | The probability-weighted mean of a random variable, E[X] = sum k P(X = k) or ∫ x f(x) dx |
| Linearity of expectation | "Means add" | E[aX + bY] = a E[X] + b E[Y], valid even for dependent variables |
| Variance | "Spread" | E[(X - E[X])^2], the expected squared deviation from the mean |
| Standard deviation | "Typical deviation" | The square root of the variance, in the same units as X |
| Bessel's correction | "Divide by n - 1" | Using n - 1 in the sample variance to make it an unbiased estimate of the true variance |
| Covariance | "Moving together" | E[(X - E[X])(Y - E[Y])], positive when both tend to be above their means together |
| Correlation | "How related" | Covariance divided by both standard deviations; measures only linear association |
| Covariance matrix | "Σ" | The d x d matrix of pairwise covariances between features, with variances on the diagonal |

## Further Reading

- [Introduction to Probability, Blitzstein and Hwang](https://projects.iq.harvard.edu/stat110/home) -- chapter 4 (expectation) and chapter 7 (covariance and correlation)
- [Mathematics for Machine Learning, Deisenroth, Faisal, Ong](https://mml-book.github.io/) -- section 6.4 covers summary statistics, independence, and covariance matrices
- [StatQuest: Covariance and Correlation, Clearly Explained](https://www.youtube.com/watch?v=qtaqvPAeEJY) -- visual intuition for both quantities
- [NumPy statistics routines](https://numpy.org/doc/stable/reference/routines.statistics.html) -- `mean`, `var`, `std`, `cov`, `corrcoef` and their `ddof` conventions
