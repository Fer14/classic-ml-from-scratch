# Principal Component Analysis

> Find the directions where your data actually varies, and throw the rest away.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra: eigenvalues & eigenvectors, projections), Phase 03 (Probability and Statistics: expectation & variance), Phase 04 (ML Fundamentals: feature scaling)
**Time:** ~75 minutes

## Learning Objectives

- Derive PCA as the eigen-decomposition of the covariance matrix and implement it from scratch in NumPy
- Use explained variance ratio to choose how many components to keep
- Project data down, reconstruct it back, and measure the information lost
- Read component loadings to explain what each principal component means in terms of the original features

## The Problem

You have a dataset with dozens or hundreds of features. Many of them move together: height in centimeters and height in inches, petal length and petal width, the pixel next to the pixel next to it. Those features carry the same information several times. They slow down training, make distances less meaningful, and make it impossible to plot the data.

You want fewer features that keep most of the information. Not by picking a subset of the original columns, but by building new axes that line up with the directions in which the data spreads out. That is exactly what PCA does, and it does it with nothing more than a covariance matrix and the eigenvectors you met in Phase 01.

PCA is used everywhere: as a preprocessing step before clustering or regression, to visualize high-dimensional data, to compress images and signals, and to remove noise by dropping the low-variance directions.

## The Concept

### Variance as information

Picture a 2D cloud of points stretched along a diagonal line. If you had to describe each point with a single number, the best number is its position along that diagonal. Its position across the diagonal is small and mostly noise.

```
        x2
         |            . .
         |        . .:.:  .        PC1: long axis, big variance
         |     ..:.:.:.            PC2: short axis, small variance
         |  . .:.:..
         |.  .:.
         +-------------------- x1
```

PCA formalizes this: find the unit vector `w` that maximizes the variance of the projected data `Xw`. Then find the next unit vector, orthogonal to the first, that maximizes the remaining variance, and so on.

### The math in four lines

Center the data so every column has mean zero. The covariance matrix is:

```
C = (1 / (n - 1)) * X_c^T X_c          shape (d, d)
```

The variance of the projection onto a unit vector `w` is `w^T C w`. Maximizing `w^T C w` subject to `||w|| = 1` with a Lagrange multiplier gives:

```
C w = lambda w
```

So the best direction is an eigenvector of `C`, and the variance along it is its eigenvalue `lambda`. Sort the eigenvectors by eigenvalue, largest first, and you have the principal components.

### Explained variance

Because `C` is symmetric and positive semi-definite, its eigenvalues are real and non-negative, and they add up to the total variance (the trace of `C`). The fraction kept by the first `k` components is:

```
explained_ratio(k) = (lambda_1 + ... + lambda_k) / (lambda_1 + ... + lambda_d)
```

A common rule is to keep enough components to reach 90-95% of the variance, or to look for the "elbow" in a scree plot of eigenvalues.

### Projection and reconstruction

```mermaid
flowchart LR
    A[X, n x d] -->|subtract mean| B[X_c]
    B -->|times W_k^T, d x k| C[Z, n x k]
    C -->|times W_k, k x d| D[X_c approx]
    D -->|add mean| E[X approx]
```

With `W_k` the matrix whose rows are the top-`k` eigenvectors:

```
Z      = (X - mean) W_k^T
X_back = Z W_k + mean
```

The reconstruction error equals the variance in the discarded directions. Keep all `d` components and the error is zero.

### Scale matters

PCA chases variance, so a feature measured in grams will dominate one measured in kilograms. Unless your features share a unit, standardize them first (Phase 04, feature scaling). That is the same as running PCA on the correlation matrix instead of the covariance matrix.

### Sign ambiguity

If `w` is an eigenvector, so is `-w`. Different libraries can return flipped components. The geometry is identical; only the sign of the coordinates changes. The code below fixes a convention (the largest-magnitude loading is positive) so results are reproducible.

## Build It

### Step 1: PCA from the covariance matrix

`np.linalg.eigh` is the eigen-solver for symmetric matrices. It returns eigenvalues in ascending order, so you reverse them.

```python
class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        covariance = X_centered.T @ X_centered / (X.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        signs = np.sign(eigenvectors[np.abs(eigenvectors).argmax(axis=0), range(eigenvectors.shape[1])])
        eigenvectors = eigenvectors * signs
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / eigenvalues.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_
```

Test it on a 2D cloud where `x2 = 0.8 * x1 + noise`:

```python
X_2d = make_correlated_2d(500, seed=0)
pca_2d = PCAFromScratch(n_components=2).fit(X_2d)
print(pca_2d.components_)
print(pca_2d.explained_variance_ratio_)
```

```
[[ 0.7761  0.6306]
 [-0.6306  0.7761]]
Explained variance ratio: [0.9913 0.0087]
```

The first direction points along the slope `0.6306 / 0.7761 = 0.81`, the true relationship. It carries 99% of the variance.

### Step 2: Reduce iris from 4D to 2D

```python
X_std = (X_iris - X_iris.mean(axis=0)) / X_iris.std(axis=0)
pca_iris = PCAFromScratch(n_components=4).fit(X_std)
cumulative = np.cumsum(pca_iris.explained_variance_ratio_)
k_for_95 = int(np.searchsorted(cumulative, 0.95) + 1)
```

```
  PC1: ratio=0.7296  cumulative=0.7296
  PC2: ratio=0.2285  cumulative=0.9581
  PC3: ratio=0.0367  cumulative=0.9948
  PC4: ratio=0.0052  cumulative=1.0000
Components needed for 95% variance: 2
```

Two components keep 96% of the variance. In that 2D space setosa sits far to the left (PC1 around -2.2) while versicolor and virginica sit to the right, which is why iris looks so easy in a PCA plot.

### Step 3: Reconstruction error

```python
for k in range(1, 5):
    model = PCAFromScratch(n_components=k).fit(X_std)
    X_back = model.inverse_transform(model.transform(X_std))
    mse = np.mean((X_std - X_back) ** 2)
```

```
  k=1: reconstruction MSE = 0.270376
  k=2: reconstruction MSE = 0.041868
  k=3: reconstruction MSE = 0.005179
  k=4: reconstruction MSE = 0.000000
```

Check the identity: with standardized data the total variance per feature is 1, and `1 - 0.7296 = 0.2704`, which is exactly the `k=1` error: the error is the share of variance you dropped.

### Step 4: Interpret the loadings

Each row of `components_` tells you how much each original feature contributes to that component.

```
  sepal length (cm)    PC1=+0.521  PC2=+0.377
  sepal width (cm)     PC1=-0.269  PC2=+0.923
  petal length (cm)    PC1=+0.580  PC2=+0.024
  petal width (cm)     PC1=+0.565  PC2=+0.067
```

PC1 is "overall flower size, dominated by petals". PC2 is almost pure sepal width. Loadings are how you turn abstract components back into a story.

## Use It

```python
from sklearn.decomposition import PCA

sk_pca = PCA(n_components=2).fit(X_std)
Z_sklearn = sk_pca.transform(X_std)
print(sk_pca.explained_variance_ratio_)
```

```
sklearn explained variance ratio: [0.7296 0.2285]
scratch explained variance ratio: [0.7296 0.2285]
Max abs difference in projections (after sign alignment): 8.88e-16
```

sklearn computes PCA with an SVD of the centered data rather than an eigen-decomposition of the covariance (next lesson explains why that is more stable). Useful options:

- `PCA(n_components=0.95)` keeps as many components as needed for 95% variance.
- `PCA(whiten=True)` rescales components to unit variance.
- `svd_solver="randomized"` is much faster for large, wide matrices.
- `IncrementalPCA` fits in mini-batches when the data does not fit in memory.

Put `StandardScaler` and `PCA` in the same pipeline so the scaling learned on training data is reused on test data.

## Exercises

1. **Easy.** Run PCA on iris without standardizing. How does the explained variance ratio of PC1 change, and which feature dominates its loadings? Explain why.
2. **Medium.** Load `sklearn.datasets.load_digits` (64 features). Plot the cumulative explained variance and find how many components you need for 90% and 99%. Reconstruct a few digits with 5, 15 and 40 components and display them side by side.
3. **Hard.** Implement PCA with power iteration plus deflation instead of `eigh`: repeatedly multiply a random vector by `C`, normalize, read off the eigenvalue, then subtract `lambda * w w^T` from `C`. Compare the components and timing with your `eigh` version on the digits data.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Principal component | "A new feature" | A unit eigenvector of the covariance matrix; a direction in feature space ordered by how much variance lies along it |
| Explained variance | "How much info the component keeps" | The eigenvalue for that component, i.e. the variance of the data projected onto it |
| Explained variance ratio | "Percentage kept" | The eigenvalue divided by the sum of all eigenvalues |
| Loadings | "Feature weights of a component" | The entries of a component vector, showing each original feature's contribution |
| Scree plot | "The elbow chart" | Eigenvalues plotted in decreasing order, used to pick how many components to keep |
| Reconstruction error | "What you lose" | Mean squared difference between data and its projection back from k components; equals the discarded variance |
| Whitening | "Making components unit-variance" | Dividing each projected coordinate by the square root of its eigenvalue |

## Further Reading

- [A Tutorial on Principal Component Analysis (Jonathon Shlens)](https://arxiv.org/abs/1404.1100) -- clear derivation of PCA from covariance and its relation to SVD
- [StatQuest: Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) -- visual walk-through of projections, loadings and scree plots
- [An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/) -- chapter 12 covers PCA as an unsupervised learning tool
- [scikit-learn: Decomposing signals in components](https://scikit-learn.org/stable/modules/decomposition.html) -- PCA, IncrementalPCA, KernelPCA and TruncatedSVD in practice
