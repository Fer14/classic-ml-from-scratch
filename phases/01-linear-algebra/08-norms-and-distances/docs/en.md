# Norms and Distances

> "Similar" means "close", and "close" depends on how you measure. Choosing the norm is choosing what your model considers similar.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 Lesson 02 (Dot Product and Projections), Phase 01 Lesson 07 (Eigenvalues and Eigenvectors)
**Time:** ~60 minutes

## Learning Objectives

- Compute L1, L2, L-infinity, and general Lp norms, and draw their unit circles
- Explain how the choice of metric, feature scaling, and cosine vs. Euclidean distance change which point is "nearest"
- Implement Mahalanobis distance and explain it as Euclidean distance after undoing the data's covariance
- Compute pairwise distances efficiently with `|a - b|^2 = |a|^2 + |b|^2 - 2 a.b`, and describe the curse of dimensionality

## The Problem

k-nearest neighbors (Phase 06), k-means (Phase 08), DBSCAN, hierarchical clustering, t-SNE, anomaly detection: all of them are built on one primitive, the distance between two feature vectors. Regularization (Phase 05) is built on another: the norm of the weight vector, where L1 produces sparse models and L2 does not.

Change the distance and you change the model. A house that is the nearest neighbor under Euclidean distance may not be under Manhattan distance. A feature measured in square meters will drown out a feature measured in rooms. In 1000 dimensions every point is about equally far from every other, and "nearest" stops meaning much. This lesson makes these effects concrete so you can choose deliberately.

## The Concept

### Norms: the length of a vector

A norm is a function `|v|` that behaves like a length:

```
|v| >= 0, and |v| = 0 only for v = 0
|c v| = |c| |v|                  scaling scales the length
|v + w| <= |v| + |w|             triangle inequality: detours are never shorter
```

The Lp family:

```
|v|_p = (|v1|^p + |v2|^p + ... + |vn|^p)^(1/p)

p = 1    |v|_1   = sum of absolute values          (Manhattan, taxicab)
p = 2    |v|_2   = sqrt(sum of squares)            (Euclidean, Lesson 02)
p = inf  |v|_inf = max of absolute values          (Chebyshev)
```

For `v = [3, -4, 1]`: L1 = 8, L2 = 5.099, L3 = 4.514, L-inf = 4. As `p` grows, the biggest component takes over.

### Unit circles

The shape of `{v : |v|_p = 1}` tells you everything about a norm (left panel of `code/out.png`):

```
          L1                 L2                L-inf
          /\               .---.             +-----+
         /  \             /     \            |     |
         \  /             \     /            |     |
          \/               '---'             +-----+
       diamond             circle            square
```

- L1 is a diamond with corners on the axes. When you minimize a loss subject to a small L1 norm (Lasso), the solution tends to hit a corner, where some coordinates are exactly zero. That is why L1 regularization produces sparse weights.
- L2 is round: no direction is special. Ridge regression shrinks all weights smoothly.
- For `p < 1` the "circle" is star-shaped (not convex) and the triangle inequality fails: `|[1,0] + [0,1]|_0.5 = 4 > 2`. It is not a norm.

### Distances from norms

Any norm gives a distance: `d(a, b) = |a - b|`. The length of the arrow from one point to the other (Lesson 01). Different norms disagree about who is nearest:

```
query = (0, 0)       A = (3.9, 0)    B = (3, 2)    C = (2.9, 2.9)

          A      B       C
L1      3.9    5.0     5.8       nearest: A
L2      3.9    3.606   4.101     nearest: B
L-inf   3.9    3.0     2.9       nearest: C
```

### Cosine distance

`1 - cos_sim(a, b)` (Lesson 02) measures the angle, ignoring length. For text, where a longer document has bigger counts but the same topic, this is usually what you want. Note that cosine distance does not satisfy the triangle inequality, so it is not a true metric, but it is used as one all the time.

### Scale matters

Euclidean distance adds squared differences across features. A 30 m2 difference contributes 900; a 2-room difference contributes 4. Without scaling, "distance" means "difference in size". Standardize features (Phase 00 Lesson 05) before using any distance-based method.

### Mahalanobis distance: the data's own ruler

Data clouds are often stretched and tilted. A point 3 units away *along* the cloud is ordinary; a point 3 units away *across* it is an outlier. Mahalanobis distance accounts for this:

```
d_M(a, b) = sqrt((a - b)^T  Sigma^-1  (a - b))
```

Geometrically (using Lesson 07): rotate into the covariance eigenbasis, divide each axis by its standard deviation, then measure ordinary Euclidean distance. It turns the elliptical cloud into a round one. With standard deviations 3 (along) and 0.5 (across), a step of 3 along the cloud has Mahalanobis distance 1, and a step of 3 across it has distance 6.

### Pairwise distances, efficiently

Expand the square:

```
|a - b|^2 = (a - b) . (a - b) = |a|^2 + |b|^2 - 2 a . b
```

For all pairs at once: a column of `|a|^2`, plus a row of `|b|^2`, minus `2 A B^T`. The heavy part is one matrix product (fast BLAS, Lesson 04), and memory is `n x m` instead of `n x m x d`. The catch: subtracting nearly equal large numbers loses precision, so tiny negative values can appear. Clamp to 0 before the square root.

### The curse of dimensionality

Draw random points in a d-dimensional cube and measure distances from one of them. As `d` grows, the farthest and nearest points become almost equally far:

```
d=2     (max - min) / min = 94
d=10                       1.9
d=100                      0.43
d=1000                     0.12
```

Each distance is a sum of `d` independent terms, and sums of many random terms concentrate around their mean (you will see why in Phase 03). When all distances look alike, nearest-neighbor methods lose their signal. This is a major reason for dimensionality reduction (Phase 09).

## Build It

### Step 1: Lp norms and distances

```python
def lp_norm(vector, p):
    absolute = [abs(component) for component in vector]
    if p == math.inf:
        return max(absolute)
    return sum(value ** p for value in absolute) ** (1 / p)


def minkowski_distance(first, second, p):
    return lp_norm(np.asarray(first) - np.asarray(second), p)


def cosine_distance(first, second):
    return 1 - (first @ second) / (lp_norm(first, 2) * lp_norm(second, 2))
```

The script checks the norm axioms for `p = 1, 2, inf` on random vectors and shows `p = 0.5` breaking the triangle inequality.

### Step 2: Cosine vs. Euclidean, raw vs. standardized

```
  euclidean(short, long) = 20.125, euclidean(short, other) = 2.828
  cosine(short, long)    = 0.000, cosine(short, other)    = 0.800

  raw [m2, rooms]: d(house0, house1) = 2.24, d(house0, house2) = 30.00
  standardized:    d(house0, house1) = 2.12, d(house0, house2) = 2.16
```

Raw distances say a house with 2 fewer rooms is almost identical, while one 30 m2 larger is far away. After standardization both differences count comparably.

### Step 3: Mahalanobis distance

```python
def mahalanobis_distance(first, second, inverse_covariance):
    difference = np.asarray(first) - np.asarray(second)
    return math.sqrt(difference @ inverse_covariance @ difference)
```

```
  Mahalanobis: along = 1.00, across = 6.04
```

### Step 4: Pairwise Euclidean with one matrix product

```python
def pairwise_euclidean(points_a, points_b):
    squared_a = (points_a ** 2).sum(axis=1)[:, np.newaxis]
    squared_b = (points_b ** 2).sum(axis=1)[np.newaxis, :]
    squared = squared_a + squared_b - 2 * points_a @ points_b.T
    return np.sqrt(np.maximum(squared, 0.0))
```

```
  |a - b|^2 = |a|^2 + |b|^2 - 2 a.b matches broadcasting: True
  largest diagonal entry (should be 0): 1.5e-07, rounding error from cancellation
  memory: dot-product version 2 MB vs broadcast intermediate 40 MB
```

The diagonal should be exactly 0 but comes out around `1e-7`: that is the cancellation error, and the reason the comparison uses a tolerance. This is the solution to the hard exercise of Phase 00 Lesson 05.

### Step 5: Watch distances concentrate

```python
for dimension in [2, 10, 100, 1000]:
    sample = generator.uniform(size=(300, dimension))
    distances_from_first = pairwise_euclidean(sample[:1], sample[1:])[0]
    contrast = (distances_from_first.max() - distances_from_first.min()) / distances_from_first.min()
```

The right panel of `code/out.png` plots the collapse on log-log axes.

## Use It

```python
np.linalg.norm(vector, ord=1)
np.linalg.norm(vector, ord=np.inf)

from scipy.spatial.distance import cdist
cdist(A, B, metric="cityblock")
cdist(A, B, metric="mahalanobis", VI=inverse_covariance)

from sklearn.metrics import pairwise_distances
pairwise_distances(points)
pairwise_distances(A, B, metric="cosine")

from sklearn.neighbors import NearestNeighbors
NearestNeighbors(n_neighbors=1, metric="chebyshev").fit(candidates).kneighbors([query])
```

All match the from-scratch values, and `NearestNeighbors` picks A, B, and C for Manhattan, Euclidean, and Chebyshev, exactly as computed by hand. scikit-learn's `pairwise_distances` uses the same dot-product trick for Euclidean distance; `cdist` computes differences directly, which is slower but more accurate for nearly identical points.

Every distance-based scikit-learn estimator (`KNeighborsClassifier`, `DBSCAN`, `AgglomerativeClustering`) takes a `metric` argument. Treat it as a hyperparameter, not a default.

## Exercises

1. **Easy.** Compute the L1, L2, and L-inf distances between `[1, 2, 3]` and `[4, 0, 3]` by hand, then with `minkowski_distance` and `cdist`.
2. **Medium.** Generate 200 points in 2-D, pick a query point, and plot the 10 nearest neighbors under L1, L2, and L-inf in three panels. Draw the "ball" that contains them (the diamond, circle, or square of the right radius) to see why the sets differ.
3. **Hard.** Implement `whiten(data)` that rotates data into its covariance eigenbasis and divides each axis by the square root of its eigenvalue (Lesson 07). Show that the Euclidean distance between two whitened points equals their Mahalanobis distance in the original space, for 100 random pairs.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Norm | "Length of a vector" | A function that is zero only at 0, scales with the vector, and obeys the triangle inequality |
| L1 norm | "Manhattan / taxicab" | Sum of absolute components; its unit ball is a diamond, which favors sparse solutions |
| L2 norm | "Euclidean length" | Square root of the sum of squared components; its unit ball is round |
| L-infinity norm | "Max norm / Chebyshev" | The largest absolute component; its unit ball is a square |
| Metric | "Distance function" | A function d(a, b) that is non-negative, symmetric, zero only for a = b, and satisfies the triangle inequality |
| Cosine distance | "Angle distance" | 1 minus cosine similarity; ignores vector length and is not a true metric |
| Mahalanobis distance | "Covariance-aware distance" | sqrt((a - b)^T Sigma^-1 (a - b)): Euclidean distance after whitening the data |
| Curse of dimensionality | "High-D is weird" | In many dimensions, distances concentrate so nearest and farthest points become almost equally far |

## Further Reading

- [3Blue1Brown, Essence of Linear Algebra (full series)](https://www.3blue1brown.com/topics/linear-algebra) -- revisit chapters 9 (dot products) and 14 (eigenvectors) for the geometry behind Euclidean and Mahalanobis distance
- [scikit-learn: Pairwise metrics](https://scikit-learn.org/stable/modules/metrics.html) -- distance and kernel functions available to every estimator
- [scipy.spatial.distance.cdist documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) -- the full list of supported metrics
- [Beyer et al., "When Is 'Nearest Neighbor' Meaningful?" (ICDT 1999)](https://doi.org/10.1007/3-540-49257-7_15) -- the classic paper on distance concentration in high dimensions
