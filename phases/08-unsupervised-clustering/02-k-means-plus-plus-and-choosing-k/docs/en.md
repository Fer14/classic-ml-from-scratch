# K-Means++ and Choosing K

> Good seeds make k-means reliable; good diagnostics tell you how many seeds to plant. Neither is magic, and both are cheap.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 08 Lesson 01 (K-Means), Phase 03 (Probability and Statistics: Random Variables & Distributions)
**Time:** ~60 minutes

## Learning Objectives

- Implement k-means++ seeding and explain why sampling proportional to squared distance spreads the centroids out
- Measure how often random initialization and k-means++ land in bad local minima
- Choose `k` with the elbow method and the gap statistic, built from scratch
- Cross-check the choice with scikit-learn's silhouette score and know when the three methods disagree

## The Problem

The previous lesson left you with two open problems. First, k-means with random starting centroids sometimes converges to a terrible solution: one run reached inertia 563, another 2386 on the same data. Second, you had to tell the algorithm `k = 3`. In real data nobody tells you how many customer segments exist.

The first problem has an elegant fix that costs almost nothing: choose the initial centroids so they are far apart. The second problem has no perfect answer, but there are a handful of diagnostics that, used together, give you a defensible choice.

## The Concept

### Why Random Seeding Fails

If two initial centroids land in the same true cluster, one of them has to "steal" another cluster during the iterations. Lloyd's algorithm only moves centroids to local means, so a centroid sitting between two blobs can stay there forever. With `k` clusters of equal size, the probability that random seeding puts exactly one seed in each cluster is `k! / k^k`. For `k = 5` that is about 4%. Most of the time Lloyd's iterations have to repair the seeding, and sometimes they cannot.

### K-Means++ Seeding

Arthur and Vassilvitskii (2007) proposed a simple fix:

```
1. Pick the first centroid uniformly at random from the data.
2. For every point x, compute D(x) = distance to the nearest centroid chosen so far.
3. Pick the next centroid at random with probability D(x)^2 / sum D^2.
4. Repeat 2-3 until you have k centroids. Then run Lloyd's algorithm as usual.
```

Points far from every existing centroid are much more likely to be chosen, so seeds spread across the data. It is still random: an outlier can occasionally be picked, and a bad run is still possible, just much rarer. The authors proved the expected inertia after seeding is within `O(log k)` of the optimum. Scikit-learn uses k-means++ by default and also runs several initializations (`n_init`).

```
random seeds           k-means++ seeds

 oo*o   oooo            oo*o   oo*o
 o*oo   oooo            oooo   oooo
                  vs
 oooo   oooo            oooo   oo*o
 oooo   oo*o            oo*o   oooo
two seeds share a      one seed per group
group, one is empty
```

### The Elbow Method

Inertia always decreases as `k` grows, so you cannot just minimize it. Instead you plot inertia against `k` and look for the "elbow": the point after which adding clusters only gives small improvements. Before the elbow, each new cluster splits a real group and inertia drops a lot. After it, a new cluster just splits a compact group in two and the gain is small.

The weakness: the elbow is a visual judgment. On real data the curve is often smooth with no clear bend.

### The Gap Statistic

Tibshirani, Walther and Hastie (2001) made the elbow quantitative by comparing your inertia with what you would get on data that has no clusters at all:

```
Gap(k) = mean over B references of log(W_ref(k))  -  log(W_data(k))
```

- `W(k)` is the inertia with `k` clusters.
- Reference datasets are drawn uniformly over the bounding box of the data.
- A large gap means your data clusters much better than structureless noise.

Their rule picks the smallest `k` with `Gap(k) >= Gap(k+1) - s(k+1)`, where `s` is the standard error of the reference log-inertias. In practice many people simply pick the `k` with the largest gap. As you will see, the original rule can stop too early when the gap curve is flat for small `k`.

### The Silhouette Score (preview)

For each point, compare the mean distance to its own cluster (`a`) with the mean distance to the nearest other cluster (`b`): `s = (b - a) / max(a, b)`. Average over all points. Values near 1 mean tight, well-separated clusters. You will build it in the cluster evaluation lesson; here you use scikit-learn's version as a third opinion.

## Build It

### Step 1: Random and k-means++ seeding

```python
def random_init(X, k, rng):
    return X[rng.choice(len(X), k, replace=False)].copy()


def kmeans_plus_plus_init(X, k, rng):
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        nearest_sq_dist = squared_distances(X, np.array(centroids)).min(axis=1)
        probabilities = nearest_sq_dist / nearest_sq_dist.sum()
        centroids.append(X[rng.choice(len(X), p=probabilities)])
    return np.array(centroids)
```

Points already chosen as centroids have `D = 0`, so they can never be picked twice.

### Step 2: Lloyd's algorithm with pluggable seeding and restarts

```python
def lloyd(X, centroids, max_iter=100):
    for _ in range(max_iter):
        labels = np.argmin(squared_distances(X, centroids), axis=1)
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(len(centroids))
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    labels = np.argmin(squared_distances(X, centroids), axis=1)
    inertia = float(np.sum((X - centroids[labels]) ** 2))
    return centroids, labels, inertia


def kmeans(X, k, init, seed, n_init=1):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        result = lloyd(X, init(X, k, rng))
        if best is None or result[2] < best[2]:
            best = result
    return best
```

### Step 3: Compare seeding strategies

Five well-separated blobs, 50 single runs with each seeding. A run is "bad" if its inertia is more than 1% above the best one found.

```
  random     | mean inertia   1030.9 | worst   4912.9 | bad runs 16/50
  k-means++  | mean inertia    780.8 | worst   1754.4 | bad runs  7/50
  best inertia found: 639.4
```

K-means++ more than halves the failure rate and the worst case is far less bad. It does not eliminate failures, which is why you still combine it with `n_init` restarts.

### Step 4: The elbow

```python
inertias = [kmeans(X, k, kmeans_plus_plus_init, seed=0, n_init=5)[2] for k in k_values]
```

```
  k=4 | inertia    1676.4 | ####
  k=5 | inertia     639.4 | #
  k=6 | inertia     585.3 | #
    4 -> 5:  61.9%
    5 -> 6:   8.5%
```

Every step up to `k = 5` removes 40-60% of the remaining inertia; after it, each new cluster removes under 10%. The elbow is at 5.

### Step 5: The gap statistic

```python
def gap_statistic(X, k_values, n_references, seed):
    rng = np.random.default_rng(seed)
    low, high = X.min(axis=0), X.max(axis=0)
    gaps, errors = [], []
    for k in k_values:
        observed = np.log(kmeans(X, k, kmeans_plus_plus_init, seed, n_init=3)[2])
        reference_logs = []
        for _ in range(n_references):
            uniform_data = rng.uniform(low, high, size=X.shape)
            reference_logs.append(np.log(kmeans(uniform_data, k, kmeans_plus_plus_init, seed, n_init=3)[2]))
        reference_logs = np.array(reference_logs)
        gaps.append(reference_logs.mean() - observed)
        errors.append(reference_logs.std() * np.sqrt(1 + 1 / n_references))
    return np.array(gaps), np.array(errors)
```

```
  k=1 | gap  0.067 | s_k 0.025
  k=2 | gap  0.058 | s_k 0.026
  k=5 | gap  1.524 | s_k 0.024
  k=6 | gap  1.427 | s_k 0.027
k with the largest gap: 5
Tibshirani rule (smallest k with Gap(k) >= Gap(k+1) - s(k+1)): 1
```

The maximum is clearly at 5. The original rule stops at 1 because, at a coarse scale, splitting five blobs into two groups is no better than splitting uniform noise into two, so `Gap(1)` and `Gap(2)` are both near zero. This is a real limitation: the rule assumes the gap curve rises steadily until the true `k`. Always look at the whole curve, not just the rule's output.

## Use It

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 9):
    sk_model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(X)
    score = silhouette_score(X, sk_model.labels_)
    print(f"  k={k} | inertia {sk_model.inertia_:9.1f} | silhouette {score:.3f}")
```

```
  k=4 | inertia    1676.4 | silhouette 0.660
  k=5 | inertia     639.4 | silhouette 0.723
  k=6 | inertia     587.4 | silhouette 0.639
```

Scikit-learn reaches the same inertia (639.4) and the silhouette score also peaks at 5. On this synthetic data the elbow, the gap maximum and the silhouette all agree. On real data they often do not; when that happens, prefer the `k` that is useful for the downstream decision (how many segments can marketing actually act on?) among the candidates the diagnostics support. `sklearn.cluster.kmeans_plusplus` exposes the seeding alone if you need it.

## Exercises

1. **Easy.** Compute `k! / k^k` for `k = 3, 5, 10` and compare with the fraction of bad random-init runs you observe when you change the data to 10 blobs.
2. **Medium.** Implement the "greedy" k-means++ variant scikit-learn uses: at each step, sample `2 + log(k)` candidates and keep the one that reduces total `D^2` the most. Does it reduce bad runs further?
3. **Hard.** Generate 5 blobs where one has standard deviation 3 and the rest 0.5. Run elbow, gap and silhouette. Which method is most misled by unequal cluster spreads, and why does inertia-based reasoning prefer splitting the wide cluster?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| k-means++ | "Smart initialization" | Seeding where each new centroid is a data point sampled with probability proportional to its squared distance to the nearest existing centroid |
| n_init | "Number of restarts" | Run k-means from several seedings and keep the solution with the lowest inertia |
| Elbow method | "Look for the bend" | Plot inertia against k and pick the k after which adding clusters yields only small decreases |
| Gap statistic | "Elbow with a null model" | Difference between expected log-inertia on uniform reference data and observed log-inertia; large gaps indicate real structure |
| Silhouette score | "How well points fit their cluster" | Mean over points of (b - a) / max(a, b), where a is mean intra-cluster distance and b mean distance to the nearest other cluster |

## Further Reading

- [Arthur and Vassilvitskii, "k-means++: The Advantages of Careful Seeding" (2007)](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) -- the original paper with the O(log k) guarantee
- [Tibshirani, Walther and Hastie, "Estimating the number of clusters in a data set via the gap statistic" (2001)](https://hastie.su.domains/Papers/gap.pdf) -- the gap statistic paper
- [scikit-learn: Selecting the number of clusters with silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) -- worked example with silhouette plots
- [The Elements of Statistical Learning, Section 14.3](https://hastie.su.domains/ElemStatLearn/) -- cluster analysis, including choosing the number of clusters
