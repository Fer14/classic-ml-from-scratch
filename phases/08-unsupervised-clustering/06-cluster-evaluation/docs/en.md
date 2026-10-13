# Cluster Evaluation

> There is no test set for clustering. You either measure geometry (internal metrics) or compare against labels you held back (external metrics), and both can lie.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 08 Lessons 01-05 (K-Means through Gaussian Mixtures), Phase 04 (ML Fundamentals: Metrics), Phase 07 (Trees and Ensembles: Entropy & Gini)
**Time:** ~75 minutes

## Learning Objectives

- Implement the silhouette score, Davies-Bouldin index and Calinski-Harabasz index from scratch and read them correctly
- Implement purity, the Adjusted Rand Index and Normalized Mutual Information from a contingency table
- Explain why purity is gameable and why ARI and NMI correct for chance
- Show a case where internal metrics prefer the wrong clustering, and match every metric against `sklearn.metrics`

## The Problem

In supervised learning you hold out a test set and measure accuracy. Clustering has no ground truth. You ran k-means with `k = 3` and `k = 4`, DBSCAN with two values of `eps`, and a Gaussian mixture. Which result is "best"?

Two situations arise in practice:

1. **No labels at all.** You can only ask whether the clusters are geometrically convincing: tight inside, separated from each other. These are internal metrics.
2. **Labels for a sample, or a benchmark.** You hide the labels during clustering and compare afterwards. These are external metrics, and they are how papers and the earlier lessons in this phase graded algorithms (you have been using ARI since Lesson 01).

Both families encode assumptions. Knowing those assumptions is the difference between using the metrics and being fooled by them.

## The Concept

### Internal Metrics

**Silhouette.** For each point `i`:

```
a(i) = mean distance from i to the other points in its own cluster
b(i) = mean distance from i to the points of the nearest other cluster
s(i) = (b(i) - a(i)) / max(a(i), b(i))          in [-1, 1]
```

`s` near 1: the point is much closer to its own cluster. Near 0: it sits on a boundary. Negative: it is probably in the wrong cluster. The silhouette score is the mean of `s(i)`. Per-cluster and per-point values are often more useful than the average. Cost: `O(n^2)` distances.

**Davies-Bouldin (lower is better).** For each cluster, find the most similar other cluster, where similarity is `(scatter_i + scatter_j) / distance(centroid_i, centroid_j)` and scatter is the mean distance of members to their centroid. Average the worst ratios.

**Calinski-Harabasz (higher is better).** The ratio of between-cluster dispersion to within-cluster dispersion, each divided by its degrees of freedom:

```
CH = [ sum_k n_k ||mu_k - mu||^2 / (K - 1) ]  /  [ sum_k sum_{x in k} ||x - mu_k||^2 / (n - K) ]
```

It is the F-statistic from one-way ANOVA. Fast, but strongly tied to the k-means notion of compact, spherical clusters.

All three reward convex, well-separated blobs. None of them knows what "correct" means for a crescent or a ring.

### External Metrics

Everything starts from the contingency table: rows are true classes, columns are found clusters, cells count points. Cluster ids are arbitrary, so metrics must not care about column order.

```
              cluster 0  cluster 1  cluster 2  cluster 3
  class A         0         95          0          5
  class B         0          7          1         92
  class C         0          0        100          0
  class D       100          0          0          0
```

**Purity.** For each cluster take its majority class count, sum, divide by `n`. Simple, but putting every point in its own cluster gives purity 1.0.

**Adjusted Rand Index.** Look at all pairs of points. A pair is "agreeing" if both clusterings put them together or both keep them apart. The Rand index is the fraction of agreeing pairs; ARI subtracts what a random labeling with the same cluster sizes would get and rescales:

```
index    = sum_ij C(n_ij, 2)
expected = [sum_i C(a_i, 2)] [sum_j C(b_j, 2)] / C(n, 2)
max      = ( sum_i C(a_i, 2) + sum_j C(b_j, 2) ) / 2
ARI      = (index - expected) / (max - expected)
```

`a_i` and `b_j` are the row and column sums. ARI is 1 for identical partitions and about 0 for random ones.

**Normalized Mutual Information.** How much knowing the cluster tells you about the class, in information units:

```
MI  = sum_ij p_ij log( p_ij / (p_i p_j) )
NMI = MI / ( (H(classes) + H(clusters)) / 2 )
```

NMI is 1 for identical partitions and 0 for independent ones. Plain NMI is not adjusted for chance and creeps up when there are many small clusters; Adjusted Mutual Information (AMI) corrects that.

## Build It

### Step 1: Silhouette

```python
def silhouette_samples(X, labels):
    distances = pairwise_distances(X)
    clusters = np.unique(labels)
    scores = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() == 1:
            continue
        a = distances[i, own].sum() / (own.sum() - 1)
        b = min(distances[i, labels == other].mean() for other in clusters if other != labels[i])
        scores[i] = (b - a) / max(a, b)
    return scores
```

Dividing by `own.sum() - 1` excludes the point itself from `a`. Singleton clusters get 0 by convention.

### Step 2: Davies-Bouldin and Calinski-Harabasz

```python
def davies_bouldin(X, labels):
    clusters = np.unique(labels)
    centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])
    scatter = np.array([np.linalg.norm(X[labels == c] - centroids[j], axis=1).mean() for j, c in enumerate(clusters)])
    worst_ratios = []
    for i in range(len(clusters)):
        ratios = [(scatter[i] + scatter[j]) / np.linalg.norm(centroids[i] - centroids[j])
                  for j in range(len(clusters)) if j != i]
        worst_ratios.append(max(ratios))
    return float(np.mean(worst_ratios))


def calinski_harabasz(X, labels):
    clusters = np.unique(labels)
    overall_mean = X.mean(axis=0)
    between = sum(np.sum(labels == c) * np.sum((X[labels == c].mean(axis=0) - overall_mean) ** 2) for c in clusters)
    within = sum(np.sum((X[labels == c] - X[labels == c].mean(axis=0)) ** 2) for c in clusters)
    n, k = len(X), len(clusters)
    return float((between / (k - 1)) / (within / (n - k)))
```

### Step 3: Contingency table, purity, ARI, NMI

```python
def contingency_table(labels_true, labels_pred):
    true_classes, true_index = np.unique(labels_true, return_inverse=True)
    pred_classes, pred_index = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((len(true_classes), len(pred_classes)), dtype=int)
    np.add.at(table, (true_index, pred_index), 1)
    return table


def adjusted_rand_index(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred)
    pairs_cells = sum(comb(int(v), 2) for v in table.ravel())
    pairs_true = sum(comb(int(v), 2) for v in table.sum(axis=1))
    pairs_pred = sum(comb(int(v), 2) for v in table.sum(axis=0))
    expected = pairs_true * pairs_pred / comb(int(table.sum()), 2)
    maximum = (pairs_true + pairs_pred) / 2
    return float((pairs_cells - expected) / (maximum - expected))


def normalized_mutual_information(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred).astype(float)
    joint = table / table.sum()
    outer = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    mutual_information = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / outer[nonzero])))
    h_true, h_pred = entropy(table.sum(axis=1)), entropy(table.sum(axis=0))
    return mutual_information / ((h_true + h_pred) / 2)
```

`np.add.at` increments cells even when the same `(row, column)` index appears many times, which plain fancy indexing would not.

### Step 4: Internal metrics across k

Four blobs with `cluster_std=2.2`, so two of them touch:

```
   k | silhouette | Davies-Bouldin | Calinski-Harabasz
   3 |      0.631 |          0.475 |             847.2
   4 |      0.597 |          0.566 |            1126.7
   5 |      0.538 |          0.764 |             963.8
```

Silhouette and Davies-Bouldin prefer `k = 3`: merging the two touching blobs gives one compact, well-separated group. Calinski-Harabasz prefers the true `k = 4`. Metrics disagree on real-looking data. Treat them as evidence, not verdicts.

### Step 5: External metrics and their traps

```
  purity 0.968 | ARI 0.918 | NMI 0.908
  one cluster per point  | purity 1.000 | ARI 0.000 | NMI 0.376
  random 4 labels        | purity 0.305 | ARI 0.005 | NMI 0.015
```

Putting every point in its own cluster earns perfect purity and a non-trivial NMI, yet ARI correctly says it is worthless. Random labels score near zero on ARI and NMI.

### Step 6: When geometry is not the truth

```
  k-means  | silhouette 0.489 | ARI vs truth 0.241
  DBSCAN   | silhouette 0.328 | ARI vs truth 1.000
```

On the two moons, DBSCAN finds the correct crescents but gets a lower silhouette than k-means, which cuts them in half. Silhouette compares average distances, so a long curved cluster looks "loose" even when it is exactly right. Never use silhouette to choose between algorithms with different shape assumptions.

## Use It

```python
from sklearn import metrics

metrics.silhouette_score(X, labels)
metrics.silhouette_samples(X, labels)
metrics.davies_bouldin_score(X, labels)
metrics.calinski_harabasz_score(X, labels)
metrics.adjusted_rand_score(y_true, labels)
metrics.normalized_mutual_info_score(y_true, labels)
metrics.adjusted_mutual_info_score(y_true, labels)
metrics.cluster.contingency_matrix(y_true, labels)
```

```
  silhouette         | ours     0.5966 | sklearn     0.5966 | match True
  Davies-Bouldin     | ours     0.5656 | sklearn     0.5656 | match True
  Calinski-Harabasz  | ours  1126.7165 | sklearn  1126.7165 | match True
  ARI                | ours     0.9180 | sklearn     0.9180 | match True
  NMI                | ours     0.9084 | sklearn     0.9084 | match True
```

Every number matches. Practical guidance:

- With labels, report ARI (and AMI). Avoid purity on its own.
- Without labels, use silhouette or Davies-Bouldin to compare k for the same algorithm, then look at per-cluster silhouettes and at the clusters themselves.
- `silhouette_score(X, labels, sample_size=2000)` subsamples for large data, since the full computation is quadratic.
- Exclude DBSCAN noise points (`-1`) before computing internal metrics, or they are treated as one big cluster.
- The best evaluation is often downstream: do the segments behave differently on the business metric you care about? Are clusters stable when you rerun on a bootstrap sample?

## Exercises

1. **Easy.** Compute AMI with `metrics.adjusted_mutual_info_score` for the "one cluster per point" labeling. How does it compare with NMI, and why?
2. **Medium.** Implement a stability score: cluster 20 bootstrap samples with k-means for each `k`, compute ARI between each run's labels and a reference run on the shared points, and average. Which `k` is most stable on the 4-blob data?
3. **Hard.** Implement a density-based internal metric: for each cluster, the median distance to the 5th nearest neighbor within the cluster, divided by the minimum distance to any other cluster. Show that it ranks DBSCAN above k-means on the moons.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Internal metric | "Score without labels" | A quality measure computed only from the data and the cluster assignment, usually rewarding compactness and separation |
| External metric | "Score against ground truth" | A measure comparing a clustering with known labels, invariant to how clusters are numbered |
| Silhouette | "How well each point fits" | Mean over points of (b - a) / max(a, b), with a the mean intra-cluster distance and b the mean distance to the nearest other cluster |
| Davies-Bouldin index | "Cluster overlap score" | Average over clusters of the worst (scatter_i + scatter_j) / centroid distance ratio; lower is better |
| Calinski-Harabasz index | "Variance ratio criterion" | Between-cluster dispersion over within-cluster dispersion, each normalized by degrees of freedom; higher is better |
| Contingency table | "Confusion matrix for clusters" | Counts of points for every (true class, found cluster) pair |
| Adjusted Rand Index | "Pair agreement, chance-corrected" | Fraction of point pairs on which two partitions agree, rescaled so random labelings score about 0 and identical ones 1 |
| NMI | "Shared information" | Mutual information between labels and clusters divided by the mean of their entropies |

## Further Reading

- [scikit-learn User Guide: Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) -- every metric here with formulas, advantages and drawbacks
- [Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis" (1987)](https://www.sciencedirect.com/science/article/pii/0377042787901257) -- the original silhouette paper
- [Hubert and Arabie, "Comparing partitions" (1985)](https://link.springer.com/article/10.1007/BF01908075) -- the derivation of the Adjusted Rand Index
- [Vinh, Epps and Bailey, "Information Theoretic Measures for Clusterings Comparison" (JMLR 2010)](https://www.jmlr.org/papers/v11/vinh10a.html) -- NMI, AMI and why chance correction matters
