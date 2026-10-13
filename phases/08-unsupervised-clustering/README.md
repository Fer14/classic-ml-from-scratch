# Phase 08: Unsupervised Clustering

> No labels, no target, just geometry. Clustering finds the groups your data already has.

Most data arrives without labels. This phase teaches you to discover structure in it. You start with k-means and the two questions it leaves open, how to initialize it and how many clusters to ask for, which k-means++, the elbow method and the gap statistic answer. Then you drop k-means' assumptions one by one: hierarchical clustering builds a full tree of merges so you choose the number of clusters afterwards, DBSCAN finds arbitrarily shaped clusters and labels noise, and Gaussian mixtures fitted with EM model elliptical clusters and give soft, probabilistic memberships. The phase ends with the metrics used to judge clusterings with and without ground truth, and the cases where those metrics mislead. Every algorithm is built from scratch in NumPy and checked against scikit-learn or SciPy.

**Prerequisites:** Phase 01 (Linear Algebra), Phase 03 (Probability and Statistics), Phase 04 (ML Fundamentals), Phase 06 (Supervised Classification)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [K-Means](01-k-means/) | Lloyd's algorithm with vectorized distances, inertia tracking, and a demonstration of bad local minima |
| 02 | [K-Means++ and Choosing K](02-k-means-plus-plus-and-choosing-k/) | k-means++ seeding, restarts, the elbow method, and the gap statistic |
| 03 | [Hierarchical Clustering](03-hierarchical-clustering/) | Agglomerative clustering with the Lance-Williams update for four linkages, tree cutting, and a SciPy-identical merge matrix |
| 04 | [DBSCAN](04-dbscan/) | Core/border/noise detection, breadth-first cluster expansion, the k-distance curve, and a comparison with HDBSCAN |
| 05 | [Gaussian Mixtures and EM](05-gaussian-mixtures-and-em/) | EM for full-covariance Gaussian mixtures in log space, soft memberships, and BIC model selection |
| 06 | [Cluster Evaluation](06-cluster-evaluation/) | Silhouette, Davies-Bouldin, Calinski-Harabasz, purity, ARI and NMI, all matching `sklearn.metrics` |

Run the first lesson:

```bash
uv run python phases/08-unsupervised-clustering/01-k-means/code/main.py
```
