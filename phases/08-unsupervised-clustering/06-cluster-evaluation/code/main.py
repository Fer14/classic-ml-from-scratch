from math import comb

import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons


def pairwise_distances(X):
    return np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))


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


def contingency_table(labels_true, labels_pred):
    true_classes, true_index = np.unique(labels_true, return_inverse=True)
    pred_classes, pred_index = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((len(true_classes), len(pred_classes)), dtype=int)
    np.add.at(table, (true_index, pred_index), 1)
    return table


def purity(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred)
    return float(table.max(axis=0).sum() / table.sum())


def adjusted_rand_index(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred)
    pairs_cells = sum(comb(int(v), 2) for v in table.ravel())
    pairs_true = sum(comb(int(v), 2) for v in table.sum(axis=1))
    pairs_pred = sum(comb(int(v), 2) for v in table.sum(axis=0))
    expected = pairs_true * pairs_pred / comb(int(table.sum()), 2)
    maximum = (pairs_true + pairs_pred) / 2
    return float((pairs_cells - expected) / (maximum - expected))


def entropy(counts):
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def normalized_mutual_information(labels_true, labels_pred):
    table = contingency_table(labels_true, labels_pred).astype(float)
    joint = table / table.sum()
    outer = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    mutual_information = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / outer[nonzero])))
    h_true, h_pred = entropy(table.sum(axis=1)), entropy(table.sum(axis=0))
    return mutual_information / ((h_true + h_pred) / 2)


def main():
    X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=2.2, random_state=7)
    print(f"Data: {len(X)} points from 4 Gaussian blobs")

    print("\n=== Internal metrics for k-means, k = 2..7 (from scratch) ===")
    print(f"  {'k':>2s} | {'silhouette':>10s} | {'Davies-Bouldin':>14s} | {'Calinski-Harabasz':>17s}")
    for k in range(2, 8):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        print(f"  {k:2d} | {silhouette_samples(X, labels).mean():10.3f} | "
              f"{davies_bouldin(X, labels):14.3f} | {calinski_harabasz(X, labels):17.1f}")
    print("  Higher silhouette and CH are better; lower DB is better")

    labels_four = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
    per_point = silhouette_samples(X, labels_four)
    print("\n  Per-cluster silhouette at k=4:")
    for c in np.unique(labels_four):
        values = per_point[labels_four == c]
        print(f"    cluster {c} | size {len(values):3d} | mean {values.mean():.3f} | negative {np.sum(values < 0)}")

    print("\n=== External metrics against the true labels (from scratch) ===")
    print("  Contingency table (rows = true class, columns = cluster):")
    for row in contingency_table(y_true, labels_four):
        print(f"    {row.tolist()}")
    print(f"  purity {purity(y_true, labels_four):.3f} | ARI {adjusted_rand_index(y_true, labels_four):.3f} | "
          f"NMI {normalized_mutual_information(y_true, labels_four):.3f}")

    print("\n=== Why purity alone is misleading ===")
    singletons = np.arange(len(X))
    random_labels = np.random.default_rng(0).integers(0, 4, len(X))
    for name, labels in [("one cluster per point", singletons), ("random 4 labels", random_labels)]:
        print(f"  {name:22s} | purity {purity(y_true, labels):.3f} | ARI {adjusted_rand_index(y_true, labels):.3f} | "
              f"NMI {normalized_mutual_information(y_true, labels):.3f}")

    print("\n=== Internal metrics can prefer the wrong answer ===")
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=0)
    kmeans_moons = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X_moons)
    dbscan_moons = DBSCAN(eps=0.2, min_samples=5).fit_predict(X_moons)
    for name, labels in [("k-means", kmeans_moons), ("DBSCAN", dbscan_moons)]:
        print(f"  {name:8s} | silhouette {silhouette_samples(X_moons, labels).mean():.3f} | "
              f"ARI vs truth {adjusted_rand_index(y_moons, labels):.3f}")

    print("\n=== Compare with sklearn.metrics ===")
    checks = [
        ("silhouette", silhouette_samples(X, labels_four).mean(), metrics.silhouette_score(X, labels_four)),
        ("Davies-Bouldin", davies_bouldin(X, labels_four), metrics.davies_bouldin_score(X, labels_four)),
        ("Calinski-Harabasz", calinski_harabasz(X, labels_four), metrics.calinski_harabasz_score(X, labels_four)),
        ("ARI", adjusted_rand_index(y_true, labels_four), metrics.adjusted_rand_score(y_true, labels_four)),
        ("NMI", normalized_mutual_information(y_true, labels_four),
         metrics.normalized_mutual_info_score(y_true, labels_four)),
    ]
    for name, ours, theirs in checks:
        print(f"  {name:18s} | ours {ours:10.4f} | sklearn {theirs:10.4f} | match {np.isclose(ours, theirs)}")


if __name__ == "__main__":
    main()
