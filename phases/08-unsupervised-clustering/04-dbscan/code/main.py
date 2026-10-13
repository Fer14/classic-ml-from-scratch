from collections import deque

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

NOISE = -1
UNVISITED = -2


def pairwise_distances(X):
    return np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        distances = pairwise_distances(X)
        neighborhoods = [np.flatnonzero(row <= self.eps) for row in distances]
        is_core = np.array([len(hood) >= self.min_samples for hood in neighborhoods])
        labels = np.full(len(X), UNVISITED)
        cluster_id = 0
        for point in range(len(X)):
            if labels[point] != UNVISITED or not is_core[point]:
                continue
            labels[point] = cluster_id
            queue = deque([point])
            while queue:
                current = queue.popleft()
                if not is_core[current]:
                    continue
                for neighbor in neighborhoods[current]:
                    if labels[neighbor] == UNVISITED or labels[neighbor] == NOISE:
                        if labels[neighbor] == UNVISITED:
                            queue.append(neighbor)
                        labels[neighbor] = cluster_id
            cluster_id += 1
        labels[labels == UNVISITED] = NOISE
        self.labels_ = labels
        self.core_sample_mask_ = is_core
        self.n_clusters_ = cluster_id
        return self


def k_distance(X, k):
    distances = np.sort(pairwise_distances(X), axis=1)
    return np.sort(distances[:, k - 1])


def summarize(name, labels, y_true):
    n_clusters = len(set(labels.tolist())) - (1 if NOISE in labels else 0)
    n_noise = int(np.sum(labels == NOISE))
    ari = adjusted_rand_score(y_true, labels[: len(y_true)])
    print(f"  {name:22s} | clusters {n_clusters:2d} | noise {n_noise:3d} | ARI vs truth {ari:.3f}")


def composition(labels, y_true):
    rows = []
    for label in sorted(set(labels.tolist())):
        counts = np.bincount(y_true[labels == label], minlength=y_true.max() + 1)
        name = "noise" if label == NOISE else f"cluster {label}"
        rows.append(f"{name}: {counts.tolist()}")
    return " | ".join(rows)


def main():
    rng = np.random.default_rng(0)
    X_moons, y_moons = make_moons(n_samples=300, noise=0.06, random_state=0)
    outliers = rng.uniform([-1.5, -1.0], [2.5, 1.5], size=(20, 2))
    X = np.vstack([X_moons, outliers])
    print("Data: two moons (300 points) + 20 uniform outliers")

    print("\n=== Choosing eps with the k-distance curve (k = min_samples = 5) ===")
    curve = k_distance(X, 5)
    for percentile in [10, 50, 80, 90, 95, 98, 100]:
        print(f"  {percentile:3d}th percentile of 5-NN distance: {np.percentile(curve, percentile):.3f}")
    print("  The curve bends sharply above ~0.15: points beyond it are sparse (outliers)")

    print("\n=== DBSCAN from scratch ===")
    for eps in [0.05, 0.15, 0.4]:
        model = DBSCAN(eps=eps, min_samples=5).fit(X)
        summarize(f"eps={eps}", model.labels_, y_moons)
    model = DBSCAN(eps=0.15, min_samples=5).fit(X)
    found_outliers = np.sum(model.labels_[300:] == NOISE)
    print(f"  With eps=0.15: {found_outliers}/20 injected outliers labelled as noise, "
          f"{int(model.core_sample_mask_.sum())} core points")

    print("\n=== K-means on the same data, for contrast ===")
    kmeans_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    summarize("k-means (k=2)", kmeans_labels, y_moons)

    print("\n=== scikit-learn DBSCAN ===")
    sk_model = SklearnDBSCAN(eps=0.15, min_samples=5).fit(X)
    summarize("sklearn eps=0.15", sk_model.labels_, y_moons)
    same_core = np.array_equal(np.sort(sk_model.core_sample_indices_), np.flatnonzero(model.core_sample_mask_))
    same_noise = np.array_equal(sk_model.labels_ == NOISE, model.labels_ == NOISE)
    print(f"  Same core points: {same_core} | same noise points: {same_noise} | "
          f"ARI between label sets: {adjusted_rand_score(model.labels_, sk_model.labels_):.3f}")

    print("\n=== Varying density: one eps cannot fit all ===")
    dense_left = 0.1 * rng.standard_normal((100, 2))
    dense_right = np.array([0.7, 0.0]) + 0.1 * rng.standard_normal((100, 2))
    sparse = np.array([5.0, 0.0]) + 1.0 * rng.standard_normal((100, 2))
    X_density = np.vstack([dense_left, dense_right, sparse])
    y_density = np.repeat([0, 1, 2], 100)
    for eps in [0.1, 0.5]:
        labels = DBSCAN(eps=eps, min_samples=5).fit(X_density).labels_
        print(f"  DBSCAN eps={eps}: {composition(labels, y_density)}")
    hdbscan_labels = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(X_density)
    print(f"  sklearn HDBSCAN: {composition(hdbscan_labels, y_density)}")
    print("  (counts per true group: [dense left, dense right, sparse])")


if __name__ == "__main__":
    main()
