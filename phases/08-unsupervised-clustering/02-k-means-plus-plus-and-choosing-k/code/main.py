import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score


def make_blobs(n_per_cluster, centers, spread, seed):
    rng = np.random.default_rng(seed)
    points = [center + spread * rng.standard_normal((n_per_cluster, len(center))) for center in centers]
    return np.vstack(points)


def squared_distances(X, centroids):
    return np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)


def random_init(X, k, rng):
    return X[rng.choice(len(X), k, replace=False)].copy()


def kmeans_plus_plus_init(X, k, rng):
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        nearest_sq_dist = squared_distances(X, np.array(centroids)).min(axis=1)
        probabilities = nearest_sq_dist / nearest_sq_dist.sum()
        centroids.append(X[rng.choice(len(X), p=probabilities)])
    return np.array(centroids)


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


def main():
    centers = np.array([[0, 0], [9, 1], [2, 9], [10, 10], [5, 5]], dtype=float)
    X = make_blobs(80, centers, 0.9, seed=3)
    print(f"Data: {X.shape[0]} points from 5 true clusters")

    print("\n=== Random init vs k-means++ (k=5, 50 single runs each) ===")
    best_possible = kmeans(X, 5, kmeans_plus_plus_init, seed=0, n_init=20)[2]
    for name, init in [("random", random_init), ("k-means++", kmeans_plus_plus_init)]:
        results = np.array([kmeans(X, 5, init, seed)[2] for seed in range(50)])
        bad_runs = np.sum(results > best_possible * 1.01)
        print(f"  {name:10s} | mean inertia {results.mean():8.1f} | worst {results.max():8.1f} | bad runs {bad_runs:2d}/50")
    print(f"  best inertia found: {best_possible:.1f}")

    k_values = list(range(1, 10))
    print("\n=== Elbow method ===")
    inertias = [kmeans(X, k, kmeans_plus_plus_init, seed=0, n_init=5)[2] for k in k_values]
    for k, value in zip(k_values, inertias):
        bar = "#" * int(40 * value / inertias[0])
        print(f"  k={k} | inertia {value:9.1f} | {bar}")
    drops = -np.diff(inertias)
    print("  Relative drop when adding a cluster:")
    for k, drop, previous in zip(k_values[1:], drops, inertias[:-1]):
        print(f"    {k - 1} -> {k}: {100 * drop / previous:5.1f}%")

    print("\n=== Gap statistic ===")
    gaps, errors = gap_statistic(X, k_values, n_references=10, seed=0)
    for k, gap, error in zip(k_values, gaps, errors):
        print(f"  k={k} | gap {gap:6.3f} | s_k {error:5.3f}")
    chosen_k = next(k for i, k in enumerate(k_values[:-1]) if gaps[i] >= gaps[i + 1] - errors[i + 1])
    print(f"k with the largest gap: {k_values[int(np.argmax(gaps))]}")
    print(f"Tibshirani rule (smallest k with Gap(k) >= Gap(k+1) - s(k+1)): {chosen_k}")

    print("\n=== scikit-learn: k-means++ and silhouette ===")
    for k in range(2, 9):
        sk_model = SklearnKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(X)
        score = silhouette_score(X, sk_model.labels_)
        print(f"  k={k} | inertia {sk_model.inertia_:9.1f} | silhouette {score:.3f}")
    sk_five = SklearnKMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
    print(f"sklearn inertia at k=5: {sk_five.inertia_:.1f} (from scratch best: {best_possible:.1f})")


if __name__ == "__main__":
    main()
