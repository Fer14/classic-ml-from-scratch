import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score


def make_blobs(n_per_cluster, centers, spread, seed):
    rng = np.random.default_rng(seed)
    points = []
    labels = []
    for label, center in enumerate(centers):
        points.append(center + spread * rng.standard_normal((n_per_cluster, len(center))))
        labels.append(np.full(n_per_cluster, label))
    return np.vstack(points), np.concatenate(labels)


def squared_distances(X, centroids):
    diff = X[:, None, :] - centroids[None, :, :]
    return np.sum(diff**2, axis=2)


def assign_clusters(X, centroids):
    return np.argmin(squared_distances(X, centroids), axis=1)


def update_centroids(X, labels, old_centroids):
    new_centroids = old_centroids.copy()
    for k in range(len(old_centroids)):
        members = X[labels == k]
        if len(members) > 0:
            new_centroids[k] = members.mean(axis=0)
    return new_centroids


def inertia(X, labels, centroids):
    return float(np.sum((X - centroids[labels]) ** 2))


class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-6, seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.history = []

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        start_indices = rng.choice(len(X), self.n_clusters, replace=False)
        centroids = X[start_indices].copy()
        for iteration in range(self.max_iter):
            labels = assign_clusters(X, centroids)
            new_centroids = update_centroids(X, labels, centroids)
            self.history.append(inertia(X, labels, new_centroids))
            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            if shift < self.tol:
                break
        self.centroids = centroids
        self.labels = assign_clusters(X, centroids)
        self.inertia = inertia(X, self.labels, centroids)
        self.n_iter = iteration + 1
        return self

    def predict(self, X):
        return assign_clusters(X, self.centroids)


def main():
    true_centers = np.array([[0.0, 0.0], [6.0, 1.0], [3.0, 6.0]])
    X, y_true = make_blobs(100, true_centers, 1.0, seed=42)
    print(f"Data: {X.shape[0]} points, {X.shape[1]} features, 3 true groups")

    print("\n=== K-Means from scratch ===")
    model = KMeans(n_clusters=3, seed=7).fit(X)
    for step, value in enumerate(model.history):
        print(f"  Iteration {step + 1:2d} | inertia: {value:9.3f}")
    print(f"Converged in {model.n_iter} iterations, final inertia {model.inertia:.3f}")
    order = np.argsort(model.centroids[:, 0])
    print("Centroids (sorted by x):")
    for centroid in model.centroids[order]:
        print(f"  [{centroid[0]:6.3f}, {centroid[1]:6.3f}]")
    print(f"Cluster sizes: {np.bincount(model.labels).tolist()}")
    print(f"Adjusted Rand Index vs truth: {adjusted_rand_score(y_true, model.labels):.4f}")

    print("\n=== Sensitivity to initialization ===")
    for seed in range(6):
        run = KMeans(n_clusters=3, seed=seed).fit(X)
        print(f"  seed {seed} | iterations {run.n_iter:2d} | inertia {run.inertia:9.3f}")

    new_points = np.array([[0.5, 0.2], [5.5, 1.5], [3.2, 5.0]])
    print(f"\nPredicted clusters for new points: {model.predict(new_points).tolist()}")

    print("\n=== scikit-learn KMeans ===")
    sk_model = SklearnKMeans(n_clusters=3, init="random", n_init=10, random_state=0).fit(X)
    sk_order = np.argsort(sk_model.cluster_centers_[:, 0])
    print("Centroids (sorted by x):")
    for centroid in sk_model.cluster_centers_[sk_order]:
        print(f"  [{centroid[0]:6.3f}, {centroid[1]:6.3f}]")
    print(f"Inertia: {sk_model.inertia_:.3f} (from scratch: {model.inertia:.3f})")
    print(f"Agreement with from-scratch labels (ARI): {adjusted_rand_score(model.labels, sk_model.labels_):.4f}")


if __name__ == "__main__":
    main()
