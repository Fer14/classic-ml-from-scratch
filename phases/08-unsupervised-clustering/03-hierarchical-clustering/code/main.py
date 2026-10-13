import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score


def pairwise_distances(X):
    return np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))


def lance_williams(method, d_ik, d_jk, d_ij, size_i, size_j, size_k):
    if method == "single":
        return np.minimum(d_ik, d_jk)
    if method == "complete":
        return np.maximum(d_ik, d_jk)
    if method == "average":
        return (size_i * d_ik + size_j * d_jk) / (size_i + size_j)
    if method == "ward":
        total = size_i + size_j + size_k
        squared = ((size_i + size_k) * d_ik**2 + (size_j + size_k) * d_jk**2 - size_k * d_ij**2) / total
        return np.sqrt(squared)
    raise ValueError(method)


def agglomerative(X, method):
    n = len(X)
    distances = pairwise_distances(X)
    np.fill_diagonal(distances, np.inf)
    active = np.ones(n, dtype=bool)
    sizes = np.ones(n)
    cluster_ids = np.arange(n)
    merges = []
    for step in range(n - 1):
        masked = np.where(active[:, None] & active[None, :], distances, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        i, j = min(i, j), max(i, j)
        height = distances[i, j]
        others = active.copy()
        others[[i, j]] = False
        updated = lance_williams(method, distances[i, others], distances[j, others], height,
                                 sizes[i], sizes[j], sizes[others])
        distances[i, others] = updated
        distances[others, i] = updated
        merges.append([cluster_ids[i], cluster_ids[j], height, sizes[i] + sizes[j]])
        sizes[i] += sizes[j]
        cluster_ids[i] = n + step
        active[j] = False
    return np.array(merges)


def cut_tree(merges, n, n_clusters):
    parent = list(range(2 * n - 1))

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node

    for step in range(n - n_clusters):
        left, right = int(merges[step, 0]), int(merges[step, 1])
        parent[left] = n + step
        parent[right] = n + step
    roots = [find(point) for point in range(n)]
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def print_dendrogram_tail(merges, count):
    print(f"  Last {count} merges (height = linkage distance):")
    for left, right, height, size in merges[-count:]:
        print(f"    merge {int(left):4d} + {int(right):4d} -> size {int(size):3d} at height {height:7.3f}")


def main():
    rng = np.random.default_rng(0)
    blob_centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
    X_blobs = np.vstack([c + 0.7 * rng.standard_normal((30, 2)) for c in blob_centers])
    y_blobs = np.repeat([0, 1, 2], 30)
    X_moons, y_moons = make_moons(n_samples=120, noise=0.05, random_state=0)

    print("=== Agglomerative clustering from scratch on 3 blobs (90 points) ===")
    ward_merges = agglomerative(X_blobs, "ward")
    print_dendrogram_tail(ward_merges, 5)
    heights = ward_merges[:, 2]
    jumps = np.diff(heights[-6:])
    print(f"  Height jumps between the last merges: {np.round(jumps, 3).tolist()}")
    print("  The big jump before the final two merges says: stop at 3 clusters")

    print("\n=== Linkage comparison (ARI vs true labels) ===")
    print(f"  {'method':9s} | {'blobs':>6s} | {'moons':>6s}")
    for method in ["single", "complete", "average", "ward"]:
        blob_labels = cut_tree(agglomerative(X_blobs, method), len(X_blobs), 3)
        moon_labels = cut_tree(agglomerative(X_moons, method), len(X_moons), 2)
        print(f"  {method:9s} | {adjusted_rand_score(y_blobs, blob_labels):6.3f} | "
              f"{adjusted_rand_score(y_moons, moon_labels):6.3f}")

    print("\n=== Compare with scipy and scikit-learn ===")
    for method in ["single", "complete", "average", "ward"]:
        ours = agglomerative(X_blobs, method)
        theirs = linkage(X_blobs, method=method)
        same_heights = np.allclose(np.sort(ours[:, 2]), np.sort(theirs[:, 2]))
        scipy_labels = fcluster(theirs, t=3, criterion="maxclust")
        sklearn_labels = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(X_blobs)
        our_labels = cut_tree(ours, len(X_blobs), 3)
        print(f"  {method:9s} | heights match scipy: {same_heights} | "
              f"ARI vs scipy {adjusted_rand_score(our_labels, scipy_labels):.3f} | "
              f"ARI vs sklearn {adjusted_rand_score(our_labels, sklearn_labels):.3f}")


if __name__ == "__main__":
    main()
