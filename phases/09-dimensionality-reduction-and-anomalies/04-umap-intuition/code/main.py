import time
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def knn_graph(X, n_neighbors):
    finder = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = finder.kneighbors(X)
    return distances[:, 1:], indices[:, 1:]


def smooth_knn_sigmas(knn_distances, n_iterations=64):
    target = np.log2(knn_distances.shape[1])
    rhos = knn_distances[:, 0]
    sigmas = np.ones(len(rhos))
    for i in range(len(rhos)):
        low, high = 0.0, np.inf
        sigma = 1.0
        for _ in range(n_iterations):
            total = np.sum(np.exp(-np.maximum(knn_distances[i] - rhos[i], 0.0) / sigma))
            if abs(total - target) < 1e-5:
                break
            if total > target:
                high = sigma
                sigma = (low + high) / 2
            else:
                low = sigma
                sigma = sigma * 2 if high == np.inf else (low + high) / 2
        sigmas[i] = sigma
    return rhos, sigmas


def fuzzy_simplicial_graph(X, n_neighbors):
    knn_distances, knn_indices = knn_graph(X, n_neighbors)
    rhos, sigmas = smooth_knn_sigmas(knn_distances)
    n = X.shape[0]
    weights = np.exp(-np.maximum(knn_distances - rhos[:, None], 0.0) / sigmas[:, None])
    directed = np.zeros((n, n))
    directed[np.repeat(np.arange(n), n_neighbors), knn_indices.ravel()] = weights.ravel()
    return directed + directed.T - directed * directed.T


def fit_curve_parameters(min_dist, spread=1.0):
    distances = np.linspace(0, spread * 3, 300)
    target = np.where(distances < min_dist, 1.0, np.exp(-(distances - min_dist) / spread))
    curve = lambda d, a, b: 1.0 / (1.0 + a * d ** (2 * b))
    (a, b), _ = curve_fit(curve, distances, target)
    return a, b


def optimize_layout(graph, initial, a, b, n_epochs=200, negative_rate=5, learning_rate=1.0, seed=0):
    rng = np.random.default_rng(seed)
    heads, tails = np.nonzero(np.triu(graph))
    edge_weights = graph[heads, tails]
    Y = initial.copy()
    n = Y.shape[0]
    for epoch in range(n_epochs):
        alpha = learning_rate * (1.0 - epoch / n_epochs)
        sampled = rng.random(len(heads)) < edge_weights
        h, t = heads[sampled], tails[sampled]
        diff = Y[h] - Y[t]
        dist_sq = np.sum(diff ** 2, axis=1) + 1e-6
        attract = (-2 * a * b * dist_sq ** (b - 1)) / (1 + a * dist_sq ** b)
        step = np.clip(attract[:, None] * diff, -4, 4) * alpha
        np.add.at(Y, h, step)
        np.add.at(Y, t, -step)
        h_neg = np.repeat(h, negative_rate)
        t_neg = rng.integers(0, n, size=len(h_neg))
        diff_neg = Y[h_neg] - Y[t_neg]
        dist_sq_neg = np.sum(diff_neg ** 2, axis=1) + 1e-3
        repel = (2 * b) / (dist_sq_neg * (1 + a * dist_sq_neg ** b))
        step_neg = np.clip(repel[:, None] * diff_neg, -4, 4) * alpha
        np.add.at(Y, h_neg, step_neg)
    return Y


def tiny_umap(X, n_neighbors=15, min_dist=0.1, n_epochs=200, seed=0):
    graph = fuzzy_simplicial_graph(X, n_neighbors)
    initial = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=seed).fit_transform(graph)
    initial = 10 * (initial - initial.min(axis=0)) / (initial.max(axis=0) - initial.min(axis=0))
    a, b = fit_curve_parameters(min_dist)
    return optimize_layout(graph, initial, a, b, n_epochs=n_epochs, seed=seed), graph, (a, b)


def knn_accuracy(embedding, labels):
    return cross_val_score(KNeighborsClassifier(n_neighbors=5), embedding, labels, cv=5).mean()


def centroid_distance_correlation(X, embedding, labels):
    classes = np.unique(labels)
    high = np.array([X[labels == c].mean(axis=0) for c in classes])
    low = np.array([embedding[labels == c].mean(axis=0) for c in classes])
    upper = np.triu_indices(len(classes), k=1)
    high_dist = np.linalg.norm(high[:, None] - high[None, :], axis=2)[upper]
    low_dist = np.linalg.norm(low[:, None] - low[None, :], axis=2)[upper]
    return spearmanr(high_dist, low_dist).correlation


warnings.filterwarnings("ignore", message="Graph is not fully connected")
digits = load_digits()
rng = np.random.default_rng(0)
subset = rng.choice(len(digits.data), size=1000, replace=False)
X = digits.data[subset] / 16.0
labels = digits.target[subset]

print("=== Step 1: Fuzzy k-NN graph ===")
graph = fuzzy_simplicial_graph(X, n_neighbors=15)
print(f"Points: {X.shape[0]}  edges with weight > 0: {np.count_nonzero(np.triu(graph))}")
print(f"Edge weight quantiles (25/50/75%): {np.round(np.quantile(graph[graph > 0], [0.25, 0.5, 0.75]), 3)}")
same_label = labels[:, None] == labels[None, :]
print(f"Share of edge weight connecting same digit: {graph[same_label].sum() / graph.sum():.3f}")

print("\n=== Step 2: Low-dimensional similarity curve ===")
for min_dist in [0.01, 0.1, 0.5]:
    a, b = fit_curve_parameters(min_dist)
    print(f"  min_dist={min_dist:<4}: a={a:.3f}  b={b:.3f}")

print("\n=== Step 3: Layout with attraction and negative sampling ===")
results = {}
start = time.perf_counter()
results["PCA"] = PCA(n_components=2, random_state=0).fit_transform(X)
timings = {"PCA": time.perf_counter() - start}
start = time.perf_counter()
results["tiny UMAP (scratch)"], _, _ = tiny_umap(X)
timings["tiny UMAP (scratch)"] = time.perf_counter() - start
start = time.perf_counter()
results["t-SNE (sklearn)"] = TSNE(n_components=2, perplexity=30.0, init="pca", random_state=0).fit_transform(X)
timings["t-SNE (sklearn)"] = time.perf_counter() - start
for name, seconds in timings.items():
    print(f"  {name:<22} fitted in {seconds:.2f}s")

print("\n=== Step 4: Local vs global structure ===")
print(f"{'method':<22}{'5-NN acc':>10}{'trust@15':>10}{'centroid rho':>14}")
for name, embedding in results.items():
    print(
        f"{name:<22}{knn_accuracy(embedding, labels):>10.3f}"
        f"{trustworthiness(X, embedding, n_neighbors=15):>10.3f}"
        f"{centroid_distance_correlation(X, embedding, labels):>14.3f}"
    )

print("\n=== Step 5: n_neighbors trades local for global ===")
for n_neighbors in [5, 50]:
    embedding, _, _ = tiny_umap(X, n_neighbors=n_neighbors)
    print(
        f"  n_neighbors={n_neighbors:2d}: trust@15={trustworthiness(X, embedding, n_neighbors=15):.3f}"
        f"  centroid rho={centroid_distance_correlation(X, embedding, labels):.3f}"
    )
