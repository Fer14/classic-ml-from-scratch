from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def squared_distances(X):
    sum_sq = np.sum(X ** 2, axis=1)
    distances = sum_sq[:, None] + sum_sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(distances, 0.0)
    return np.maximum(distances, 0.0)


def conditional_probabilities_for_row(distances_row, beta):
    logits = -distances_row * beta
    logits -= logits.max()
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return probabilities, entropy


def binary_search_sigmas(distances, perplexity, tolerance=1e-5, max_steps=60):
    n = distances.shape[0]
    target_entropy = np.log(perplexity)
    P_conditional = np.zeros((n, n))
    betas = np.ones(n)
    for i in range(n):
        others = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        row = distances[i, others]
        beta_low, beta_high = 0.0, np.inf
        beta = 1.0
        for _ in range(max_steps):
            probabilities, entropy = conditional_probabilities_for_row(row, beta)
            gap = entropy - target_entropy
            if abs(gap) < tolerance:
                break
            if gap > 0:
                beta_low = beta
                beta = beta * 2 if beta_high == np.inf else (beta + beta_high) / 2
            else:
                beta_high = beta
                beta = (beta + beta_low) / 2
        P_conditional[i, others] = probabilities
        betas[i] = beta
    return P_conditional, betas


def joint_probabilities(X, perplexity):
    distances = squared_distances(X)
    P_conditional, betas = binary_search_sigmas(distances, perplexity)
    P = (P_conditional + P_conditional.T) / (2 * X.shape[0])
    return np.maximum(P, 1e-12), betas


def student_t_affinities(Y):
    inverse_kernel = 1.0 / (1.0 + squared_distances(Y))
    np.fill_diagonal(inverse_kernel, 0.0)
    Q = inverse_kernel / inverse_kernel.sum()
    return np.maximum(Q, 1e-12), inverse_kernel


def kl_divergence(P, Q):
    return np.sum(P * np.log(P / Q))


def tsne_from_scratch(X, perplexity=30.0, n_iterations=500, learning_rate=200.0, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    P, betas = joint_probabilities(X, perplexity)
    Y = rng.normal(0, 1e-4, size=(n, 2))
    velocity = np.zeros_like(Y)
    gains = np.ones_like(Y)
    for iteration in range(n_iterations):
        exaggeration = 12.0 if iteration < 100 else 1.0
        momentum = 0.5 if iteration < 100 else 0.8
        Q, inverse_kernel = student_t_affinities(Y)
        attraction_minus_repulsion = (exaggeration * P - Q) * inverse_kernel
        gradient = 4 * (np.diag(attraction_minus_repulsion.sum(axis=1)) - attraction_minus_repulsion) @ Y
        same_sign = np.sign(gradient) == np.sign(velocity)
        gains = np.where(same_sign, gains * 0.8, gains + 0.2)
        gains = np.maximum(gains, 0.01)
        velocity = momentum * velocity - learning_rate * gains * gradient
        Y = Y + velocity
        Y = Y - Y.mean(axis=0)
        if verbose and (iteration % 100 == 0 or iteration == n_iterations - 1):
            print(f"  iter {iteration:3d} | KL(P||Q) = {kl_divergence(P, Q):.4f}")
    return Y, betas


def knn_accuracy(embedding, labels):
    classifier = KNeighborsClassifier(n_neighbors=5)
    return cross_val_score(classifier, embedding, labels, cv=5).mean()


digits = load_digits()
rng = np.random.default_rng(42)
subset = rng.choice(len(digits.data), size=400, replace=False)
X = digits.data[subset] / 16.0
labels = digits.target[subset]

print("=== Step 1: Perplexity calibration ===")
_, betas = joint_probabilities(X, perplexity=30.0)
sigmas = np.sqrt(1.0 / (2.0 * betas))
print(f"sigma per point: min={sigmas.min():.3f}  median={np.median(sigmas):.3f}  max={sigmas.max():.3f}")

print("\n=== Step 2: Optimize the 2D embedding ===")
Y_scratch, _ = tsne_from_scratch(X, perplexity=30.0, n_iterations=500)

print("\n=== Step 3: Compare with PCA ===")
Y_pca = PCA(n_components=2, random_state=0).fit_transform(X)
print(f"5-NN accuracy in 2D  | PCA: {knn_accuracy(Y_pca, labels):.3f}  t-SNE (scratch): {knn_accuracy(Y_scratch, labels):.3f}")
print(f"Trustworthiness k=10 | PCA: {trustworthiness(X, Y_pca, n_neighbors=10):.3f}  t-SNE (scratch): {trustworthiness(X, Y_scratch, n_neighbors=10):.3f}")

print("\n=== Step 4: Perplexity changes the picture ===")
for perplexity in [5, 50]:
    Y_p, _ = tsne_from_scratch(X, perplexity=perplexity, n_iterations=300, verbose=False)
    print(f"  perplexity={perplexity:2d}: 5-NN accuracy={knn_accuracy(Y_p, labels):.3f}  trustworthiness={trustworthiness(X, Y_p, n_neighbors=10):.3f}")

print("\n=== Use It: sklearn TSNE ===")
Y_sklearn = TSNE(n_components=2, perplexity=30.0, init="pca", random_state=0).fit_transform(X)
print(f"sklearn t-SNE | 5-NN accuracy={knn_accuracy(Y_sklearn, labels):.3f}  trustworthiness={trustworthiness(X, Y_sklearn, n_neighbors=10):.3f}")

figure, axes = plt.subplots(1, 3, figsize=(15, 5))
for axis, embedding, title in zip(axes, [Y_pca, Y_scratch, Y_sklearn], ["PCA", "t-SNE (scratch)", "t-SNE (sklearn)"]):
    scatter = axis.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=10)
    axis.set_title(title)
figure.colorbar(scatter, ax=axes, ticks=range(10))
output_path = Path(__file__).parent / "tsne_vs_pca.png"
figure.savefig(output_path, dpi=100)
print(f"\nSaved plot to {output_path.name}")
