import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        covariance = X_centered.T @ X_centered / (X.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        signs = np.sign(eigenvectors[np.abs(eigenvectors).argmax(axis=0), range(eigenvectors.shape[1])])
        eigenvectors = eigenvectors * signs
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / eigenvalues.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


def make_correlated_2d(n_samples, seed):
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 3.0, n_samples)
    noise = rng.normal(0, 0.5, n_samples)
    x1 = latent
    x2 = 0.8 * latent + noise
    return np.column_stack([x1, x2])


print("=== Step 1: PCA on a 2D correlated cloud ===")
X_2d = make_correlated_2d(500, seed=0)
pca_2d = PCAFromScratch(n_components=2).fit(X_2d)
print(f"Principal directions (rows):\n{np.round(pca_2d.components_, 4)}")
print(f"Variance along each direction: {np.round(pca_2d.explained_variance_, 4)}")
print(f"Explained variance ratio:      {np.round(pca_2d.explained_variance_ratio_, 4)}")

print("\n=== Step 2: Project iris (4D) down to 2D ===")
iris = load_iris()
X_iris = iris.data
X_std = (X_iris - X_iris.mean(axis=0)) / X_iris.std(axis=0)
pca_iris = PCAFromScratch(n_components=4).fit(X_std)
cumulative = np.cumsum(pca_iris.explained_variance_ratio_)
for k in range(4):
    print(f"  PC{k + 1}: ratio={pca_iris.explained_variance_ratio_[k]:.4f}  cumulative={cumulative[k]:.4f}")
k_for_95 = int(np.searchsorted(cumulative, 0.95) + 1)
print(f"Components needed for 95% variance: {k_for_95}")

pca_iris_2 = PCAFromScratch(n_components=2).fit(X_std)
Z = pca_iris_2.transform(X_std)
for label, name in enumerate(iris.target_names):
    centroid = Z[iris.target == label].mean(axis=0)
    print(f"  {name:<11} centroid in PC space: ({centroid[0]:+.3f}, {centroid[1]:+.3f})")

print("\n=== Step 3: Reconstruction error vs number of components ===")
for k in range(1, 5):
    model = PCAFromScratch(n_components=k).fit(X_std)
    X_back = model.inverse_transform(model.transform(X_std))
    mse = np.mean((X_std - X_back) ** 2)
    print(f"  k={k}: reconstruction MSE = {mse:.6f}")

print("\n=== Step 4: Loadings of PC1 and PC2 ===")
for name, loading_1, loading_2 in zip(iris.feature_names, pca_iris_2.components_[0], pca_iris_2.components_[1]):
    print(f"  {name:<20} PC1={loading_1:+.3f}  PC2={loading_2:+.3f}")

print("\n=== Use It: sklearn PCA ===")
sk_pca = PCA(n_components=2).fit(X_std)
Z_sklearn = sk_pca.transform(X_std)
sign_fix = np.sign(np.sum(Z_sklearn * Z, axis=0))
print(f"sklearn explained variance ratio: {np.round(sk_pca.explained_variance_ratio_, 4)}")
print(f"scratch explained variance ratio: {np.round(pca_iris_2.explained_variance_ratio_, 4)}")
print(f"Max abs difference in projections (after sign alignment): {np.max(np.abs(Z_sklearn * sign_fix - Z)):.2e}")
