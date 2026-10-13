import numpy as np
from sklearn.decomposition import TruncatedSVD


def power_iteration(A, n_iterations, rng):
    v = rng.normal(size=A.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(n_iterations):
        v = A.T @ (A @ v)
        v /= np.linalg.norm(v)
    sigma = np.linalg.norm(A @ v)
    u = A @ v / sigma
    return sigma, u, v


def svd_from_scratch(A, rank, n_iterations=500, seed=0):
    rng = np.random.default_rng(seed)
    residual = A.astype(float).copy()
    singular_values, left_vectors, right_vectors = [], [], []
    for _ in range(rank):
        sigma, u, v = power_iteration(residual, n_iterations, rng)
        singular_values.append(sigma)
        left_vectors.append(u)
        right_vectors.append(v)
        residual = residual - sigma * np.outer(u, v)
    return np.array(left_vectors).T, np.array(singular_values), np.array(right_vectors)


def low_rank_approximation(U, S, Vt, k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def align_signs(reference, candidate):
    signs = np.sign(np.sum(reference * candidate, axis=0))
    return candidate * signs


print("=== Step 1: SVD of a small matrix by power iteration ===")
A = np.array([[3.0, 1.0, 1.0], [-1.0, 3.0, 1.0], [2.0, 0.0, 4.0], [0.0, 1.0, 2.0]])
U, S, Vt = svd_from_scratch(A, rank=3)
print(f"Singular values (scratch): {np.round(S, 6)}")
print(f"Singular values (numpy):   {np.round(np.linalg.svd(A, compute_uv=False), 6)}")
print(f"Reconstruction error ||A - U S Vt||: {np.linalg.norm(A - U @ np.diag(S) @ Vt):.2e}")
print(f"U^T U close to identity: {np.allclose(U.T @ U, np.eye(3), atol=1e-6)}")

print("\n=== Step 2: Low-rank image compression ===")
rows, cols = np.mgrid[0:64, 0:64]
image = (
    np.sin(rows / 6.0)
    + np.cos(cols / 9.0)
    + ((rows - 32) ** 2 + (cols - 32) ** 2 < 200).astype(float)
)
rng = np.random.default_rng(1)
image = image + rng.normal(0, 0.05, image.shape)
U_img, S_img, Vt_img = svd_from_scratch(image, rank=20, n_iterations=200)
total_energy = np.sum(np.linalg.svd(image, compute_uv=False) ** 2)
for k in [1, 2, 5, 10, 20]:
    approx = low_rank_approximation(U_img, S_img, Vt_img, k)
    relative_error = np.linalg.norm(image - approx) / np.linalg.norm(image)
    stored = k * (image.shape[0] + image.shape[1] + 1)
    energy = np.sum(S_img[:k] ** 2) / total_energy
    print(f"  rank {k:2d}: relative error={relative_error:.4f}  energy kept={energy:.4f}  numbers stored={stored:5d} / {image.size}")

print("\n=== Step 3: PCA through SVD ===")
rng = np.random.default_rng(2)
latent = rng.normal(size=(300, 2))
mixing = rng.normal(size=(2, 6))
X = latent @ mixing + 0.1 * rng.normal(size=(300, 6))
X_centered = X - X.mean(axis=0)
U_x, S_x, Vt_x = np.linalg.svd(X_centered, full_matrices=False)
variance_from_svd = S_x ** 2 / (X.shape[0] - 1)
eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(X_centered, rowvar=False)))[::-1]
np.set_printoptions(suppress=True)
print(f"Variance from singular values: {np.round(variance_from_svd, 4)}")
print(f"Eigenvalues of covariance:     {np.round(eigenvalues, 4)}")

print("\n=== Step 4: Latent factors in a ratings matrix ===")
users = ["ana", "ben", "carla", "dev", "eli", "fay"]
movies = ["Alien", "Matrix", "Terminator", "Notebook", "Titanic", "Amelie"]
ratings = np.array(
    [
        [5, 5, 4, 1, 1, 2],
        [4, 5, 5, 2, 3, 1],
        [5, 3, 4, 3, 1, 2],
        [1, 2, 2, 5, 4, 4],
        [2, 1, 3, 4, 5, 5],
        [3, 4, 3, 4, 4, 5],
    ],
    dtype=float,
)
ratings_centered = ratings - ratings.mean()
U_r, S_r, Vt_r = svd_from_scratch(ratings_centered, rank=2)
print(f"Top singular values: {np.round(S_r, 3)}")
for movie, weight in zip(movies, Vt_r[0]):
    print(f"  factor 1 loading for {movie:<10}: {weight:+.3f}")
for user, score in zip(users, U_r[:, 0] * S_r[0]):
    print(f"  user {user:<6} factor 1 score: {score:+.3f}")

print("\n=== Use It: numpy and sklearn ===")
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
print(f"numpy U matches scratch U (up to sign): {np.allclose(align_signs(U_np, U), U_np, atol=1e-6)}")
truncated = TruncatedSVD(n_components=2, random_state=0).fit(ratings_centered)
print(f"TruncatedSVD singular values: {np.round(truncated.singular_values_, 3)}")
