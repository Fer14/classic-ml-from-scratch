import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def linear_kernel(A, B):
    return A @ B.T


def polynomial_kernel(A, B, degree=3, coef0=1.0):
    return (A @ B.T + coef0) ** degree


def rbf_kernel(A, B, gamma=1.0):
    squared_distances = (
        np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
    )
    return np.exp(-gamma * np.maximum(squared_distances, 0.0))


def explicit_degree2_features(X):
    x1, x2 = X[:, 0], X[:, 1]
    root2 = np.sqrt(2.0)
    return np.column_stack([np.ones(len(X)), root2 * x1, root2 * x2, x1**2, x2**2, root2 * x1 * x2])


class KernelSVM:
    def __init__(self, kernel, C=1.0, tolerance=1e-3, max_passes=10, seed=0):
        self.kernel = kernel
        self.C = C
        self.tolerance = tolerance
        self.max_passes = max_passes
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        K = self.kernel(X, X)
        alphas = np.zeros(n)
        bias = 0.0
        passes = 0
        while passes < self.max_passes:
            changed = 0
            for i in range(n):
                error_i = (alphas * y) @ K[:, i] + bias - y[i]
                violates_kkt = (y[i] * error_i < -self.tolerance and alphas[i] < self.C) or (
                    y[i] * error_i > self.tolerance and alphas[i] > 0
                )
                if not violates_kkt:
                    continue
                j = rng.integers(n - 1)
                j = j + 1 if j >= i else j
                error_j = (alphas * y) @ K[:, j] + bias - y[j]
                alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                if y[i] != y[j]:
                    low = max(0.0, alpha_j_old - alpha_i_old)
                    high = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    low = max(0.0, alpha_i_old + alpha_j_old - self.C)
                    high = min(self.C, alpha_i_old + alpha_j_old)
                if low == high:
                    continue
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                alphas[j] = np.clip(alpha_j_old - y[j] * (error_i - error_j) / eta, low, high)
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue
                alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])
                bias_i = bias - error_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                bias_j = bias - error_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                if 0 < alphas[i] < self.C:
                    bias = bias_i
                elif 0 < alphas[j] < self.C:
                    bias = bias_j
                else:
                    bias = (bias_i + bias_j) / 2
                changed += 1
            passes = passes + 1 if changed == 0 else 0
        support = alphas > 1e-6
        self.support_vectors = X[support]
        self.support_labels = y[support]
        self.support_alphas = alphas[support]
        self.bias = bias
        return self

    def decision_function(self, X):
        K = self.kernel(X, self.support_vectors)
        return K @ (self.support_alphas * self.support_labels) + self.bias

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)


def to_signed(labels):
    return np.where(labels == 1, 1.0, -1.0)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    print("=== Step 1: A kernel is a dot product in a hidden space ===")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(1, 2)), rng.normal(size=(1, 2))
    explicit = explicit_degree2_features(a) @ explicit_degree2_features(b).T
    implicit = polynomial_kernel(a, b, degree=2, coef0=1.0)
    print(f"phi(a).phi(b) in 6-D space: {explicit[0, 0]:.6f}")
    print(f"(a.b + 1)^2 in 2-D space:   {implicit[0, 0]:.6f}")
    print(f"RBF k(a, a) = {rbf_kernel(a, a)[0, 0]:.3f}, k(a, b) = {rbf_kernel(a, b)[0, 0]:.3f}")

    print("\n=== Step 2: Data no line can separate ===")
    X_circles, labels_circles = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=0)
    X_moons, labels_moons = make_moons(n_samples=200, noise=0.2, random_state=0)
    datasets = {
        "circles": (X_circles, to_signed(labels_circles)),
        "moons": (X_moons, to_signed(labels_moons)),
    }
    kernels = {
        "linear": linear_kernel,
        "poly d=3": lambda A, B: polynomial_kernel(A, B, degree=3, coef0=1.0),
        "rbf g=1": lambda A, B: rbf_kernel(A, B, gamma=1.0),
    }
    for data_name, (X, y) in datasets.items():
        print(f"{data_name}: X {X.shape}, positives {int(np.sum(y == 1))}, negatives {int(np.sum(y == -1))}")

    print("\n=== Step 3: Kernel SVM via simplified SMO ===")
    results = {}
    for data_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        for kernel_name, kernel in kernels.items():
            model = KernelSVM(kernel=kernel, C=1.0, seed=0).fit(X_train, y_train)
            test_acc = accuracy(y_test, model.predict(X_test))
            results[(data_name, kernel_name)] = test_acc
            print(f"{data_name:8} {kernel_name:9} support vectors: {len(model.support_alphas):3d}  test acc: {test_acc:.3f}")

    print("\n=== Step 4: RBF gamma controls smoothness ===")
    X_noisy, labels_noisy = make_moons(n_samples=300, noise=0.35, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, to_signed(labels_noisy), test_size=0.4, random_state=0)
    for gamma in [0.1, 1.0, 10.0, 100.0]:
        model = KernelSVM(kernel=lambda A, B, g=gamma: rbf_kernel(A, B, gamma=g), C=10.0, seed=0).fit(X_train, y_train)
        train_acc = accuracy(y_train, model.predict(X_train))
        test_acc = accuracy(y_test, model.predict(X_test))
        print(f"gamma={gamma:6.1f}  SVs: {len(model.support_alphas):3d}  train: {train_acc:.3f}  test: {test_acc:.3f}")

    print("\n=== Use It: scikit-learn SVC ===")
    sk_kernels = {
        "linear": dict(kernel="linear"),
        "poly d=3": dict(kernel="poly", degree=3, coef0=1.0, gamma=1.0),
        "rbf g=1": dict(kernel="rbf", gamma=1.0),
    }
    for data_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        for kernel_name, params in sk_kernels.items():
            sk_model = SVC(C=1.0, **params).fit(X_train, y_train)
            print(
                f"{data_name:8} {kernel_name:9} sklearn SVs: {len(sk_model.support_):3d}  "
                f"test acc: {sk_model.score(X_test, y_test):.3f}  (scratch {results[(data_name, kernel_name)]:.3f})"
            )


if __name__ == "__main__":
    main()
