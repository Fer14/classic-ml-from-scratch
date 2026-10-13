import itertools

import numpy as np
from scipy.stats import loguniform
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from threadpoolctl import threadpool_limits


def make_data(n_samples=240, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n_samples, 1))
    y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 0] + rng.normal(scale=0.3, size=n_samples)
    return X, y


class RBFKernelRidge:
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma

    def kernel(self, A, B):
        squared_distances = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
        return np.exp(-self.gamma * squared_distances)

    def fit(self, X, y):
        self.X_train = X
        K = self.kernel(X, X)
        self.dual_coef = np.linalg.solve(K + self.alpha * np.eye(len(X)), y)
        return self

    def predict(self, X):
        return self.kernel(X, self.X_train) @ self.dual_coef


def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def make_folds(n_samples, n_folds=5, seed=0):
    order = np.random.default_rng(seed).permutation(n_samples)
    folds = []
    for test_idx in np.array_split(order, n_folds):
        train_idx = np.setdiff1d(order, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def cross_val_score(params, X, y, folds):
    scores = []
    for train_idx, test_idx in folds:
        model = RBFKernelRidge(**params).fit(X[train_idx], y[train_idx])
        scores.append(r2_score(y[test_idx], model.predict(X[test_idx])))
    return float(np.mean(scores))


def grid_search(param_grid, X, y, folds):
    names = list(param_grid)
    results = []
    for values in itertools.product(*(param_grid[name] for name in names)):
        params = dict(zip(names, values))
        results.append((cross_val_score(params, X, y, folds), params))
    return max(results, key=lambda item: item[0]), results


def random_search(param_ranges, n_trials, X, y, folds, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        params = {
            name: float(10 ** rng.uniform(np.log10(low), np.log10(high)))
            for name, (low, high) in param_ranges.items()
        }
        results.append((cross_val_score(params, X, y, folds), params))
    return max(results, key=lambda item: item[0]), results


def format_params(params):
    return ", ".join(f"{name}={value:.4g}" for name, value in params.items())


def main():
    X, y = make_data()
    X_train, y_train = X[:180], y[:180]
    X_test, y_test = X[180:], y[180:]
    folds = make_folds(len(X_train))
    print(f"=== Data: {len(X_train)} train / {len(X_test)} held-out test ===")

    print("\n=== Why tuning matters ===")
    for params in [{"alpha": 1e-4, "gamma": 100.0}, {"alpha": 10.0, "gamma": 0.01}, {"alpha": 0.1, "gamma": 1.0}]:
        print(f"  {format_params(params):<26} cv R^2 = {cross_val_score(params, X_train, y_train, folds):+.4f}")

    print("\n=== Grid search (6 x 6 = 36 configurations) ===")
    param_grid = {"alpha": np.logspace(-4, 1, 6), "gamma": np.logspace(-2, 3, 6)}
    (grid_best_score, grid_best_params), grid_results = grid_search(param_grid, X_train, y_train, folds)
    print("  cv R^2 table (rows alpha, cols gamma):")
    print("            " + " ".join(f"{g:>8.0e}" for g in param_grid["gamma"]))
    for i, alpha in enumerate(param_grid["alpha"]):
        row = [grid_results[i * len(param_grid["gamma"]) + j][0] for j in range(len(param_grid["gamma"]))]
        print(f"  {alpha:>8.0e}  " + " ".join(f"{score:>8.3f}" for score in row))
    print(f"  best: {format_params(grid_best_params)} cv R^2 = {grid_best_score:.4f}")

    print("\n=== Random search (36 log-uniform trials) ===")
    param_ranges = {"alpha": (1e-4, 10.0), "gamma": (1e-2, 1e3)}
    (random_best_score, random_best_params), random_results = random_search(param_ranges, 36, X_train, y_train, folds)
    running_best = np.maximum.accumulate([score for score, _ in random_results])
    print("  best-so-far after trials 1, 5, 10, 20, 36:", [round(float(running_best[i - 1]), 4) for i in [1, 5, 10, 20, 36]])
    print(f"  best: {format_params(random_best_params)} cv R^2 = {random_best_score:.4f}")

    print("\n=== Final check on the untouched test set ===")
    for name, params in [("grid", grid_best_params), ("random", random_best_params)]:
        model = RBFKernelRidge(**params).fit(X_train, y_train)
        print(f"  {name:<7} test R^2 = {r2_score(y_test, model.predict(X_test)):.4f}")

    print("\n=== Use It: scikit-learn ===")
    grid = GridSearchCV(KernelRidge(kernel="rbf"), param_grid, cv=folds, scoring="r2").fit(X_train, y_train)
    print(f"  GridSearchCV best: {format_params(grid.best_params_)} cv R^2 = {grid.best_score_:.4f}")
    print(f"  matches scratch grid: {np.isclose(grid.best_score_, grid_best_score)}")
    randomized = RandomizedSearchCV(
        KernelRidge(kernel="rbf"),
        {"alpha": loguniform(1e-4, 10.0), "gamma": loguniform(1e-2, 1e3)},
        n_iter=36,
        cv=folds,
        scoring="r2",
        random_state=0,
    ).fit(X_train, y_train)
    print(f"  RandomizedSearchCV best: {format_params(randomized.best_params_)} cv R^2 = {randomized.best_score_:.4f}")
    print(f"  refit model test R^2 = {randomized.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
