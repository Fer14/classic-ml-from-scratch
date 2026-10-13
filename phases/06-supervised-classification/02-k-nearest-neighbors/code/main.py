import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


def train_test_split_simple(X, y, test_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def pairwise_euclidean(A, B):
    squared = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(squared, 0.0))


def pairwise_manhattan(A, B):
    return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)


class KNNClassifierScratch:
    def __init__(self, k=5, metric="euclidean", weighted=False):
        self.k = k
        self.metric = metric
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        return self

    def distances(self, X):
        if self.metric == "manhattan":
            return pairwise_manhattan(X, self.X_train)
        return pairwise_euclidean(X, self.X_train)

    def predict(self, X):
        dist = self.distances(X)
        neighbor_idx = np.argsort(dist, axis=1, kind="stable")[:, : self.k]
        neighbor_labels = self.y_train[neighbor_idx]
        neighbor_dist = np.take_along_axis(dist, neighbor_idx, axis=1)
        if self.weighted:
            votes_weight = 1.0 / (neighbor_dist + 1e-9)
        else:
            votes_weight = np.ones_like(neighbor_dist)
        scores = np.zeros((X.shape[0], len(self.classes)))
        for class_position, label in enumerate(self.classes):
            scores[:, class_position] = (votes_weight * (neighbor_labels == label)).sum(axis=1)
        return self.classes[np.argmax(scores, axis=1)]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def k_fold_accuracy(X, y, k, n_folds, seed):
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), n_folds)
    scores = []
    for fold in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[fold] = False
        X_tr, X_va = standardize(X[mask], X[fold])
        model = KNNClassifierScratch(k=k).fit(X_tr, y[mask])
        scores.append(accuracy(y[fold], model.predict(X_va)))
    return round(float(np.mean(scores)), 6)


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split_simple(X, y, 0.3, seed=0)
    print(f"Iris: train {X_train.shape}, test {X_test.shape}, classes {np.unique(y)}")

    print("\n=== Step 1: distance from one test point to every training point ===")
    query = X_test[:1]
    dist = pairwise_euclidean(query, X_train)[0]
    nearest = np.argsort(dist)[:5]
    for idx in nearest:
        print(f"  train #{idx:3d} | distance {dist[idx]:.3f} | label {y_train[idx]}")
    print(f"  true label of query: {y_test[0]}")

    print("\n=== Step 2: scaling matters ===")
    X_train_big = X_train.copy()
    X_test_big = X_test.copy()
    X_train_big[:, 0] *= 1000
    X_test_big[:, 0] *= 1000
    raw = KNNClassifierScratch(k=5).fit(X_train_big, y_train)
    print(f"  sepal length in millimetres x1000, unscaled: accuracy {accuracy(y_test, raw.predict(X_test_big)):.4f}")
    X_tr_s, X_te_s = standardize(X_train_big, X_test_big)
    scaled = KNNClassifierScratch(k=5).fit(X_tr_s, y_train)
    print(f"  same data, standardized:                     accuracy {accuracy(y_test, scaled.predict(X_te_s)):.4f}")

    print("\n=== Step 3: choose k with 5-fold cross-validation ===")
    best_k, best_score = None, -1.0
    for k in [1, 3, 5, 7, 11, 15, 25, 51]:
        score = k_fold_accuracy(X_train, y_train, k, n_folds=5, seed=0)
        print(f"  k={k:2d} | cv accuracy {score:.4f}")
        if score >= best_score:
            best_k, best_score = k, score
    print(f"  best k = {best_k} (ties go to the larger, smoother k)")

    print("\n=== Step 4: final model on the test set ===")
    X_tr_s, X_te_s = standardize(X_train, X_test)
    for metric in ["euclidean", "manhattan"]:
        for weighted in [False, True]:
            model = KNNClassifierScratch(k=best_k, metric=metric, weighted=weighted).fit(X_tr_s, y_train)
            print(f"  {metric:9s} weighted={str(weighted):5s} | test accuracy {accuracy(y_test, model.predict(X_te_s)):.4f}")

    print("\n=== Step 5: the curse of dimensionality ===")
    rng = np.random.default_rng(0)
    for dim in [2, 10, 100, 1000]:
        points = rng.uniform(size=(500, dim))
        d = pairwise_euclidean(points[:1], points[1:])[0]
        print(f"  dim {dim:4d} | nearest/farthest distance ratio {d.min() / d.max():.3f}")

    print("\n=== Use It: scikit-learn ===")
    scratch = KNNClassifierScratch(k=best_k).fit(X_tr_s, y_train)
    start = time.perf_counter()
    scratch_pred = scratch.predict(X_te_s)
    scratch_time = time.perf_counter() - start
    sk_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_tr_s, y_train)
    sk_pred = sk_model.predict(X_te_s)
    print(f"  sklearn test accuracy = {accuracy(y_test, sk_pred):.4f}")
    print(f"  scratch test accuracy = {accuracy(y_test, scratch_pred):.4f}")
    print(f"  agreement = {np.mean(sk_pred == scratch_pred):.4f}")
    print(f"  scratch predict time = {scratch_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
