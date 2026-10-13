import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from threadpoolctl import threadpool_limits


def make_imbalanced_data(n_majority=1900, n_minority=100, seed=0):
    rng = np.random.default_rng(seed)
    X_majority = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n_majority, 2))
    X_minority = rng.normal(loc=[1.8, 1.8], scale=0.8, size=(n_minority, 2))
    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([np.zeros(n_majority, dtype=int), np.ones(n_minority, dtype=int)])
    order = rng.permutation(len(y))
    return X[order], y[order]


def stratified_split(X, y, test_fraction=0.3, seed=0):
    rng = np.random.default_rng(seed)
    test_idx = []
    for label in np.unique(y):
        label_idx = np.flatnonzero(y == label)
        rng.shuffle(label_idx)
        test_idx.extend(label_idx[: int(len(label_idx) * test_fraction)])
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True
    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]


def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def report(name, y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"{name:<24} acc={accuracy:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


def average_precision(y_true, scores):
    order = np.argsort(-scores)
    sorted_labels = y_true[order]
    true_positives = np.cumsum(sorted_labels)
    precision_at_k = true_positives / np.arange(1, len(scores) + 1)
    return float(np.sum(precision_at_k * sorted_labels) / sorted_labels.sum())


class WeightedLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        weights_normalized = sample_weight / sample_weight.sum()
        self.w = np.zeros(n_features)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self.predict_proba(X)
            error = (p - y) * weights_normalized
            self.w -= self.learning_rate * (X.T @ error)
            self.b -= self.learning_rate * error.sum()
        return self

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def balanced_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = len(y) / (len(classes) * counts)
    return weight_per_class[np.searchsorted(classes, y)]


def random_oversample(X, y, seed=0):
    rng = np.random.default_rng(seed)
    minority_idx = np.flatnonzero(y == 1)
    n_needed = np.sum(y == 0) - len(minority_idx)
    extra = rng.choice(minority_idx, size=n_needed, replace=True)
    return np.vstack([X, X[extra]]), np.concatenate([y, y[extra]])


def random_undersample(X, y, seed=0):
    rng = np.random.default_rng(seed)
    minority_idx = np.flatnonzero(y == 1)
    majority_idx = rng.choice(np.flatnonzero(y == 0), size=len(minority_idx), replace=False)
    keep = np.concatenate([majority_idx, minority_idx])
    return X[keep], y[keep]


def smote(X, y, k_neighbors=5, seed=0):
    rng = np.random.default_rng(seed)
    X_minority = X[y == 1]
    n_needed = np.sum(y == 0) - len(X_minority)
    distances = np.linalg.norm(X_minority[:, None, :] - X_minority[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :k_neighbors]
    base = rng.integers(0, len(X_minority), size=n_needed)
    chosen_neighbor = neighbors[base, rng.integers(0, k_neighbors, size=n_needed)]
    gap = rng.random((n_needed, 1))
    synthetic = X_minority[base] + gap * (X_minority[chosen_neighbor] - X_minority[base])
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_needed, dtype=int)])


def best_f1_threshold(y_true, scores):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        tp, fp, fn, _ = confusion_counts(y_true, (scores >= threshold).astype(int))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold, best_f1


def main():
    X, y = make_imbalanced_data()
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print("=== Data ===")
    print(f"train: {len(y_train)} samples, positives={y_train.sum()} ({y_train.mean():.1%})")
    print(f"test:  {len(y_test)} samples, positives={y_test.sum()} ({y_test.mean():.1%})")

    print("\n=== The accuracy trap ===")
    report("always predict 0", y_test, np.zeros_like(y_test))

    print("\n=== From scratch ===")
    plain = WeightedLogisticRegression().fit(X_train, y_train)
    report("plain logistic", y_test, plain.predict(X_test))

    weighted = WeightedLogisticRegression().fit(X_train, y_train, balanced_class_weights(y_train))
    report("class weights", y_test, weighted.predict(X_test))

    X_over, y_over = random_oversample(X_train, y_train)
    report("random oversample", y_test, WeightedLogisticRegression().fit(X_over, y_over).predict(X_test))

    X_under, y_under = random_undersample(X_train, y_train)
    report("random undersample", y_test, WeightedLogisticRegression().fit(X_under, y_under).predict(X_test))

    X_smote, y_smote = smote(X_train, y_train)
    print(f"  SMOTE grew the training set from {len(y_train)} to {len(y_smote)} rows")
    report("SMOTE", y_test, WeightedLogisticRegression().fit(X_smote, y_smote).predict(X_test))

    train_scores = plain.predict_proba(X_train)
    threshold, train_f1 = best_f1_threshold(y_train, train_scores)
    print(f"  best threshold on train: {threshold:.2f} (train f1={train_f1:.3f})")
    report("plain + tuned threshold", y_test, plain.predict(X_test, threshold=threshold))

    print(f"\naverage precision (threshold-free), plain model: {average_precision(y_test, plain.predict_proba(X_test)):.3f}")

    print("\n=== Use It: scikit-learn ===")
    for name, class_weight in [("sklearn plain", None), ("sklearn balanced", "balanced")]:
        model = SklearnLogisticRegression(class_weight=class_weight).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(
            f"{name:<24} precision={precision_score(y_test, y_pred):.3f} "
            f"recall={recall_score(y_test, y_pred):.3f} f1={f1_score(y_test, y_pred):.3f}"
        )
    sklearn_scores = SklearnLogisticRegression().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print(f"sklearn average_precision_score: {average_precision_score(y_test, sklearn_scores):.3f}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
