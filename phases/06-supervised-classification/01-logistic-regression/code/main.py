import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, log_loss


def make_two_blobs(n_per_class, seed):
    rng = np.random.default_rng(seed)
    class_zero = rng.normal(loc=[-1.5, -1.0], scale=1.2, size=(n_per_class, 2))
    class_one = rng.normal(loc=[1.5, 1.0], scale=1.2, size=(n_per_class, 2))
    X = np.vstack([class_zero, class_one])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    order = rng.permutation(len(y))
    return X[order], y[order]


def train_test_split_simple(X, y, test_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, epochs=2000, l2=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def predict_proba(self, X):
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X, y, print_every=500):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for epoch in range(self.epochs):
            p = self.predict_proba(X)
            error = p - y
            grad_w = X.T @ error / n_samples + self.l2 * self.weights
            grad_b = error.mean()
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            loss = binary_cross_entropy(y, p)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  epoch {epoch:4d} | loss {loss:.4f}")
        return self


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def main():
    X, y = make_two_blobs(n_per_class=200, seed=0)
    X_train, X_test, y_train, y_test = train_test_split_simple(X, y, 0.25, seed=1)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n=== Step 1: the sigmoid squashes scores into probabilities ===")
    for z in [-6, -2, 0, 2, 6]:
        print(f"  sigmoid({z:+d}) = {sigmoid(z):.4f}")

    print("\n=== Step 2: train logistic regression with gradient descent ===")
    model = LogisticRegressionScratch(learning_rate=0.1, epochs=2000)
    model.fit(X_train, y_train)
    print(f"  weights = {np.round(model.weights, 4)}, bias = {model.bias:.4f}")

    print("\n=== Step 3: evaluate ===")
    y_pred = model.predict(X_test)
    p_test = model.predict_proba(X_test)
    tp, tn, fp, fn = confusion_counts(y_test, y_pred)
    print(f"  test accuracy = {accuracy(y_test, y_pred):.4f}")
    print(f"  test log loss = {binary_cross_entropy(y_test, p_test):.4f}")
    print(f"  confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

    print("\n=== Step 4: the decision boundary is a line ===")
    w1, w2 = model.weights
    print(f"  boundary: {w1:.3f}*x1 + {w2:.3f}*x2 + {model.bias:.3f} = 0")
    print(f"  as a line: x2 = {-w1 / w2:.3f}*x1 + {-model.bias / w2:.3f}")

    print("\n=== Step 5: moving the threshold trades precision for recall ===")
    for threshold in [0.3, 0.5, 0.7]:
        y_t = model.predict(X_test, threshold)
        tp, tn, fp, fn = confusion_counts(y_test, y_t)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"  threshold {threshold:.1f} | precision {precision:.3f} | recall {recall:.3f}")

    print("\n=== Use It: scikit-learn ===")
    sk_model = SklearnLogisticRegression(C=1e6, max_iter=1000)
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    sk_proba = sk_model.predict_proba(X_test)[:, 1]
    print(f"  sklearn weights = {np.round(sk_model.coef_[0], 4)}, bias = {sk_model.intercept_[0]:.4f}")
    print(f"  sklearn test accuracy = {accuracy_score(y_test, sk_pred):.4f}")
    print(f"  sklearn test log loss = {log_loss(y_test, sk_proba):.4f}")
    agreement = np.mean(sk_pred == y_pred)
    print(f"  prediction agreement with scratch model = {agreement:.4f}")


if __name__ == "__main__":
    main()
