import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_two_classes(n_per_class, separation, seed):
    rng = np.random.default_rng(seed)
    positives = rng.normal(loc=[separation, separation], scale=1.0, size=(n_per_class, 2))
    negatives = rng.normal(loc=[-separation, -separation], scale=1.0, size=(n_per_class, 2))
    X = np.vstack([positives, negatives])
    y = np.concatenate([np.ones(n_per_class), -np.ones(n_per_class)])
    order = rng.permutation(len(y))
    return X[order], y[order]


def hinge_objective(X, y, weights, bias, C):
    margins = y * (X @ weights + bias)
    return 0.5 * weights @ weights + C * np.sum(np.maximum(0.0, 1.0 - margins))


class LinearSVM:
    def __init__(self, C=1.0, learning_rate=5.0, epochs=20000):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.history = []

    def decision_function(self, X):
        return X @ self.weights + self.bias

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []
        for epoch in range(self.epochs):
            margins = y * self.decision_function(X)
            violators = margins < 1.0
            grad_weights = self.weights - self.C * (y[violators] @ X[violators])
            grad_bias = -self.C * np.sum(y[violators])
            step = self.learning_rate / np.sqrt(epoch + 1)
            self.weights -= step * grad_weights / n_samples
            self.bias -= step * grad_bias / n_samples
            self.history.append(hinge_objective(X, y, self.weights, self.bias, self.C))
        return self

    def support_vector_mask(self, X, y, tolerance=1e-2):
        return y * self.decision_function(X) <= 1.0 + tolerance


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    print("=== Step 1: Data with a little overlap ===")
    X, y = make_two_classes(n_per_class=150, separation=1.2, seed=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n=== Step 2: Hinge loss by hand ===")
    for margin in [2.0, 1.0, 0.5, 0.0, -1.0]:
        print(f"  y*f(x) = {margin:5.1f} -> hinge = {max(0.0, 1.0 - margin):.1f}")

    print("\n=== Step 3: Train linear SVM with subgradient descent ===")
    svm = LinearSVM(C=1.0, learning_rate=5.0, epochs=20000).fit(X_train, y_train)
    for epoch in [0, 10, 100, 1000, 19999]:
        print(f"  epoch {epoch:4d} | objective {svm.history[epoch]:.3f}")
    margin_width = 2.0 / np.linalg.norm(svm.weights)
    print(f"Weights: {np.round(svm.weights, 3)}, bias: {svm.bias:.3f}")
    print(f"Margin width 2/||w||: {margin_width:.3f}")
    print(f"Points on or inside the margin: {svm.support_vector_mask(X_train, y_train).sum()}")
    print(f"Train accuracy: {accuracy(y_train, svm.predict(X_train)):.3f}")
    print(f"Test accuracy:  {accuracy(y_test, svm.predict(X_test)):.3f}")

    print("\n=== Step 4: Effect of C ===")
    print(f"{'C':>8} {'||w||':>8} {'margin':>8} {'in margin':>10} {'test acc':>9}")
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LinearSVM(C=C, learning_rate=5.0, epochs=20000).fit(X_train, y_train)
        norm = np.linalg.norm(model.weights)
        inside = model.support_vector_mask(X_train, y_train).sum()
        test_acc = accuracy(y_test, model.predict(X_test))
        print(f"{C:>8} {norm:>8.3f} {2 / norm:>8.3f} {inside:>10d} {test_acc:>9.3f}")

    print("\n=== Use It: scikit-learn ===")
    sk_linear = LinearSVC(C=1.0, loss="hinge", max_iter=20000, random_state=0).fit(X_train, y_train)
    sk_svc = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
    print(f"Scratch  w = {np.round(svm.weights, 3)}, b = {svm.bias:.3f}")
    print(f"LinearSVC w = {np.round(sk_linear.coef_[0], 3)}, b = {sk_linear.intercept_[0]:.3f}")
    print(f"SVC(linear) w = {np.round(sk_svc.coef_[0], 3)}, b = {sk_svc.intercept_[0]:.3f}")
    print(f"SVC support vectors: {len(sk_svc.support_)}")
    print(f"Scratch objective: {hinge_objective(X_train, y_train, svm.weights, svm.bias, 1.0):.3f}")
    print(f"SVC objective:     {hinge_objective(X_train, y_train, sk_svc.coef_[0], sk_svc.intercept_[0], 1.0):.3f}")
    print(f"SVC test accuracy: {sk_svc.score(X_test, y_test):.3f}")


if __name__ == "__main__":
    main()
