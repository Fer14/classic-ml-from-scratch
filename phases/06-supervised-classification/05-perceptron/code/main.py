import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.model_selection import train_test_split


def make_blobs_2d(n_per_class, separation, seed):
    rng = np.random.default_rng(seed)
    positives = rng.normal(loc=[separation, separation], scale=1.0, size=(n_per_class, 2))
    negatives = rng.normal(loc=[-separation, -separation], scale=1.0, size=(n_per_class, 2))
    X = np.vstack([positives, negatives])
    y = np.concatenate([np.ones(n_per_class), -np.ones(n_per_class)])
    order = rng.permutation(len(y))
    return X[order], y[order]


class Perceptron:
    def __init__(self, learning_rate=1.0, max_epochs=100, seed=0):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.seed = seed
        self.weights = None
        self.bias = 0.0
        self.mistakes_per_epoch = []

    def decision_function(self, X):
        return X @ self.weights + self.bias

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.mistakes_per_epoch = []
        for _ in range(self.max_epochs):
            mistakes = 0
            for i in rng.permutation(len(y)):
                if y[i] * (X[i] @ self.weights + self.bias) <= 0:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    mistakes += 1
            self.mistakes_per_epoch.append(mistakes)
            if mistakes == 0:
                break
        return self


class AveragedPerceptron(Perceptron):
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        weights = np.zeros(X.shape[1])
        bias = 0.0
        weights_sum = np.zeros(X.shape[1])
        bias_sum = 0.0
        steps = 0
        self.mistakes_per_epoch = []
        for _ in range(self.max_epochs):
            mistakes = 0
            for i in rng.permutation(len(y)):
                if y[i] * (X[i] @ weights + bias) <= 0:
                    weights += self.learning_rate * y[i] * X[i]
                    bias += self.learning_rate * y[i]
                    mistakes += 1
                weights_sum += weights
                bias_sum += bias
                steps += 1
            self.mistakes_per_epoch.append(mistakes)
        self.weights = weights_sum / steps
        self.bias = bias_sum / steps
        return self


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def novikoff_bound(X, y, weights, bias):
    augmented = np.hstack([X, np.ones((len(X), 1))])
    direction = np.append(weights, bias)
    direction = direction / np.linalg.norm(direction)
    radius = np.max(np.linalg.norm(augmented, axis=1))
    margin = np.min(y * (augmented @ direction))
    return radius, margin, (radius / margin) ** 2


def main():
    print("=== Step 1: Linearly separable data ===")
    X, y = make_blobs_2d(n_per_class=100, separation=2.0, seed=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n=== Step 2: Train the perceptron ===")
    model = Perceptron(max_epochs=100, seed=0).fit(X_train, y_train)
    print(f"Mistakes per epoch: {model.mistakes_per_epoch}")
    print(f"Converged after {len(model.mistakes_per_epoch)} epochs")
    print(f"Weights: {np.round(model.weights, 3)}, bias: {model.bias:.3f}")
    print(f"Train accuracy: {accuracy(y_train, model.predict(X_train)):.3f}")
    print(f"Test accuracy:  {accuracy(y_test, model.predict(X_test)):.3f}")

    print("\n=== Step 3: Convergence bound (Novikoff) ===")
    total_mistakes = sum(model.mistakes_per_epoch)
    radius, margin, bound = novikoff_bound(X_train, y_train, model.weights, model.bias)
    print(f"R = {radius:.3f}, margin of found separator = {margin:.4f}")
    print(f"Bound (R/gamma)^2 = {bound:.1f}, actual total mistakes = {total_mistakes}")

    print("\n=== Step 4: Overlapping classes (not separable) ===")
    X_noisy, y_noisy = make_blobs_2d(n_per_class=100, separation=0.8, seed=7)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_noisy, y_noisy, test_size=0.3, random_state=7)
    plain = Perceptron(max_epochs=50, seed=0).fit(Xn_train, yn_train)
    averaged = AveragedPerceptron(max_epochs=50, seed=0).fit(Xn_train, yn_train)
    print(f"Plain perceptron mistakes in last 5 epochs: {plain.mistakes_per_epoch[-5:]}")
    print(f"Plain perceptron test accuracy:    {accuracy(yn_test, plain.predict(Xn_test)):.3f}")
    print(f"Averaged perceptron test accuracy: {accuracy(yn_test, averaged.predict(Xn_test)):.3f}")

    print("\n=== Step 5: XOR is impossible ===")
    X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_xor = np.array([-1.0, 1.0, 1.0, -1.0])
    xor_model = Perceptron(max_epochs=1000, seed=0).fit(X_xor, y_xor)
    print(f"Epochs run: {len(xor_model.mistakes_per_epoch)}, mistakes in last epoch: {xor_model.mistakes_per_epoch[-1]}")
    print(f"XOR accuracy: {accuracy(y_xor, xor_model.predict(X_xor)):.2f}")
    X_xor_lifted = np.hstack([X_xor, (X_xor[:, 0] * X_xor[:, 1])[:, None]])
    lifted_model = Perceptron(max_epochs=1000, seed=0).fit(X_xor_lifted, y_xor)
    print(f"With extra feature x1*x2: accuracy {accuracy(y_xor, lifted_model.predict(X_xor_lifted)):.2f} after {len(lifted_model.mistakes_per_epoch)} epochs")

    print("\n=== Use It: scikit-learn Perceptron ===")
    sk_model = SklearnPerceptron(max_iter=100, random_state=0, tol=None)
    sk_model.fit(X_train, y_train)
    print(f"sklearn test accuracy (separable): {sk_model.score(X_test, y_test):.3f}")
    sk_noisy = SklearnPerceptron(max_iter=50, random_state=0, tol=None).fit(Xn_train, yn_train)
    print(f"sklearn test accuracy (overlapping): {sk_noisy.score(Xn_test, yn_test):.3f}")


if __name__ == "__main__":
    main()
