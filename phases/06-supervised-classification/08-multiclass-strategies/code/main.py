from itertools import combinations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class BinaryLogistic:
    def __init__(self, learning_rate=0.5, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y01):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.epochs):
            error = sigmoid(X @ self.weights + self.bias) - y01
            self.weights -= self.learning_rate * (X.T @ error) / len(y01)
            self.bias -= self.learning_rate * error.mean()
        return self

    def decision_function(self, X):
        return X @ self.weights + self.bias


class OneVsRest:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = [BinaryLogistic().fit(X, (y == c).astype(float)) for c in self.classes]
        return self

    def predict(self, X):
        scores = np.column_stack([model.decision_function(X) for model in self.models])
        return self.classes[np.argmax(scores, axis=1)]


class OneVsOne:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.pairs = []
        for class_a, class_b in combinations(self.classes, 2):
            mask = (y == class_a) | (y == class_b)
            model = BinaryLogistic().fit(X[mask], (y[mask] == class_a).astype(float))
            self.pairs.append((class_a, class_b, model))
        return self

    def predict(self, X):
        votes = np.zeros((len(X), len(self.classes)))
        index_of = {c: i for i, c in enumerate(self.classes)}
        for class_a, class_b, model in self.pairs:
            prefers_a = model.decision_function(X) >= 0
            votes[prefers_a, index_of[class_a]] += 1
            votes[~prefers_a, index_of[class_b]] += 1
        return self.classes[np.argmax(votes, axis=1)]


class SoftmaxRegression:
    def __init__(self, learning_rate=0.5, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        one_hot = (y[:, None] == self.classes[None, :]).astype(float)
        self.weights = np.zeros((n_features, len(self.classes)))
        self.bias = np.zeros(len(self.classes))
        for _ in range(self.epochs):
            probabilities = softmax(X @ self.weights + self.bias)
            self.loss_history.append(-np.mean(np.sum(one_hot * np.log(probabilities + 1e-12), axis=1)))
            error = probabilities - one_hot
            self.weights -= self.learning_rate * (X.T @ error) / n_samples
            self.bias -= self.learning_rate * error.mean(axis=0)
        return self

    def predict_proba(self, X):
        return softmax(X @ self.weights + self.bias)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def main():
    print("=== Step 1: A 4-class problem ===")
    X, y = make_classification(
        n_samples=800, n_features=6, n_informative=4, n_redundant=0,
        n_classes=4, n_clusters_per_class=1, class_sep=1.2, random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    print(f"Train: {X_train.shape}, classes: {np.bincount(y_train)}")

    print("\n=== Step 2: Softmax turns scores into probabilities ===")
    example_scores = np.array([[2.0, 1.0, 0.1, -1.0]])
    print(f"scores {example_scores[0]} -> probabilities {np.round(softmax(example_scores)[0], 3)}")

    print("\n=== Step 3: One-vs-Rest ===")
    ovr = OneVsRest().fit(X_train, y_train)
    print(f"Binary models trained: {len(ovr.models)}")
    print(f"Test accuracy: {accuracy(y_test, ovr.predict(X_test)):.3f}")

    print("\n=== Step 4: One-vs-One ===")
    ovo = OneVsOne().fit(X_train, y_train)
    print(f"Binary models trained: {len(ovo.pairs)} (K*(K-1)/2 with K=4)")
    print(f"Test accuracy: {accuracy(y_test, ovo.predict(X_test)):.3f}")

    print("\n=== Step 5: Softmax regression ===")
    soft = SoftmaxRegression().fit(X_train, y_train)
    print(f"Cross-entropy: start {soft.loss_history[0]:.3f} -> end {soft.loss_history[-1]:.3f}")
    print(f"Test accuracy: {accuracy(y_test, soft.predict(X_test)):.3f}")
    print(f"First test row probabilities: {np.round(soft.predict_proba(X_test[:1])[0], 3)} (sum = {soft.predict_proba(X_test[:1]).sum():.3f})")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, soft.predict(X_test), 4))

    print("\n=== Use It: scikit-learn ===")
    base = LogisticRegression(max_iter=1000)
    sk_ovr = OneVsRestClassifier(base).fit(X_train, y_train)
    sk_ovo = OneVsOneClassifier(base).fit(X_train, y_train)
    sk_softmax = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    print(f"OneVsRestClassifier: {len(sk_ovr.estimators_)} models, test acc {sk_ovr.score(X_test, y_test):.3f}")
    print(f"OneVsOneClassifier:  {len(sk_ovo.estimators_)} models, test acc {sk_ovo.score(X_test, y_test):.3f}")
    print(f"LogisticRegression (multinomial): test acc {sk_softmax.score(X_test, y_test):.3f}")
    agreement = np.mean(sk_softmax.predict(X_test) == soft.predict(X_test))
    print(f"Agreement between scratch softmax and sklearn: {agreement:.3f}")


if __name__ == "__main__":
    main()
