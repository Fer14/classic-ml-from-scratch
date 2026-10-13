import numpy as np
from sklearn.datasets import make_classification, make_friedman1
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


class RegressionTree:
    def __init__(self, max_depth=3, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, target):
        self.nodes = []
        self.build(X, target, np.arange(len(target)), depth=0)
        return self

    def best_split(self, X, target, indices):
        n = len(indices)
        best_feature, best_threshold, best_score = None, None, -np.inf
        total_sum = target[indices].sum()
        n_left = np.arange(1, n)
        n_right = n - n_left
        for feature in range(X.shape[1]):
            order = indices[np.argsort(X[indices, feature])]
            values = X[order, feature]
            left_sum = np.cumsum(target[order])[:-1]
            right_sum = total_sum - left_sum
            score = left_sum ** 2 / n_left + right_sum ** 2 / n_right
            valid = (values[:-1] < values[1:]) & (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not valid.any():
                continue
            score = np.where(valid, score, -np.inf)
            i = int(np.argmax(score))
            if score[i] > best_score:
                best_feature, best_threshold, best_score = feature, (values[i] + values[i + 1]) / 2, score[i]
        return best_feature, best_threshold

    def build(self, X, target, indices, depth):
        node_id = len(self.nodes)
        self.nodes.append({"leaf": True, "value": target[indices].mean(), "indices": indices})
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node_id
        feature, threshold = self.best_split(X, target, indices)
        if feature is None:
            return node_id
        mask = X[indices, feature] <= threshold
        left = self.build(X, target, indices[mask], depth + 1)
        right = self.build(X, target, indices[~mask], depth + 1)
        self.nodes[node_id].update({"leaf": False, "feature": feature, "threshold": threshold, "left": left, "right": right})
        return node_id

    def leaves(self):
        return [node for node in self.nodes if node["leaf"]]

    def predict(self, X):
        output = np.empty(len(X))
        for row, x in enumerate(X):
            node = self.nodes[0]
            while not node["leaf"]:
                node = self.nodes[node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]]
            output[row] = node["value"]
        return output


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y, probability):
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))


class GradientBoosting:
    def __init__(self, loss="squared", n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, seed=0):
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.rng = np.random.default_rng(seed)

    def initial_score(self, y):
        if self.loss == "squared":
            return y.mean()
        positive_rate = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        return np.log(positive_rate / (1 - positive_rate))

    def negative_gradient(self, y, score):
        if self.loss == "squared":
            return y - score
        return y - sigmoid(score)

    def update_leaves(self, tree, y, score, residual):
        if self.loss == "squared":
            return
        probability = sigmoid(score)
        hessian = probability * (1 - probability)
        for leaf in tree.leaves():
            rows = leaf["indices"]
            leaf["value"] = residual[rows].sum() / (hessian[rows].sum() + 1e-12)

    def fit(self, X, y, X_eval=None, y_eval=None):
        self.base_score = self.initial_score(y)
        score = np.full(len(y), self.base_score)
        eval_score = np.full(len(y_eval), self.base_score) if y_eval is not None else None
        self.trees, self.eval_history = [], []
        for _ in range(self.n_estimators):
            residual = self.negative_gradient(y, score)
            n_rows = max(1, int(self.subsample * len(y)))
            rows = self.rng.choice(len(y), size=n_rows, replace=False) if self.subsample < 1 else np.arange(len(y))
            tree = RegressionTree(max_depth=self.max_depth).fit(X[rows], residual[rows])
            self.update_leaves(tree, y[rows], score[rows], residual[rows])
            self.trees.append(tree)
            score += self.learning_rate * tree.predict(X)
            if eval_score is not None:
                eval_score += self.learning_rate * tree.predict(X_eval)
                self.eval_history.append(self.evaluate(y_eval, eval_score))
        return self

    def evaluate(self, y, score):
        if self.loss == "squared":
            return np.mean((y - score) ** 2)
        return log_loss(y, sigmoid(score))

    def raw_score(self, X):
        return self.base_score + self.learning_rate * sum(tree.predict(X) for tree in self.trees)

    def predict(self, X):
        score = self.raw_score(X)
        return score if self.loss == "squared" else (sigmoid(score) >= 0.5).astype(int)


def main():
    X, y = make_friedman1(n_samples=1200, n_features=10, noise=1.0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print("=== Step 1: boosting by hand, three rounds on residuals ===")
    prediction = np.full(len(y_train), y_train.mean())
    print(f"round 0: predict the mean {y_train.mean():.3f}  train MSE={np.mean((y_train - prediction) ** 2):.3f}")
    for round_number in range(1, 4):
        residual = y_train - prediction
        tree = RegressionTree(max_depth=3).fit(X_train, residual)
        prediction += 0.5 * tree.predict(X_train)
        print(f"round {round_number}: fit tree to residuals   train MSE={np.mean((y_train - prediction) ** 2):.3f}")

    print("\n=== Step 2: shrinkage (learning rate) on regression ===")
    print(f"variance of y_test (MSE of predicting the mean): {np.var(y_test):.3f}")
    for rate in [1.0, 0.3, 0.1]:
        model = GradientBoosting("squared", n_estimators=200, learning_rate=rate, max_depth=3)
        model.fit(X_train, y_train, X_test, y_test)
        history = np.array(model.eval_history)
        print(f"lr={rate:<4} best test MSE={history.min():.3f} at tree {history.argmin() + 1:3d}  "
              f"after 200 trees={history[-1]:.3f}")

    print("\n=== Step 3: stochastic gradient boosting (subsample) ===")
    for fraction in [1.0, 0.5]:
        model = GradientBoosting("squared", n_estimators=200, learning_rate=0.1, max_depth=3, subsample=fraction, seed=1)
        model.fit(X_train, y_train, X_test, y_test)
        print(f"subsample={fraction:<4} test MSE after 200 trees={model.eval_history[-1]:.3f}")

    print("\n=== Step 4: binary classification with log-loss ===")
    Xc, yc = make_classification(n_samples=1500, n_features=10, n_informative=6, flip_y=0.05, random_state=2)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.3, random_state=2)
    clf = GradientBoosting("log", n_estimators=150, learning_rate=0.1, max_depth=3)
    clf.fit(Xc_train, yc_train, Xc_test, yc_test)
    print(f"initial log-odds: {clf.base_score:.3f}")
    for trees in [1, 10, 50, 150]:
        print(f"trees={trees:3d}  test log-loss={clf.eval_history[trees - 1]:.3f}")
    ours_accuracy = np.mean(clf.predict(Xc_test) == yc_test)
    print(f"test accuracy: {ours_accuracy:.3f}")

    print("\n=== Use It: sklearn GradientBoostingRegressor / Classifier ===")
    ours_reg = GradientBoosting("squared", n_estimators=200, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
    sk_reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_leaf=5, random_state=0)
    sk_reg.fit(X_train, y_train)
    print(f"regression test MSE   ours={np.mean((ours_reg.predict(X_test) - y_test) ** 2):.3f}  "
          f"sklearn={np.mean((sk_reg.predict(X_test) - y_test) ** 2):.3f}")
    sk_clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, min_samples_leaf=5, random_state=0)
    sk_clf.fit(Xc_train, yc_train)
    print(f"classification acc    ours={ours_accuracy:.3f}  sklearn={sk_clf.score(Xc_test, yc_test):.3f}")
    early = GradientBoostingRegressor(
        n_estimators=2000, learning_rate=0.1, max_depth=3, validation_fraction=0.2, n_iter_no_change=20, random_state=0,
    ).fit(X_train, y_train)
    print(f"early stopping kept {early.n_estimators_} of 2000 trees, test MSE={np.mean((early.predict(X_test) - y_test) ** 2):.3f}")


if __name__ == "__main__":
    main()
