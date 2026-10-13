import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnTreeRegressor


class Node:
    def __init__(self, value):
        self.value = value
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


def best_split(X, Y, min_samples_leaf):
    n_samples, n_features = X.shape
    parent_sum = Y.sum(axis=0)
    parent_sse = np.sum((Y - parent_sum / n_samples) ** 2)
    best_gain, best_feature, best_threshold = 1e-12, None, None
    left_n = np.arange(1, n_samples)
    right_n = n_samples - left_n
    for feature in range(n_features):
        order = np.argsort(X[:, feature], kind="stable")
        sorted_values = X[order, feature]
        sorted_Y = Y[order]
        left_sum = np.cumsum(sorted_Y, axis=0)[:-1]
        left_sq = np.cumsum(sorted_Y**2, axis=0)[:-1]
        total_sq = np.sum(sorted_Y**2, axis=0)
        left_sse = np.sum(left_sq - left_sum**2 / left_n[:, None], axis=1)
        right_sum = parent_sum - left_sum
        right_sse = np.sum((total_sq - left_sq) - right_sum**2 / right_n[:, None], axis=1)
        valid = (sorted_values[:-1] < sorted_values[1:]) & (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
        gains = np.where(valid, parent_sse - left_sse - right_sse, -np.inf)
        index = int(np.argmax(gains))
        if gains[index] > best_gain:
            best_gain = float(gains[index])
            best_feature = feature
            best_threshold = (sorted_values[index] + sorted_values[index + 1]) / 2
    return best_feature, best_threshold


class Tree:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, Y):
        self.root = self.grow(X, Y, 0)
        return self

    def grow(self, X, Y, depth):
        node = Node(Y.mean(axis=0))
        if len(Y) < 2 * self.min_samples_leaf or (self.max_depth is not None and depth >= self.max_depth):
            return node
        feature, threshold = best_split(X, Y, self.min_samples_leaf)
        if feature is None:
            return node
        mask = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self.grow(X[mask], Y[mask], depth + 1)
        node.right = self.grow(X[~mask], Y[~mask], depth + 1)
        return node

    def predict(self, X):
        output = np.empty((len(X), len(self.root.value)))
        self.fill(self.root, X, np.arange(len(X)), output)
        return output

    def fill(self, node, X, rows, output):
        if node.left is None:
            output[rows] = node.value
            return
        goes_left = X[rows, node.feature] <= node.threshold
        self.fill(node.left, X, rows[goes_left], output)
        self.fill(node.right, X, rows[~goes_left], output)


def bootstrap_indices(n_samples, rng):
    return rng.integers(0, n_samples, n_samples)


class Bagging:
    def __init__(self, n_estimators=50, max_depth=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, Y):
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        self.trees = []
        oob_sum = np.zeros(Y.shape)
        oob_count = np.zeros(n_samples)
        for _ in range(self.n_estimators):
            indices = bootstrap_indices(n_samples, rng)
            tree = Tree(max_depth=self.max_depth).fit(X[indices], Y[indices])
            self.trees.append(tree)
            out_of_bag = np.setdiff1d(np.arange(n_samples), indices)
            oob_sum[out_of_bag] += tree.predict(X[out_of_bag])
            oob_count[out_of_bag] += 1
        covered = oob_count > 0
        self.oob_prediction = np.full(Y.shape, np.nan)
        self.oob_prediction[covered] = oob_sum[covered] / oob_count[covered, None]
        return self

    def predict(self, X, n_trees=None):
        trees = self.trees[: n_trees or len(self.trees)]
        return np.mean([tree.predict(X) for tree in trees], axis=0)


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]


def accuracy_from_scores(y, scores):
    return float(np.mean(np.argmax(scores, axis=1) == y))


def true_function(x):
    return np.sin(2 * x) + 0.5 * x


def bias_variance(make_model, n_repeats, x_grid, rng):
    predictions = []
    for _ in range(n_repeats):
        x = rng.uniform(0, 4, 100)
        y = true_function(x) + rng.normal(0, 0.4, 100)
        model = make_model().fit(x.reshape(-1, 1), y.reshape(-1, 1))
        predictions.append(model.predict(x_grid.reshape(-1, 1))[:, 0])
    predictions = np.array(predictions)
    mean_prediction = predictions.mean(axis=0)
    bias_squared = np.mean((mean_prediction - true_function(x_grid)) ** 2)
    variance = np.mean(predictions.var(axis=0))
    return bias_squared, variance


def main():
    print("=== Step 1: A bootstrap sample contains about 63.2% of the rows ===")
    rng = np.random.default_rng(0)
    for n_samples in [10, 100, 10000]:
        unique_fractions = [len(np.unique(bootstrap_indices(n_samples, rng))) / n_samples for _ in range(200)]
        print(f"  n={n_samples:6d}  mean unique fraction={np.mean(unique_fractions):.3f}")
    print(f"  limit 1 - 1/e = {1 - np.exp(-1):.3f}")

    print("\n=== Step 2: Bagging reduces variance, not bias ===")
    x_grid = np.linspace(0.1, 3.9, 200)
    rows = [
        ("single deep tree", lambda: Tree()),
        ("single stump (depth 1)", lambda: Tree(max_depth=1)),
        ("bagged deep trees (30)", lambda: Bagging(n_estimators=30)),
        ("bagged stumps (30)", lambda: Bagging(n_estimators=30, max_depth=1)),
    ]
    for name, make_model in rows:
        bias_squared, variance = bias_variance(make_model, 20, x_grid, np.random.default_rng(1))
        print(f"  {name:24s} bias^2={bias_squared:.4f}  variance={variance:.4f}  sum={bias_squared + variance:.4f}")

    print("\n=== Step 3: Bagged classification trees with out-of-bag error ===")
    X, y = make_moons(n_samples=600, noise=0.3, random_state=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    Y_train = one_hot(y_train, 2)
    single = Tree().fit(X_train, Y_train)
    print(f"  single deep tree       test_acc={accuracy_from_scores(y_test, single.predict(X_test)):.3f}")
    bagger = Bagging(n_estimators=100, random_state=0).fit(X_train, Y_train)
    for n_trees in [1, 5, 10, 25, 50, 100]:
        acc = accuracy_from_scores(y_test, bagger.predict(X_test, n_trees))
        print(f"  bagging with {n_trees:3d} trees  test_acc={acc:.3f}")
    print(f"  OOB accuracy (100 trees)  = {accuracy_from_scores(y_train, bagger.oob_prediction):.3f}")

    print("\n=== Use It: scikit-learn BaggingClassifier / BaggingRegressor ===")
    sklearn_bagger = BaggingClassifier(
        estimator=SklearnTreeClassifier(), n_estimators=100, oob_score=True, random_state=0
    ).fit(X_train, y_train)
    print(f"  sklearn single tree    test_acc={SklearnTreeClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test):.3f}")
    print(f"  sklearn bagging        test_acc={sklearn_bagger.score(X_test, y_test):.3f}  oob_acc={sklearn_bagger.oob_score_:.3f}")
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 4, 300).reshape(-1, 1)
    y_reg = true_function(x[:, 0]) + rng.normal(0, 0.4, 300)
    x_test = rng.uniform(0, 4, 300).reshape(-1, 1)
    y_reg_test = true_function(x_test[:, 0]) + rng.normal(0, 0.4, 300)
    tree_mse = np.mean((SklearnTreeRegressor(random_state=0).fit(x, y_reg).predict(x_test) - y_reg_test) ** 2)
    bag_mse = np.mean((BaggingRegressor(n_estimators=100, random_state=0).fit(x, y_reg).predict(x_test) - y_reg_test) ** 2)
    ours_mse = np.mean((Bagging(n_estimators=100).fit(x, y_reg.reshape(-1, 1)).predict(x_test)[:, 0] - y_reg_test) ** 2)
    print(f"  regression test_mse: sklearn tree={tree_mse:.4f}  sklearn bagging={bag_mse:.4f}  our bagging={ours_mse:.4f}  noise floor=0.1600")


if __name__ == "__main__":
    main()
