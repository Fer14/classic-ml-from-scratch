import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnTreeRegressor
from sklearn.tree import export_text


def gini(labels):
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    return 1.0 - np.sum(proportions**2)


def variance(targets):
    if len(targets) == 0:
        return 0.0
    return float(np.var(targets))


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


def best_split(X, y, impurity):
    n_samples, n_features = X.shape
    parent_impurity = impurity(y)
    best = {"gain": 0.0, "feature": None, "threshold": None}
    for feature in range(n_features):
        values = np.unique(X[:, feature])
        thresholds = (values[:-1] + values[1:]) / 2
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            y_left, y_right = y[left_mask], y[~left_mask]
            weighted = (len(y_left) * impurity(y_left) + len(y_right) * impurity(y_right)) / n_samples
            gain = parent_impurity - weighted
            if gain > best["gain"]:
                best = {"gain": gain, "feature": feature, "threshold": threshold}
    return best


class DecisionTree:
    def __init__(self, impurity, leaf_value, max_depth=5, min_samples_split=2):
        self.impurity = impurity
        self.leaf_value = leaf_value
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self.grow(X, y, depth=0)
        return self

    def grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or self.impurity(y) == 0:
            return Node(value=self.leaf_value(y))
        split = best_split(X, y, self.impurity)
        if split["feature"] is None:
            return Node(value=self.leaf_value(y))
        left_mask = X[:, split["feature"]] <= split["threshold"]
        left = self.grow(X[left_mask], y[left_mask], depth + 1)
        right = self.grow(X[~left_mask], y[~left_mask], depth + 1)
        return Node(split["feature"], split["threshold"], left, right)

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def depth(self, node=None):
        node = node or self.root
        if node.is_leaf():
            return 0
        return 1 + max(self.depth(node.left), self.depth(node.right))

    def n_leaves(self, node=None):
        node = node or self.root
        if node.is_leaf():
            return 1
        return self.n_leaves(node.left) + self.n_leaves(node.right)

    def print_tree(self, feature_names, node=None, indent=""):
        node = node or self.root
        if node.is_leaf():
            print(f"{indent}predict {node.value:.3f}" if isinstance(node.value, float) else f"{indent}predict {node.value}")
            return
        name = feature_names[node.feature]
        print(f"{indent}if {name} <= {node.threshold:.3f}:")
        self.print_tree(feature_names, node.left, indent + "    ")
        print(f"{indent}else:")
        self.print_tree(feature_names, node.right, indent + "    ")


def majority_class(labels):
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def mean_value(targets):
    return float(np.mean(targets))


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=5, min_samples_split=2):
        super().__init__(gini, majority_class, max_depth, min_samples_split)


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth=5, min_samples_split=2):
        super().__init__(variance, mean_value, max_depth, min_samples_split)


def main():
    print("=== Step 1: Gini impurity of a few nodes ===")
    for labels in [np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1])]:
        print(f"  labels={labels.tolist()}  gini={gini(labels):.3f}")

    print("\n=== Step 2: Best first split on a toy dataset ===")
    X_toy = np.array([[2.0], [3.0], [10.0], [19.0], [20.0], [21.0]])
    y_toy = np.array([0, 0, 0, 1, 1, 1])
    split = best_split(X_toy, y_toy, gini)
    print(f"  feature={split['feature']}  threshold={split['threshold']:.1f}  gain={split['gain']:.3f}")

    print("\n=== Step 3: Classification tree on two moons ===")
    X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    for max_depth in [1, 3, 5, 10]:
        tree = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
        train_acc = accuracy_score(y_train, tree.predict(X_train))
        test_acc = accuracy_score(y_test, tree.predict(X_test))
        print(f"  max_depth={max_depth:2d}  leaves={tree.n_leaves():3d}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    shallow = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
    print("\n  Learned rules (max_depth=2):")
    shallow.print_tree(["x0", "x1"], indent="  ")

    print("\n=== Step 4: Regression tree on a noisy sine ===")
    rng = np.random.default_rng(0)
    X_reg = np.sort(rng.uniform(0, 6, 200)).reshape(-1, 1)
    y_reg = np.sin(X_reg[:, 0]) + rng.normal(0, 0.2, 200)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.25, random_state=0)
    for max_depth in [1, 2, 4, 8]:
        reg = DecisionTreeRegressor(max_depth=max_depth).fit(Xr_train, yr_train)
        train_mse = mean_squared_error(yr_train, reg.predict(Xr_train))
        test_mse = mean_squared_error(yr_test, reg.predict(Xr_test))
        print(f"  max_depth={max_depth}  leaves={reg.n_leaves():3d}  train_mse={train_mse:.4f}  test_mse={test_mse:.4f}")

    print("\n=== Use It: scikit-learn ===")
    ours = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    theirs = SklearnTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    agreement = np.mean(ours.predict(X_test) == theirs.predict(X_test))
    print(f"  ours test_acc={accuracy_score(y_test, ours.predict(X_test)):.3f}")
    print(f"  sklearn test_acc={accuracy_score(y_test, theirs.predict(X_test)):.3f}")
    print(f"  prediction agreement on test set: {agreement:.3f}")
    print(export_text(theirs, feature_names=["x0", "x1"], decimals=3))

    ours_reg = DecisionTreeRegressor(max_depth=4).fit(Xr_train, yr_train)
    theirs_reg = SklearnTreeRegressor(max_depth=4, random_state=0).fit(Xr_train, yr_train)
    print(f"  ours regressor test_mse={mean_squared_error(yr_test, ours_reg.predict(Xr_test)):.4f}")
    print(f"  sklearn regressor test_mse={mean_squared_error(yr_test, theirs_reg.predict(Xr_test)):.4f}")


if __name__ == "__main__":
    main()
