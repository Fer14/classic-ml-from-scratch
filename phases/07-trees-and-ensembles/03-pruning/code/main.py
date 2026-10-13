import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


class Node:
    def __init__(self, counts, depth):
        self.counts = counts
        self.n_samples = int(counts.sum())
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.pruned = False

    @property
    def impurity(self):
        proportions = self.counts / self.n_samples
        return 1.0 - float(np.sum(proportions**2))

    @property
    def prediction(self):
        return int(np.argmax(self.counts))

    def is_leaf(self):
        return self.left is None or self.pruned


def best_split(X, y_onehot, min_samples_leaf):
    n_samples, n_features = X.shape
    parent_counts = y_onehot.sum(axis=0)
    parent_score = n_samples - np.sum(parent_counts**2) / n_samples
    best_gain, best_feature, best_threshold = 0.0, None, None
    for feature in range(n_features):
        order = np.argsort(X[:, feature], kind="stable")
        sorted_values = X[order, feature]
        left_counts = np.cumsum(y_onehot[order], axis=0)[:-1]
        right_counts = parent_counts - left_counts
        left_n = np.arange(1, n_samples)
        right_n = n_samples - left_n
        valid = (sorted_values[:-1] < sorted_values[1:]) & (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
        if not valid.any():
            continue
        left_score = left_n - np.sum(left_counts**2, axis=1) / left_n
        right_score = right_n - np.sum(right_counts**2, axis=1) / right_n
        gains = np.where(valid, (parent_score - left_score - right_score) / n_samples, -np.inf)
        index = int(np.argmax(gains))
        if gains[index] > best_gain + 1e-12:
            best_gain = float(gains[index])
            best_feature = feature
            best_threshold = (sorted_values[index] + sorted_values[index + 1]) / 2
    return best_gain, best_feature, best_threshold


class PrunableTree:
    def __init__(self, max_depth=None, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y):
        self.n_classes = int(y.max()) + 1
        self.n_total = len(y)
        y_onehot = np.eye(self.n_classes)[y]
        self.root = self.grow(X, y_onehot, depth=0)
        return self

    def grow(self, X, y_onehot, depth):
        node = Node(y_onehot.sum(axis=0), depth)
        if node.impurity == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return node
        gain, feature, threshold = best_split(X, y_onehot, self.min_samples_leaf)
        weighted_gain = gain * node.n_samples / self.n_total
        if feature is None or weighted_gain < self.min_impurity_decrease:
            return node
        mask = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self.grow(X[mask], y_onehot[mask], depth + 1)
        node.right = self.grow(X[~mask], y_onehot[~mask], depth + 1)
        return node

    def predict(self, X):
        predictions = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf():
                node = node.left if x[node.feature] <= node.threshold else node.right
            predictions[i] = node.prediction
        return predictions

    def nodes(self, node=None):
        node = node or self.root
        yield node
        if not node.is_leaf():
            yield from self.nodes(node.left)
            yield from self.nodes(node.right)

    def node_risk(self, node):
        return node.n_samples / self.n_total * node.impurity

    def subtree_risk_and_leaves(self, node):
        if node.is_leaf():
            return self.node_risk(node), 1
        left_risk, left_leaves = self.subtree_risk_and_leaves(node.left)
        right_risk, right_leaves = self.subtree_risk_and_leaves(node.right)
        return left_risk + right_risk, left_leaves + right_leaves

    def n_leaves(self):
        return self.subtree_risk_and_leaves(self.root)[1]

    def total_leaf_impurity(self):
        return self.subtree_risk_and_leaves(self.root)[0]

    def reset_pruning(self):
        for node in self.all_nodes(self.root):
            node.pruned = False

    def all_nodes(self, node):
        yield node
        if node.left is not None:
            yield from self.all_nodes(node.left)
            yield from self.all_nodes(node.right)

    def pruning_path(self):
        self.reset_pruning()
        alphas = [0.0]
        impurities = [self.total_leaf_impurity()]
        while not self.root.is_leaf():
            weakest_links = []
            for node in self.nodes():
                if node.is_leaf():
                    continue
                subtree_risk, subtree_leaves = self.subtree_risk_and_leaves(node)
                effective_alpha = (self.node_risk(node) - subtree_risk) / (subtree_leaves - 1)
                weakest_links.append((effective_alpha, node))
            smallest = min(alpha for alpha, _ in weakest_links)
            for alpha, node in weakest_links:
                if alpha <= smallest + 1e-12:
                    node.pruned = True
            alphas.append(max(smallest, alphas[-1]))
            impurities.append(self.total_leaf_impurity())
        self.reset_pruning()
        return np.array(alphas), np.array(impurities)

    def prune(self, ccp_alpha):
        self.reset_pruning()
        while not self.root.is_leaf():
            candidates = []
            for node in self.nodes():
                if node.is_leaf():
                    continue
                subtree_risk, subtree_leaves = self.subtree_risk_and_leaves(node)
                candidates.append(((self.node_risk(node) - subtree_risk) / (subtree_leaves - 1), node))
            smallest = min(alpha for alpha, _ in candidates)
            if smallest > ccp_alpha:
                break
            for alpha, node in candidates:
                if alpha <= smallest + 1e-12:
                    node.pruned = True
        return self


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def cross_validated_alpha(X, y, candidate_alphas, n_splits=5):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = np.zeros(len(candidate_alphas))
    for train_index, valid_index in folds.split(X, y):
        tree = PrunableTree().fit(X[train_index], y[train_index])
        for i, alpha in enumerate(candidate_alphas):
            tree.prune(alpha)
            scores[i] += accuracy(y[valid_index], tree.predict(X[valid_index])) / n_splits
    return scores


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0, stratify=data.target
    )

    print("=== Step 1: The fully grown tree overfits ===")
    full_tree = PrunableTree().fit(X_train, y_train)
    print(f"  leaves={full_tree.n_leaves()}  train_acc={accuracy(y_train, full_tree.predict(X_train)):.3f}  test_acc={accuracy(y_test, full_tree.predict(X_test)):.3f}")

    print("\n=== Step 2: Pre-pruning (stop early) ===")
    settings = [
        {"max_depth": 2},
        {"max_depth": 4},
        {"min_samples_leaf": 10},
        {"min_impurity_decrease": 0.01},
    ]
    for params in settings:
        tree = PrunableTree(**params).fit(X_train, y_train)
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {label:28s} leaves={tree.n_leaves():3d}  train_acc={accuracy(y_train, tree.predict(X_train)):.3f}  test_acc={accuracy(y_test, tree.predict(X_test)):.3f}")

    print("\n=== Step 3: Cost-complexity pruning path (weakest link) ===")
    alphas, impurities = full_tree.pruning_path()
    print(f"  {'alpha':>8s}  {'leaf impurity':>13s}  {'leaves':>6s}  {'test_acc':>8s}")
    for alpha, impurity in zip(alphas, impurities):
        full_tree.prune(alpha)
        print(f"  {alpha:8.5f}  {impurity:13.4f}  {full_tree.n_leaves():6d}  {accuracy(y_test, full_tree.predict(X_test)):8.3f}")

    print("\n=== Step 4: Choose alpha with 5-fold cross-validation ===")
    candidate_alphas = alphas[:-1]
    cv_scores = cross_validated_alpha(X_train, y_train, candidate_alphas)
    best_alpha = float(candidate_alphas[int(np.argmax(cv_scores))])
    for alpha, score in zip(candidate_alphas, cv_scores):
        marker = "  <- best" if alpha == best_alpha else ""
        print(f"  alpha={alpha:.5f}  cv_acc={score:.3f}{marker}")
    full_tree.prune(best_alpha)
    print(f"  pruned tree: leaves={full_tree.n_leaves()}  test_acc={accuracy(y_test, full_tree.predict(X_test)):.3f}")

    print("\n=== Use It: scikit-learn cost_complexity_pruning_path ===")
    sklearn_tree = DecisionTreeClassifier(random_state=0)
    path = sklearn_tree.cost_complexity_pruning_path(X_train, y_train)
    print(f"  ours    alphas: {np.round(alphas, 5).tolist()}")
    print(f"  sklearn alphas: {np.round(path.ccp_alphas, 5).tolist()}")
    if len(alphas) == len(path.ccp_alphas):
        print(f"  max |difference| = {np.max(np.abs(alphas - path.ccp_alphas)):.2e}")
    sklearn_alpha = float(path.ccp_alphas[int(np.argmax(cv_scores))])
    pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=sklearn_alpha).fit(X_train, y_train)
    print(f"  sklearn with ccp_alpha={sklearn_alpha:.5f}: leaves={pruned.get_n_leaves()}  test_acc={accuracy(y_test, pruned.predict(X_test)):.3f}")


if __name__ == "__main__":
    main()
