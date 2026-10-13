import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def gini(class_counts):
    total = class_counts.sum()
    if total == 0:
        return 0.0
    proportions = class_counts / total
    return 1.0 - np.sum(proportions ** 2)


class RandomizedTree:
    def __init__(self, max_depth=None, min_samples_leaf=1, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def fit(self, X, y, n_classes):
        self.n_classes = n_classes
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self.grow(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def best_split(self, X, y):
        n_samples = len(y)
        parent_impurity = gini(np.bincount(y, minlength=self.n_classes))
        k = self.max_features or self.n_features
        candidate_features = self.rng.choice(self.n_features, size=k, replace=False)
        one_hot = np.eye(self.n_classes)[y]
        n_left = np.arange(1, n_samples)
        n_right = n_samples - n_left
        best = (None, None, 0.0)
        for feature in candidate_features:
            order = np.argsort(X[:, feature])
            sorted_values = X[order, feature]
            left_counts = np.cumsum(one_hot[order], axis=0)[:-1]
            right_counts = left_counts[-1] + one_hot[order][-1] - left_counts
            gini_left = 1.0 - np.sum((left_counts / n_left[:, None]) ** 2, axis=1)
            gini_right = 1.0 - np.sum((right_counts / n_right[:, None]) ** 2, axis=1)
            gain = parent_impurity - (n_left * gini_left + n_right * gini_right) / n_samples
            valid = (sorted_values[:-1] < sorted_values[1:])
            valid &= (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not valid.any():
                continue
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best[2]:
                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                best = (feature, threshold, gain[i])
        return best

    def grow(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes)
        leaf = {"leaf": True, "proba": counts / counts.sum()}
        if (self.max_depth is not None and depth >= self.max_depth) or counts.max() == len(y):
            return leaf
        feature, threshold, gain = self.best_split(X, y)
        if feature is None:
            return leaf
        self.feature_importances_[feature] += gain * len(y)
        mask = X[:, feature] <= threshold
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self.grow(X[mask], y[mask], depth + 1),
            "right": self.grow(X[~mask], y[~mask], depth + 1),
        }

    def predict_proba_one(self, x):
        node = self.root
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["proba"]

    def predict_proba(self, X):
        return np.array([self.predict_proba_one(x) for x in X])


class RandomForest:
    def __init__(self, n_trees=50, max_features="sqrt", max_depth=None, seed=0):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

    def resolve_max_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features is None:
            return n_features
        return int(self.max_features)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_classes = int(y.max()) + 1
        k = self.resolve_max_features(n_features)
        self.trees = []
        self.in_bag_masks = []
        for _ in range(self.n_trees):
            indices = self.rng.integers(0, n_samples, size=n_samples)
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[indices] = True
            tree = RandomizedTree(max_depth=self.max_depth, max_features=k, rng=self.rng)
            tree.fit(X[indices], y[indices], self.n_classes)
            self.trees.append(tree)
            self.in_bag_masks.append(in_bag)
        self.compute_oob_score(X, y)
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees], axis=0)
        return self

    def compute_oob_score(self, X, y):
        votes = np.zeros((len(y), self.n_classes))
        for tree, in_bag in zip(self.trees, self.in_bag_masks):
            out_of_bag = ~in_bag
            votes[out_of_bag] += tree.predict_proba(X[out_of_bag])
        has_vote = votes.sum(axis=1) > 0
        self.oob_score_ = np.mean(votes[has_vote].argmax(axis=1) == y[has_vote])

    def predict_proba(self, X):
        return np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def average_pairwise_correlation(prediction_matrix):
    correlation = np.corrcoef(prediction_matrix)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    return np.nanmean(upper)


def main():
    X, y = make_classification(
        n_samples=800, n_features=30, n_informative=12, n_redundant=6,
        n_repeated=0, n_clusters_per_class=3, flip_y=0.03, random_state=3,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)

    print("=== Step 1: one fully grown tree ===")
    single = RandomizedTree(max_features=None, rng=np.random.default_rng(0)).fit(X_train, y_train, 2)
    single_pred = single.predict_proba(X_test).argmax(axis=1)
    print(f"single tree test accuracy: {accuracy(y_test, single_pred):.3f}")

    print("\n=== Step 2: bagging vs random forest (decorrelation) ===")
    start = time.time()
    bagged = RandomForest(n_trees=60, max_features=None, seed=1).fit(X_train, y_train)
    forest = RandomForest(n_trees=60, max_features="sqrt", seed=1).fit(X_train, y_train)
    for name, model in [("bagging (all features)", bagged), ("random forest (sqrt)", forest)]:
        per_tree = np.array([t.predict_proba(X_test)[:, 1] for t in model.trees])
        corr = average_pairwise_correlation(per_tree)
        test_acc = accuracy(y_test, model.predict(X_test))
        print(f"{name:24s} test acc={test_acc:.3f}  oob acc={model.oob_score_:.3f}  tree corr={corr:.3f}")
    print(f"(trained 120 trees in {time.time() - start:.1f}s)")

    print("\n=== Step 3: accuracy vs number of trees ===")
    all_tree_probas = np.array([t.predict_proba(X_test)[:, 1] for t in forest.trees])
    for n in [1, 5, 10, 30, 60]:
        vote = (all_tree_probas[:n].mean(axis=0) > 0.5).astype(int)
        print(f"trees={n:3d}  test acc={accuracy(y_test, vote):.3f}")

    print("\n=== Step 4: impurity-based feature importance ===")
    ranking = np.argsort(forest.feature_importances_)[::-1]
    for feature in ranking[:6]:
        print(f"feature {feature:2d}  importance={forest.feature_importances_[feature]:.3f}")

    print("\n=== Use It: sklearn RandomForestClassifier ===")
    sk_tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    sk_forest = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", oob_score=True, n_jobs=-1, random_state=0,
    ).fit(X_train, y_train)
    print(f"sklearn single tree test acc: {sk_tree.score(X_test, y_test):.3f}")
    print(f"sklearn forest test acc:      {sk_forest.score(X_test, y_test):.3f}")
    print(f"sklearn forest oob acc:       {sk_forest.oob_score_:.3f}")
    sk_ranking = np.argsort(sk_forest.feature_importances_)[::-1][:6]
    print(f"top features (ours):    {ranking[:6].tolist()}")
    print(f"top features (sklearn): {sk_ranking.tolist()}")


if __name__ == "__main__":
    main()
