import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


def build_bin_edges(X, n_bins=64):
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return [np.unique(np.quantile(X[:, j], quantiles)) for j in range(X.shape[1])]


def apply_bins(X, edges):
    return np.column_stack([np.searchsorted(edges[j], X[:, j], side="right") for j in range(X.shape[1])]).astype(np.uint8)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class HistogramTree:
    def __init__(self, n_bins, max_depth=4, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def leaf_weight(self, grad_sum, hess_sum):
        return -grad_sum / (hess_sum + self.reg_lambda)

    def score(self, grad_sum, hess_sum):
        return grad_sum ** 2 / (hess_sum + self.reg_lambda)

    def histograms(self, binned, grad, hess):
        n_features = binned.shape[1]
        offsets = binned + (np.arange(n_features) * self.n_bins)[None, :]
        size = n_features * self.n_bins
        grad_hist = np.bincount(offsets.ravel(), weights=np.repeat(grad, n_features), minlength=size)
        hess_hist = np.bincount(offsets.ravel(), weights=np.repeat(hess, n_features), minlength=size)
        return grad_hist.reshape(n_features, self.n_bins), hess_hist.reshape(n_features, self.n_bins)

    def best_split(self, binned, grad, hess):
        grad_hist, hess_hist = self.histograms(binned, grad, hess)
        grad_left = np.cumsum(grad_hist, axis=1)[:, :-1]
        hess_left = np.cumsum(hess_hist, axis=1)[:, :-1]
        grad_total, hess_total = grad.sum(), hess.sum()
        grad_right, hess_right = grad_total - grad_left, hess_total - hess_left
        gain = 0.5 * (self.score(grad_left, hess_left) + self.score(grad_right, hess_right)
                      - self.score(grad_total, hess_total)) - self.gamma
        valid = (hess_left >= self.min_child_weight) & (hess_right >= self.min_child_weight)
        gain = np.where(valid, gain, -np.inf)
        feature, bin_index = np.unravel_index(np.argmax(gain), gain.shape)
        return int(feature), int(bin_index), gain[feature, bin_index]

    def fit(self, binned, grad, hess):
        self.nodes = []
        self.build(binned, grad, hess, depth=0)
        return self

    def build(self, binned, grad, hess, depth):
        node_id = len(self.nodes)
        self.nodes.append({"leaf": True, "value": self.leaf_weight(grad.sum(), hess.sum())})
        if depth >= self.max_depth or len(grad) < 2:
            return node_id
        feature, bin_index, gain = self.best_split(binned, grad, hess)
        if not gain > 0:
            return node_id
        mask = binned[:, feature] <= bin_index
        left = self.build(binned[mask], grad[mask], hess[mask], depth + 1)
        right = self.build(binned[~mask], grad[~mask], hess[~mask], depth + 1)
        self.nodes[node_id].update({"leaf": False, "feature": feature, "bin": bin_index, "left": left, "right": right})
        return node_id

    def predict(self, binned):
        node_index = np.zeros(len(binned), dtype=int)
        for _ in range(self.max_depth + 1):
            for current in np.unique(node_index):
                node = self.nodes[current]
                if node["leaf"]:
                    continue
                rows = node_index == current
                goes_left = binned[rows, node["feature"]] <= node["bin"]
                node_index[rows] = np.where(goes_left, node["left"], node["right"])
        return np.array([node["value"] for node in self.nodes])[node_index]


class HistGradientBoosting:
    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=4, n_bins=64, reg_lambda=1.0,
                 gamma=0.0, early_stopping_rounds=20):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, X_valid, y_valid):
        self.edges = build_bin_edges(X, self.n_bins)
        binned = apply_bins(X, self.edges)
        binned_valid = apply_bins(X_valid, self.edges)
        positive_rate = y.mean()
        self.base_score = np.log(positive_rate / (1 - positive_rate))
        score = np.full(len(y), self.base_score)
        valid_score = np.full(len(y_valid), self.base_score)
        self.trees, self.valid_history = [], []
        best_loss, rounds_without_improvement = np.inf, 0
        for _ in range(self.n_estimators):
            probability = sigmoid(score)
            grad = probability - y
            hess = probability * (1 - probability)
            tree = HistogramTree(self.n_bins, self.max_depth, self.reg_lambda, self.gamma).fit(binned, grad, hess)
            self.trees.append(tree)
            score += self.learning_rate * tree.predict(binned)
            valid_score += self.learning_rate * tree.predict(binned_valid)
            loss = log_loss(y_valid, sigmoid(valid_score))
            self.valid_history.append(loss)
            if loss < best_loss - 1e-6:
                best_loss, rounds_without_improvement = loss, 0
            else:
                rounds_without_improvement += 1
                if rounds_without_improvement >= self.early_stopping_rounds:
                    break
        best_round = int(np.argmin(self.valid_history)) + 1
        self.trees = self.trees[:best_round]
        return self

    def predict_proba(self, X):
        binned = apply_bins(X, self.edges)
        score = self.base_score + self.learning_rate * sum(tree.predict(binned) for tree in self.trees)
        return sigmoid(score)


def make_data_with_category(n_samples, seed):
    rng = np.random.default_rng(seed)
    X_numeric, y = make_classification(
        n_samples=n_samples, n_features=8, n_informative=5, flip_y=0.03, random_state=seed,
    )
    n_levels = 200
    level_effect = rng.normal(0, 2.0, n_levels)
    category = rng.integers(0, n_levels, n_samples)
    logit = 2.0 * (y - 0.5) + level_effect[category]
    y = (rng.random(n_samples) < sigmoid(logit)).astype(int)
    return np.column_stack([X_numeric, category]), y


def main():
    X, y = make_classification(n_samples=12000, n_features=20, n_informative=8, flip_y=0.05, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_fit, X_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    print("=== Step 1: histogram binning ===")
    edges = build_bin_edges(X_fit, n_bins=64)
    binned = apply_bins(X_fit, edges)
    print(f"raw feature 0 unique values: {len(np.unique(X_fit[:, 0]))}")
    print(f"binned feature 0 unique values: {len(np.unique(binned[:, 0]))}  dtype={binned.dtype}")
    print(f"memory: raw {X_fit.nbytes / 1e6:.2f} MB -> binned {binned.nbytes / 1e6:.2f} MB")

    print("\n=== Step 2: second-order gain on the first split ===")
    base = np.log(y_fit.mean() / (1 - y_fit.mean()))
    probability = sigmoid(np.full(len(y_fit), base))
    grad, hess = probability - y_fit, probability * (1 - probability)
    root = HistogramTree(n_bins=64, max_depth=1, reg_lambda=1.0)
    feature, bin_index, gain = root.best_split(binned, grad, hess)
    print(f"best root split: feature {feature}, bin <= {bin_index}, gain={gain:.2f}")
    goes_left = binned[:, feature] <= bin_index
    for side, rows in [("left", goes_left), ("right", ~goes_left)]:
        g_sum, h_sum = grad[rows].sum(), hess[rows].sum()
        print(f"{side:5s} child: n={rows.sum():5d}  G={g_sum:8.2f}  H={h_sum:7.2f}  weight=-G/(H+lambda)={root.leaf_weight(g_sum, h_sum):+.3f}")

    print("\n=== Step 3: from-scratch histogram GBDT with early stopping ===")
    start = time.time()
    ours = HistGradientBoosting(n_estimators=300, learning_rate=0.2, max_depth=4).fit(X_fit, y_fit, X_valid, y_valid)
    ours_time = time.time() - start
    ours_proba = ours.predict_proba(X_test)
    print(f"trained {len(ours.valid_history)} rounds, kept {len(ours.trees)} trees in {ours_time:.1f}s")
    print(f"test AUC={roc_auc_score(y_test, ours_proba):.4f}  log-loss={log_loss(y_test, ours_proba):.4f}")

    print("\n=== Use It: sklearn HistGradientBoostingClassifier vs classic GradientBoosting ===")
    start = time.time()
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300, max_leaf_nodes=31, early_stopping=True,
        validation_fraction=0.2, n_iter_no_change=20, random_state=0,
    ).fit(X_train, y_train)
    hgb_time = time.time() - start
    start = time.time()
    classic = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(X_train, y_train)
    classic_time = time.time() - start
    for name, model, seconds in [("HistGradientBoosting", hgb, hgb_time), ("GradientBoosting (100)", classic, classic_time)]:
        proba = model.predict_proba(X_test)[:, 1]
        print(f"{name:24s} AUC={roc_auc_score(y_test, proba):.4f}  time={seconds:.2f}s")
    print(f"HistGradientBoosting stopped after {hgb.n_iter_} iterations")

    print("\n=== Use It: native categorical support ===")
    Xc, yc = make_data_with_category(6000, seed=1)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.25, random_state=1)
    categorical_mask = np.zeros(Xc.shape[1], dtype=bool)
    categorical_mask[-1] = True
    as_numeric = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(Xc_train, yc_train)
    as_category = HistGradientBoostingClassifier(
        max_iter=100, categorical_features=categorical_mask, random_state=0,
    ).fit(Xc_train, yc_train)
    print(f"category code treated as a number: AUC={roc_auc_score(yc_test, as_numeric.predict_proba(Xc_test)[:, 1]):.4f}")
    print(f"category as categorical_features:  AUC={roc_auc_score(yc_test, as_category.predict_proba(Xc_test)[:, 1]):.4f}")

    print("\n=== Use It: HistGradientBoostingRegressor with monotonic constraint ===")
    rng = np.random.default_rng(3)
    Xr = rng.uniform(0, 10, size=(3000, 2))
    yr = 2.0 * Xr[:, 0] + np.sin(Xr[:, 1]) * 3 + rng.normal(0, 2.0, 3000)
    free = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xr, yr)
    constrained = HistGradientBoostingRegressor(max_iter=200, monotonic_cst=[1, 0], random_state=0).fit(Xr, yr)
    grid = np.column_stack([np.linspace(0, 10, 200), np.full(200, 5.0)])
    for name, model in [("unconstrained", free), ("monotonic_cst=[1, 0]", constrained)]:
        drops = int(np.sum(np.diff(model.predict(grid)) < -1e-9))
        print(f"{name:22s} prediction decreases along x0 at {drops:3d} of 199 grid steps")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
