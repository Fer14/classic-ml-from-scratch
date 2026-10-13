import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


def z_scores(values):
    return (values - values.mean()) / values.std()


def robust_z_scores(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 0.6745 * (values - median) / mad


def average_path_length(n):
    n = np.asarray(n, dtype=float)
    harmonic = np.log(np.maximum(n - 1, 1)) + 0.5772156649
    result = 2.0 * harmonic - 2.0 * (n - 1) / np.maximum(n, 1)
    return np.where(n > 2, result, np.where(n == 2, 1.0, 0.0))


class IsolationTree:
    def __init__(self, height_limit, rng):
        self.height_limit = height_limit
        self.rng = rng

    def fit(self, X, depth=0):
        self.size = X.shape[0]
        self.depth = depth
        is_constant = np.all(X.max(axis=0) == X.min(axis=0))
        if depth >= self.height_limit or self.size <= 1 or is_constant:
            self.is_leaf = True
            return self
        self.is_leaf = False
        spread = X.max(axis=0) - X.min(axis=0)
        candidate_features = np.flatnonzero(spread > 0)
        self.feature = self.rng.choice(candidate_features)
        low, high = X[:, self.feature].min(), X[:, self.feature].max()
        self.threshold = self.rng.uniform(low, high)
        goes_left = X[:, self.feature] < self.threshold
        self.left = IsolationTree(self.height_limit, self.rng).fit(X[goes_left], depth + 1)
        self.right = IsolationTree(self.height_limit, self.rng).fit(X[~goes_left], depth + 1)
        return self

    def path_length(self, x):
        node = self
        while not node.is_leaf:
            node = node.left if x[node.feature] < node.threshold else node.right
        return node.depth + average_path_length(node.size)


class IsolationForestFromScratch:
    def __init__(self, n_trees=100, sample_size=256, seed=0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)

    def fit(self, X):
        self.sample_size_ = min(self.sample_size, X.shape[0])
        height_limit = int(np.ceil(np.log2(self.sample_size_)))
        self.trees = []
        for _ in range(self.n_trees):
            rows = self.rng.choice(X.shape[0], size=self.sample_size_, replace=False)
            self.trees.append(IsolationTree(height_limit, self.rng).fit(X[rows]))
        return self

    def score(self, X):
        mean_depth = np.array([np.mean([tree.path_length(x) for tree in self.trees]) for x in X])
        return 2.0 ** (-mean_depth / average_path_length(self.sample_size_))


def make_dataset(seed):
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal([0, 0], 0.6, size=(450, 2))
    cluster_b = rng.normal([4, 4], 0.9, size=(450, 2))
    between = rng.normal([2, 2], 0.25, size=(10, 2))
    far = rng.uniform(-6, 10, size=(30, 2))
    X = np.vstack([cluster_a, cluster_b, between, far])
    y = np.r_[np.zeros(900), np.ones(40)].astype(int)
    return X, y


print("=== Step 1: z-score on a single feature ===")
rng = np.random.default_rng(1)
latencies = np.r_[rng.normal(120, 10, 200), [400, 410, 395, 420, 405, 390, 415, 398]]
standard = z_scores(latencies)
robust = robust_z_scores(latencies)
print(f"Injected 8 spikes around 400 ms into 200 normal readings near 120 ms")
print(f"Mean={latencies.mean():.1f}  std={latencies.std():.1f}  median={np.median(latencies):.1f}")
print(f"Flagged with |z| > 3:        {np.sum(np.abs(standard) > 3)}")
print(f"Flagged with |robust z| > 3.5: {np.sum(np.abs(robust) > 3.5)}")
print(f"z-score of first spike: {standard[200]:.2f}  robust z: {robust[200]:.2f}")

masked = np.r_[rng.normal(120, 10, 200), rng.normal(400, 5, 30)]
print(f"With 30 spikes (13% contamination): |z|>3 flags {np.sum(np.abs(z_scores(masked)) > 3)}, robust flags {np.sum(np.abs(robust_z_scores(masked)) > 3.5)}")

print("\n=== Step 2: Isolation path lengths ===")
X, y = make_dataset(seed=0)
forest = IsolationForestFromScratch(n_trees=100, sample_size=256, seed=0).fit(X)
scores = forest.score(X)
print(f"c(256) = {average_path_length(256):.3f} expected depth of an unsuccessful BST search")
print(f"Mean anomaly score  inliers: {scores[y == 0].mean():.3f}  outliers: {scores[y == 1].mean():.3f}")

print("\n=== Step 3: Compare detectors with ROC AUC ===")
per_feature_z = np.abs((X - X.mean(axis=0)) / X.std(axis=0)).max(axis=1)
print(f"Max |z| across features   AUC: {roc_auc_score(y, per_feature_z):.3f}")
print(f"Isolation forest (scratch) AUC: {roc_auc_score(y, scores):.3f}")

print("\n=== Step 4: Choose a threshold by contamination ===")
contamination = 0.05
threshold = np.quantile(scores, 1 - contamination)
flagged = scores >= threshold
true_positives = np.sum(flagged & (y == 1))
print(f"Threshold at top {contamination:.0%}: score >= {threshold:.3f}")
print(f"Flagged {flagged.sum()}  precision={true_positives / flagged.sum():.3f}  recall={true_positives / y.sum():.3f}")

print("\n=== Use It: sklearn IsolationForest ===")
sk_forest = IsolationForest(n_estimators=100, max_samples=256, contamination=contamination, random_state=0).fit(X)
sk_scores = -sk_forest.score_samples(X)
sk_flagged = sk_forest.predict(X) == -1
print(f"sklearn AUC: {roc_auc_score(y, sk_scores):.3f}")
print(f"sklearn flagged {sk_flagged.sum()}  precision={np.sum(sk_flagged & (y == 1)) / sk_flagged.sum():.3f}")
print(f"Correlation between scratch and sklearn scores: {np.corrcoef(scores, sk_scores)[0, 1]:.3f}")
