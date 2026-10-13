# XGBoost and LightGBM in Practice

> Modern gradient boosting is the same algorithm as the last lesson with three engineering upgrades: bin the features, use second derivatives, regularize the leaves.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 02 (Calculus and Optimization: second derivatives, Newton's method), Phase 04 (ML Fundamentals: validation sets, metrics), Phase 07 Lesson 07 (Gradient Boosting)
**Time:** ~90 minutes

## Learning Objectives

- Explain how histogram binning turns split finding from "sort every feature at every node" into "add gradients into 256 buckets"
- Derive the XGBoost leaf weight $-G/(H+\lambda)$ and split gain from a second-order Taylor expansion, and implement them from scratch
- Train scikit-learn's `HistGradientBoostingClassifier/Regressor` with early stopping, native categorical features and monotonic constraints
- Map the main XGBoost and LightGBM parameters to the concepts in this lesson so you can read and tune their APIs

## The Problem

The classic gradient boosting you built in Lesson 07 sorts every feature at every node. That costs $O(n \log n)$ per feature per node, and with a million rows and a few hundred features it becomes hours of training. It also has no explicit regularization beyond shrinkage and depth, and it cannot handle missing values or categorical columns directly.

Between 2014 and 2017 three libraries fixed this: XGBoost (Chen and Guestrin), LightGBM (Microsoft) and CatBoost (Yandex). scikit-learn followed with `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`, inspired by LightGBM. They dominate tabular data in industry and in Kaggle competitions. This lesson builds their core ideas in about 100 lines of NumPy, then uses `HistGradientBoosting*` as the runnable stand-in. XGBoost and LightGBM are not dependencies of this course; their APIs are shown so you recognize them, not run.

## The Concept

### Idea 1: Histogram binning

Before training, replace each feature by its quantile bin index (at most 255 bins, stored as `uint8`):

```
raw feature:  0.031  1.442 -0.271  2.913  0.884 ...   (7200 unique float64 values)
bin edges:    63 quantile cut points
binned:       31     52     26     63     44   ...   (64 unique uint8 values)
```

Now a node's split search is:

1. For each feature, add every sample's gradient and Hessian into its bin: one `bincount`, $O(n)$.
2. Cumulative-sum the bins left to right: $O(\text{bins})$ candidate splits instead of $O(n)$.

No sorting, 8x less memory, and better cache behavior. Coarse bins act as mild regularization too, so accuracy rarely suffers. A second trick, histogram subtraction, computes only the smaller child's histogram and gets the sibling as `parent - child`.

### Idea 2: Second-order (Newton) boosting

Write the loss after adding a tree $f$ to the current score $F$, and Taylor-expand to second order:

```
L(F + f) ~ sum_i [ l(y_i, F_i) + g_i * f(x_i) + 0.5 * h_i * f(x_i)^2 ] + Omega(f)

g_i = dl/dF      (gradient)        for log-loss: p_i - y_i
h_i = d2l/dF2    (Hessian)         for log-loss: p_i * (1 - p_i)
Omega(f) = gamma * (number of leaves) + 0.5 * lambda * sum(w_j^2)
```

Every sample in leaf j gets the same value $w_j$. Group the sum by leaves, with $G_j = \sum g_i$ and $H_j = \sum h_i$ over the leaf, and minimize each leaf independently:

```
optimal leaf weight:   w_j* = - G_j / (H_j + lambda)
loss reduction:        -0.5 * G_j^2 / (H_j + lambda)
```

So the gain from splitting a node into left and right children is:

```
gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma
```

Split only if `gain > 0`. Every term comes straight from the histograms: cumulative sums of g give $G_L$, of h give $H_L$.

### Idea 3: Regularization knobs

| Concept | XGBoost | LightGBM | sklearn HistGradientBoosting |
|---|---|---|---|
| Shrinkage | `learning_rate` (eta) | `learning_rate` | `learning_rate` |
| Number of trees | `n_estimators` | `n_estimators` / `num_iterations` | `max_iter` |
| L2 on leaf weights (lambda) | `reg_lambda` | `lambda_l2` / `reg_lambda` | `l2_regularization` |
| Min gain to split (gamma) | `gamma` / `min_split_loss` | `min_split_gain` | not exposed |
| Min Hessian per child | `min_child_weight` | `min_child_weight` / `min_sum_hessian_in_leaf` | `min_samples_leaf` (counts) |
| Tree size | `max_depth` (depth-wise) | `num_leaves` (leaf-wise) | `max_leaf_nodes`, `max_depth` |
| Row subsampling | `subsample` | `bagging_fraction` | not exposed |
| Feature subsampling | `colsample_bytree` | `feature_fraction` | `max_features` |
| Number of bins | `max_bin` | `max_bin` | `max_bins` (at most 255) |
| Early stopping | `early_stopping_rounds` | callback `early_stopping` | `early_stopping`, `n_iter_no_change` |

Growth strategy differs: XGBoost (by default) grows level by level (depth-wise), LightGBM and HistGradientBoosting grow leaf-wise, always splitting the leaf with the largest gain. Leaf-wise trees reach lower loss with the same number of leaves but can overfit small data; cap them with `num_leaves`/`max_leaf_nodes`.

### Missing values and categories

- **Missing values.** For each split, the libraries try sending all missing values left and right and keep the better direction. No imputation needed.
- **Categorical features.** Instead of one-hot encoding (many weak splits) or ordinal codes (arbitrary order), they sort categories by $G/H$ within the node and find the best partition of categories into two groups. In HistGradientBoosting you pass `categorical_features`; in LightGBM, `categorical_feature`; in XGBoost, `enable_categorical=True` with a pandas category dtype.

## Build It

### Step 1: Quantile binning

```python
def build_bin_edges(X, n_bins=64):
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return [np.unique(np.quantile(X[:, j], quantiles)) for j in range(X.shape[1])]


def apply_bins(X, edges):
    return np.column_stack([np.searchsorted(edges[j], X[:, j], side="right") for j in range(X.shape[1])]).astype(np.uint8)
```

The edges are computed once on training data and reused for validation and test data.

### Step 2: Gradient and Hessian histograms for all features at once

Offset each feature's bin index so all features share one `bincount` call:

```python
def histograms(self, binned, grad, hess):
    n_features = binned.shape[1]
    offsets = binned + (np.arange(n_features) * self.n_bins)[None, :]
    size = n_features * self.n_bins
    grad_hist = np.bincount(offsets.ravel(), weights=np.repeat(grad, n_features), minlength=size)
    hess_hist = np.bincount(offsets.ravel(), weights=np.repeat(hess, n_features), minlength=size)
    return grad_hist.reshape(n_features, self.n_bins), hess_hist.reshape(n_features, self.n_bins)
```

### Step 3: Second-order gain and leaf weights

```python
def leaf_weight(self, grad_sum, hess_sum):
    return -grad_sum / (hess_sum + self.reg_lambda)

def score(self, grad_sum, hess_sum):
    return grad_sum ** 2 / (hess_sum + self.reg_lambda)

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
```

`gain` is a (features x bins) matrix; one `argmax` finds the best split over every feature and threshold.

### Step 4: The boosting loop with early stopping

```python
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
```

After the loop, the model keeps only the trees up to the best validation round, exactly what `best_iteration` means in XGBoost and LightGBM.

### Step 5: Run it

```bash
uv run python phases/07-trees-and-ensembles/08-xgboost-and-lightgbm-in-practice/code/main.py
```

```
raw feature 0 unique values: 7200
binned feature 0 unique values: 64  dtype=uint8
memory: raw 1.15 MB -> binned 0.14 MB
best root split: feature 0, bin <= 30, gain=1043.26
left  child: n= 3488  G=  969.02  H= 871.94  weight=-G/(H+lambda)=-1.110
right child: n= 3712  G= -969.02  H= 927.93  weight=-G/(H+lambda)=+1.043
trained 156 rounds, kept 136 trees in 1.0s
test AUC=0.9703  log-loss=0.1909
```

At the root, every sample has p equal to the base rate, so the left child (mostly negatives, positive gradient sum) gets a negative weight and the right child a positive one. The from-scratch model stops on its own after validation loss stops improving for 20 rounds.

## Use It

### scikit-learn (runnable, used in this course)

```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

hgb = HistGradientBoostingClassifier(
    learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
    early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=0,
).fit(X_train, y_train)

as_category = HistGradientBoostingClassifier(
    max_iter=100, categorical_features=categorical_mask, random_state=0,
).fit(Xc_train, yc_train)

constrained = HistGradientBoostingRegressor(max_iter=200, monotonic_cst=[1, 0]).fit(Xr, yr)
```

```
HistGradientBoosting     AUC=0.9702  time=0.21s
GradientBoosting (100)   AUC=0.9676  time=4.12s
HistGradientBoosting stopped after 92 iterations
category code treated as a number: AUC=0.7233
category as categorical_features:  AUC=0.7959
unconstrained          prediction decreases along x0 at  73 of 199 grid steps
monotonic_cst=[1, 0]   prediction decreases along x0 at   0 of 199 grid steps
```

- Histogram boosting is about 20x faster than classic boosting here, with equal or better AUC, and matches the from-scratch version.
- A 200-level category coded as integers forces the trees to carve arbitrary intervals; native categorical splits gain 7 AUC points.
- `monotonic_cst=[1, 0]` guarantees predictions never decrease as feature 0 grows, a common business requirement (price vs size, risk vs debt).
- HistGradientBoosting parallelizes with OpenMP. On small data or a busy machine, too many threads can make it slower, not faster. The script wraps `main()` in `threadpoolctl.threadpool_limits(limits=2)` (threadpoolctl ships with scikit-learn); setting the `OMP_NUM_THREADS` environment variable does the same.

### XGBoost and LightGBM (for reference, not run in this course)

If you install them yourself (`uv add xgboost lightgbm`), their scikit-learn-style APIs look like this:

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", enable_categorical=True,
    early_stopping_rounds=50, eval_metric="logloss",
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
print(model.best_iteration)
```

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.05, num_leaves=31,
    reg_lambda=1.0, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[lgb.early_stopping(50)],
)
print(model.best_iteration_)
```

A practical tuning order for any of the three: fix `learning_rate` at 0.05-0.1 and use early stopping for the tree count; tune tree size (`max_depth` or `num_leaves`/`max_leaf_nodes`) and `min_child_weight`/`min_samples_leaf`; then row and column subsampling; then `reg_lambda`; finally lower the learning rate and let early stopping add trees.

## Exercises

1. **Easy.** Run the from-scratch model with `n_bins` in {4, 16, 64, 255} (change `uint8` only if needed). How much does test AUC drop with 4 or 16 bins? What does that tell you about how much split precision matters?
2. **Medium.** Implement histogram subtraction: in `build`, compute the histogram only for the smaller child and obtain the sibling's histogram as parent minus child. Pass histograms down the recursion and measure the speedup.
3. **Hard.** Replace depth-wise growth with leaf-wise growth: keep a priority queue of leaves keyed by their best gain, always split the best one, and stop at `max_leaves`. Compare with depth-wise growth at the same number of leaves on the 12000-row dataset.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Histogram binning | "Bucketize features" | Mapping each feature to at most 255 quantile bins once, so split search sums gradients per bin instead of sorting |
| Gradient (g) | "First derivative" | dl/dF for each sample; p - y for log-loss |
| Hessian (h) | "Second derivative" | d2l/dF2 for each sample; p(1 - p) for log-loss; acts like a sample weight |
| Leaf weight | "Leaf value" | -G/(H + lambda), the minimizer of the second-order approximation of the loss in a leaf |
| Split gain | "How good a split is" | 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma |
| lambda / gamma | "Regularization" | L2 penalty on leaf weights / minimum gain required to make a split |
| min_child_weight | "Minimum leaf size" | Minimum sum of Hessians allowed in a child; stops splits on nearly-certain samples |
| Leaf-wise growth | "LightGBM style" | Always split the leaf with the highest gain, instead of completing each depth level |
| Native categorical split | "No one-hot" | Partitioning the categories of a feature into two groups ordered by G/H |
| Monotonic constraint | "Force the direction" | Restrict splits so predictions only increase (or decrease) with a feature |

## Further Reading

- [Chen and Guestrin (2016). XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) -- the second-order objective, gain formula, sparsity-aware splits and system design
- [Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) -- histograms, leaf-wise growth, GOSS and exclusive feature bundling
- [scikit-learn User Guide: Histogram-Based Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) -- early stopping, missing values, categorical features, monotonic and interaction constraints
- [XGBoost documentation: Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) -- a clear derivation of the leaf weight and gain formulas used in this lesson
