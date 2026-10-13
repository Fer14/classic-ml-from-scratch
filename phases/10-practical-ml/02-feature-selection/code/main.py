import numpy as np
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, VarianceThreshold, f_regression
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from threadpoolctl import threadpool_limits


def make_data(n_samples=300, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n_samples, 4))
    redundant = signal[:, [0]] * 0.9 + rng.normal(scale=0.3, size=(n_samples, 1))
    noise = rng.normal(size=(n_samples, 5))
    constant = np.full((n_samples, 1), 3.0) + rng.normal(scale=1e-4, size=(n_samples, 1))
    X = np.hstack([signal, redundant, noise, constant])
    true_coefs = np.array([3.0, -2.0, 1.5, 0.5])
    y = signal @ true_coefs + rng.normal(scale=1.0, size=n_samples)
    names = ["s0", "s1", "s2", "s3", "copy_s0", "n0", "n1", "n2", "n3", "n4", "const"]
    return X, y, names


def variance_filter(X, threshold=1e-3):
    return np.flatnonzero(X.var(axis=0) > threshold)


def correlation_scores(X, y):
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    numerator = X_centered.T @ y_centered
    denominator = np.sqrt((X_centered**2).sum(axis=0) * (y_centered**2).sum()) + 1e-12
    return numerator / denominator


def f_statistic(correlations, n_samples):
    r_squared = correlations**2
    return r_squared / (1 - r_squared) * (n_samples - 2)


def fit_least_squares(X, y):
    X_with_bias = np.hstack([np.ones((len(X), 1)), X])
    coefs, *_ = np.linalg.lstsq(X_with_bias, y, rcond=None)
    return coefs


def predict_least_squares(coefs, X):
    return coefs[0] + X @ coefs[1:]


def kfold_indices(n_samples, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    return np.array_split(order, n_folds)


def cv_r2(X, y, n_folds=5):
    folds = kfold_indices(len(y), n_folds)
    scores = []
    for fold in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[fold] = False
        coefs = fit_least_squares(X[train_mask], y[train_mask])
        y_pred = predict_least_squares(coefs, X[fold])
        ss_res = np.sum((y[fold] - y_pred) ** 2)
        ss_tot = np.sum((y[fold] - y[fold].mean()) ** 2)
        scores.append(1 - ss_res / ss_tot)
    return float(np.mean(scores))


def forward_selection(X, y, candidate_idx, max_features):
    selected = []
    history = []
    remaining = list(candidate_idx)
    while remaining and len(selected) < max_features:
        trial_scores = [(cv_r2(X[:, selected + [j]], y), j) for j in remaining]
        best_score, best_feature = max(trial_scores)
        selected.append(best_feature)
        remaining.remove(best_feature)
        history.append((best_feature, best_score))
    return selected, history


def backward_elimination(X, y, candidate_idx, min_features):
    selected = list(candidate_idx)
    history = [(None, cv_r2(X[:, selected], y))]
    while len(selected) > min_features:
        trial_scores = [(cv_r2(X[:, [f for f in selected if f != j]], y), j) for j in selected]
        best_score, worst_feature = max(trial_scores)
        selected.remove(worst_feature)
        history.append((worst_feature, best_score))
    return selected, history


def main():
    X, y, names = make_data()
    print(f"=== Data: {X.shape[0]} samples, {X.shape[1]} features ===")
    print("true signal: s0, s1, s2, s3 | copy_s0 is redundant | n0..n4 are noise | const has ~zero variance")

    print("\n=== Filter 1: variance threshold ===")
    kept = variance_filter(X)
    print("kept:", [names[i] for i in kept])

    print("\n=== Filter 2: univariate correlation with y ===")
    correlations = correlation_scores(X[:, kept], y)
    f_scores = f_statistic(correlations, len(y))
    for rank, j in enumerate(np.argsort(-np.abs(correlations))):
        print(f"  {rank + 1:2d}. {names[kept[j]]:<8} r={correlations[j]:+.3f} F={f_scores[j]:8.1f}")
    print("note: copy_s0 ranks high even though it adds nothing once s0 is in the model")

    print("\n=== Wrapper: forward selection with 5-fold CV R^2 ===")
    selected, history = forward_selection(X, y, kept, max_features=7)
    for feature, score in history:
        print(f"  add {names[feature]:<8} -> cv R^2 = {score:.4f}")

    print("\n=== Wrapper: backward elimination ===")
    remaining, history = backward_elimination(X, y, kept, min_features=3)
    print(f"  start with all {len(kept)} features -> cv R^2 = {history[0][1]:.4f}")
    for feature, score in history[1:]:
        print(f"  drop {names[feature]:<8} -> cv R^2 = {score:.4f}")

    print("\n=== Use It: scikit-learn ===")
    variance_selector = VarianceThreshold(threshold=1e-3).fit(X)
    print("VarianceThreshold kept:", [names[i] for i in variance_selector.get_support(indices=True)])
    X_kept = variance_selector.transform(X)
    kept_names = [names[i] for i in variance_selector.get_support(indices=True)]

    kbest = SelectKBest(score_func=f_regression, k=4).fit(X_kept, y)
    print("SelectKBest(f_regression, k=4):", [kept_names[i] for i in kbest.get_support(indices=True)])
    sklearn_f, _ = f_regression(X_kept, y)
    print("max |F_scratch - F_sklearn|:", float(np.max(np.abs(sklearn_f - f_scores))))

    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=4, direction="forward", cv=5)
    sfs.fit(X_kept, y)
    print("SequentialFeatureSelector(forward, 4):", [kept_names[i] for i in sfs.get_support(indices=True)])

    lasso = LassoCV(cv=5, random_state=0).fit(X_kept, y)
    print(f"LassoCV alpha={lasso.alpha_:.4f}, coefficients:")
    strong_lasso = Lasso(alpha=0.1).fit(X_kept, y)
    print(f"  {'feature':<8} {'LassoCV':>8} {'alpha=0.1':>10}")
    for name, coef, strong_coef in zip(kept_names, lasso.coef_, strong_lasso.coef_):
        print(f"  {name:<8} {coef:+8.3f} {strong_coef:+10.3f}")
    print("Lasso(alpha=0.1) keeps:", [n for n, c in zip(kept_names, strong_lasso.coef_) if c != 0])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
