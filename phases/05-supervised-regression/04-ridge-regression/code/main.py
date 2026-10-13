import numpy as np
from sklearn.linear_model import Ridge, RidgeCV


def make_collinear_data(n_samples, rng):
    base = rng.normal(0, 1, n_samples)
    feature_a = base + rng.normal(0, 0.05, n_samples)
    feature_b = base + rng.normal(0, 0.05, n_samples)
    noise_features = rng.normal(0, 1, (n_samples, 8))
    features = np.column_stack([feature_a, feature_b, noise_features])
    targets = 3 * feature_a + 3 * feature_b + rng.normal(0, 1.0, n_samples)
    return features, targets


def standardize(train, other):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    return (train - mean) / std, (other - mean) / std


def fit_ridge(features, targets, alpha):
    n_features = features.shape[1]
    x_mean = features.mean(axis=0)
    y_mean = targets.mean()
    centered_x = features - x_mean
    centered_y = targets - y_mean
    weights = np.linalg.solve(centered_x.T @ centered_x + alpha * np.eye(n_features), centered_x.T @ centered_y)
    bias = y_mean - x_mean @ weights
    return weights, bias


def fit_ridge_svd(features, targets, alpha):
    x_mean = features.mean(axis=0)
    y_mean = targets.mean()
    u, singular_values, vt = np.linalg.svd(features - x_mean, full_matrices=False)
    shrink = singular_values / (singular_values ** 2 + alpha)
    weights = vt.T @ (shrink * (u.T @ (targets - y_mean)))
    return weights, y_mean - x_mean @ weights, singular_values


def mse(features, targets, weights, bias):
    return np.mean((features @ weights + bias - targets) ** 2)


def kfold_indices(n_samples, n_folds, rng):
    order = rng.permutation(n_samples)
    return np.array_split(order, n_folds)


def cross_validate_alpha(features, targets, alphas, n_folds, rng):
    folds = kfold_indices(len(targets), n_folds, rng)
    scores = []
    for alpha in alphas:
        fold_errors = []
        for fold in folds:
            train_mask = np.ones(len(targets), dtype=bool)
            train_mask[fold] = False
            weights, bias = fit_ridge(features[train_mask], targets[train_mask], alpha)
            fold_errors.append(mse(features[fold], targets[fold], weights, bias))
        scores.append(np.mean(fold_errors))
    return np.array(scores)


def main():
    rng = np.random.default_rng(0)
    features, targets = make_collinear_data(60, rng)
    test_features, test_targets = make_collinear_data(500, rng)
    train_s, test_s = standardize(features, test_features)

    print("=== Step 1: OLS on nearly collinear features ===")
    ols_weights, ols_bias = fit_ridge(train_s, targets, 0.0)
    print(f"OLS weights a, b: {ols_weights[0]:8.3f} {ols_weights[1]:8.3f}")
    print(f"OLS test MSE: {mse(test_s, test_targets, ols_weights, ols_bias):.3f}")
    gram = (train_s - train_s.mean(axis=0)).T @ (train_s - train_s.mean(axis=0))
    print(f"condition number of X^T X: {np.linalg.cond(gram):.1f}")

    print("\n=== Step 2: ridge closed form for several alphas ===")
    print(f"{'alpha':>8} {'w_a':>8} {'w_b':>8} {'||w||':>8} {'test MSE':>9}")
    for alpha in [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        weights, bias = fit_ridge(train_s, targets, alpha)
        print(f"{alpha:>8} {weights[0]:>8.3f} {weights[1]:>8.3f} {np.linalg.norm(weights):>8.3f} {mse(test_s, test_targets, weights, bias):>9.3f}")

    print("\n=== Step 3: the SVD view of shrinkage ===")
    weights_svd, _, singular_values = fit_ridge_svd(train_s, targets, 10.0)
    weights_direct, _ = fit_ridge(train_s, targets, 10.0)
    print(f"SVD and direct solutions agree: {np.allclose(weights_svd, weights_direct)}")
    factors = singular_values ** 2 / (singular_values ** 2 + 10.0)
    print("shrink factor per direction:", np.round(factors, 3))

    print("\n=== Step 4: choose alpha with 5-fold cross-validation ===")
    alphas = np.logspace(-3, 3, 25)
    cv_scores = cross_validate_alpha(train_s, targets, alphas, 5, rng)
    best_alpha = alphas[np.argmin(cv_scores)]
    best_weights, best_bias = fit_ridge(train_s, targets, best_alpha)
    print(f"best alpha {best_alpha:.3f}, CV MSE {cv_scores.min():.3f}")
    print(f"test MSE with best alpha: {mse(test_s, test_targets, best_weights, best_bias):.3f}")

    print("\n=== Use It: scikit-learn Ridge and RidgeCV ===")
    ridge = Ridge(alpha=10.0).fit(train_s, targets)
    print(f"Ridge(alpha=10) matches ours: {np.allclose(ridge.coef_, weights_direct)}")
    ridge_cv = RidgeCV(alphas=alphas).fit(train_s, targets)
    print(f"RidgeCV picked alpha {ridge_cv.alpha_:.3f}, test R^2 {ridge_cv.score(test_s, test_targets):.4f}")


if __name__ == "__main__":
    main()
