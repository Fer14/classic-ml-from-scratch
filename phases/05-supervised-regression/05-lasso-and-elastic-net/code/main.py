import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LassoCV


def make_sparse_data(n_samples, n_features, rng):
    features = rng.normal(0, 1, (n_samples, n_features))
    true_weights = np.zeros(n_features)
    true_weights[:5] = [4.0, -3.0, 2.0, 1.5, -1.0]
    targets = features @ true_weights + rng.normal(0, 1.0, n_samples)
    return features, targets, true_weights


def soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def elastic_net_coordinate_descent(features, targets, alpha, l1_ratio, max_iter=1000, tol=1e-8):
    n_samples, n_features = features.shape
    x_mean = features.mean(axis=0)
    y_mean = targets.mean()
    centered_x = features - x_mean
    centered_y = targets - y_mean
    column_norms = (centered_x ** 2).sum(axis=0) / n_samples
    weights = np.zeros(n_features)
    residual = centered_y.copy()
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1 - l1_ratio)
    for _ in range(max_iter):
        largest_change = 0.0
        for j in range(n_features):
            old_weight = weights[j]
            residual += centered_x[:, j] * old_weight
            correlation = centered_x[:, j] @ residual / n_samples
            weights[j] = soft_threshold(correlation, l1_penalty) / (column_norms[j] + l2_penalty)
            residual -= centered_x[:, j] * weights[j]
            largest_change = max(largest_change, abs(weights[j] - old_weight))
        if largest_change < tol:
            break
    bias = y_mean - x_mean @ weights
    return weights, bias


def lasso(features, targets, alpha):
    return elastic_net_coordinate_descent(features, targets, alpha, 1.0)


def mse(features, targets, weights, bias):
    return np.mean((features @ weights + bias - targets) ** 2)


def main():
    rng = np.random.default_rng(1)
    features, targets, true_weights = make_sparse_data(80, 30, rng)
    test_x, test_y, _ = make_sparse_data(1000, 30, np.random.default_rng(2))

    print("=== Step 1: soft-thresholding ===")
    sample = np.array([-2.0, -0.5, 0.0, 0.3, 1.2])
    print(f"input            {sample}")
    print(f"S(input, 0.5)    {soft_threshold(sample, 0.5)}")

    print("\n=== Step 2: the lasso path ===")
    print(f"{'alpha':>6} {'nonzero':>8} {'test MSE':>9}  first 6 weights")
    for alpha in [0.001, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0]:
        weights, bias = lasso(features, targets, alpha)
        nonzero = np.sum(np.abs(weights) > 1e-10)
        print(f"{alpha:>6} {nonzero:>8} {mse(test_x, test_y, weights, bias):>9.3f}  {np.round(weights[:6], 2)}")
    print(f"true first 6 weights: {true_weights[:6]}")

    alpha_max = np.max(np.abs((features - features.mean(0)).T @ (targets - targets.mean()))) / len(targets)
    print(f"\nalpha_max (every weight is zero above this): {alpha_max:.3f}")

    print("\n=== Step 3: correlated twins, lasso vs elastic net ===")
    twin_rng = np.random.default_rng(3)
    base = twin_rng.normal(0, 1, 200)
    twins = np.column_stack([base + twin_rng.normal(0, 0.01, 200) for _ in range(3)])
    twin_targets = 3 * base + twin_rng.normal(0, 0.5, 200)
    lasso_twins, _ = elastic_net_coordinate_descent(twins, twin_targets, 0.1, 1.0)
    enet_twins, _ = elastic_net_coordinate_descent(twins, twin_targets, 0.1, 0.3)
    print(f"lasso       weights on 3 near-copies: {np.round(lasso_twins, 3)}")
    print(f"elastic net weights on 3 near-copies: {np.round(enet_twins, 3)}")

    print("\n=== Use It: scikit-learn Lasso, ElasticNet, LassoCV ===")
    ours, ours_bias = lasso(features, targets, 0.1)
    sk_lasso = Lasso(alpha=0.1, tol=1e-10, max_iter=100000).fit(features, targets)
    print(f"Lasso(alpha=0.1) max |ours - sklearn| = {np.max(np.abs(ours - sk_lasso.coef_)):.2e}")
    ours_enet, _ = elastic_net_coordinate_descent(features, targets, 0.1, 0.5)
    sk_enet = ElasticNet(alpha=0.1, l1_ratio=0.5, tol=1e-10, max_iter=100000).fit(features, targets)
    print(f"ElasticNet(0.1, 0.5) max |ours - sklearn| = {np.max(np.abs(ours_enet - sk_enet.coef_)):.2e}")
    lasso_cv = LassoCV(cv=5, random_state=0).fit(features, targets)
    selected = np.flatnonzero(lasso_cv.coef_)
    print(f"LassoCV alpha {lasso_cv.alpha_:.4f}, selected features {selected.tolist()}")
    print(f"LassoCV test R^2 {lasso_cv.score(test_x, test_y):.4f}")


if __name__ == "__main__":
    main()
