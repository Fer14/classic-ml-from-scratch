import itertools
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

FEATURE_NAMES = ["linear", "sine", "inter_a", "inter_b", "noise"]


def make_data(n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n_samples, 5))
    y = 2.0 * X[:, 0] + 2.0 * np.sin(2 * X[:, 1]) + 1.5 * X[:, 2] * X[:, 3] + rng.normal(scale=0.3, size=n_samples)
    return X, y


def r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def permutation_importances(predict, X, y, n_repeats=10, seed=0):
    rng = np.random.default_rng(seed)
    baseline = r2(y, predict(X))
    importances = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for repeat in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, j] = rng.permutation(X_permuted[:, j])
            importances[j, repeat] = baseline - r2(y, predict(X_permuted))
    return importances.mean(axis=1), importances.std(axis=1)


def partial_dependence_curve(predict, X, feature, grid):
    averages = []
    for value in grid:
        X_fixed = X.copy()
        X_fixed[:, feature] = value
        averages.append(predict(X_fixed).mean())
    return np.array(averages)


def ice_curves(predict, X_rows, feature, grid):
    curves = np.zeros((len(X_rows), len(grid)))
    for k, value in enumerate(grid):
        X_fixed = X_rows.copy()
        X_fixed[:, feature] = value
        curves[:, k] = predict(X_fixed)
    return curves


def coalition_value(predict, x, background, coalition):
    X_mixed = background.copy()
    columns = list(coalition)
    X_mixed[:, columns] = x[columns]
    return predict(X_mixed).mean()


def exact_shapley_values(predict, x, background):
    n_features = len(x)
    values = {}
    for size in range(n_features + 1):
        for coalition in itertools.combinations(range(n_features), size):
            values[coalition] = coalition_value(predict, x, background, coalition)
    phi = np.zeros(n_features)
    for j in range(n_features):
        others = [k for k in range(n_features) if k != j]
        for size in range(n_features):
            weight = math.factorial(size) * math.factorial(n_features - size - 1) / math.factorial(n_features)
            for coalition in itertools.combinations(others, size):
                with_j = tuple(sorted(coalition + (j,)))
                phi[j] += weight * (values[with_j] - values[coalition])
    return phi, values[()]


def main():
    X, y = make_data()
    X_train, y_train, X_test, y_test = X[:1500], y[:1500], X[1500:], y[1500:]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0).fit(X_train, y_train)
    print("=== Model ===")
    print("true function: y = 2*linear + 2*sin(2*sine) + 1.5*inter_a*inter_b + noise_term (feature 'noise' unused)")
    print(f"HistGradientBoosting test R^2 = {r2(y_test, model.predict(X_test)):.3f}")

    print("\n=== Permutation importance (scratch, 10 repeats, test set) ===")
    mean_importance, std_importance = permutation_importances(model.predict, X_test, y_test)
    for j in np.argsort(-mean_importance):
        print(f"  {FEATURE_NAMES[j]:<8} drop in R^2 = {mean_importance[j]:.3f} +/- {std_importance[j]:.3f}")

    print("\n=== Partial dependence (scratch) ===")
    grid = np.linspace(-1.8, 1.8, 7)
    print("  grid:      " + " ".join(f"{v:+6.2f}" for v in grid))
    scratch_pd = {}
    for feature in [0, 1, 2, 4]:
        scratch_pd[feature] = partial_dependence_curve(model.predict, X_train, feature, grid)
        print(f"  {FEATURE_NAMES[feature]:<9}  " + " ".join(f"{v:+6.2f}" for v in scratch_pd[feature]))
    print("  inter_a looks flat on average, but the ICE curves below show it is not irrelevant:")
    rows_low_b = X_test[X_test[:, 3] < -1][:3]
    rows_high_b = X_test[X_test[:, 3] > 1][:3]
    for label, rows in [("inter_b < -1", rows_low_b), ("inter_b > +1", rows_high_b)]:
        curves = ice_curves(model.predict, rows, 2, grid)
        slopes = (curves[:, -1] - curves[:, 0]) / (grid[-1] - grid[0])
        print(f"    ICE slopes of inter_a when {label}: {np.round(slopes, 2)}")

    print("\n=== Exact Shapley values (scratch, 2^5 = 32 coalitions) ===")
    background = X_train[:200]
    x_explain = np.array([1.5, 0.8, 1.5, 1.5, 0.0])
    phi, base_value = exact_shapley_values(model.predict, x_explain, background)
    prediction = model.predict(x_explain[None, :])[0]
    print(f"  explaining x = {x_explain}")
    print(f"  base value E[f(X)] = {base_value:+.3f}")
    for name, contribution in zip(FEATURE_NAMES, phi):
        print(f"  {name:<8} phi = {contribution:+.3f}")
    print(f"  base + sum(phi) = {base_value + phi.sum():+.3f}   model prediction = {prediction:+.3f}")

    print("\n=== Use It: scikit-learn ===")
    sklearn_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, scoring="r2")
    for j in np.argsort(-sklearn_result.importances_mean):
        print(f"  permutation_importance {FEATURE_NAMES[j]:<8} {sklearn_result.importances_mean[j]:.3f} (scratch {mean_importance[j]:.3f})")
    sklearn_pd = partial_dependence(model, X_train, features=[1], grid_resolution=7, method="brute", custom_values={1: grid})
    print(f"  partial_dependence(sine) max abs diff vs scratch: {np.max(np.abs(sklearn_pd['average'][0] - scratch_pd[1])):.2e}")

    linear = LinearRegression().fit(X_train, y_train)
    phi_linear, base_linear = exact_shapley_values(linear.predict, x_explain, background)
    closed_form = linear.coef_ * (x_explain - background.mean(axis=0))
    print(f"  linear model: exact Shapley  = {" ".join(f"{v:+.3f}" for v in phi_linear)}")
    print(f"  linear model: w*(x - mean)   = {" ".join(f"{v:+.3f}" for v in closed_form)}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
