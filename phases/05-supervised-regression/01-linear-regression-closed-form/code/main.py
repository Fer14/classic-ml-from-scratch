import numpy as np
from sklearn.linear_model import LinearRegression


def make_house_data(n_samples, rng):
    size_m2 = rng.uniform(40, 220, n_samples)
    bedrooms = rng.integers(1, 6, n_samples).astype(float)
    age_years = rng.uniform(0, 60, n_samples)
    noise = rng.normal(0, 15, n_samples)
    price_k = 30 + 2.5 * size_m2 + 12 * bedrooms - 0.8 * age_years + noise
    features = np.column_stack([size_m2, bedrooms, age_years])
    return features, price_k


def add_intercept_column(features):
    ones = np.ones((features.shape[0], 1))
    return np.hstack([ones, features])


def fit_simple_line(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return intercept, slope


def fit_normal_equation(features, targets):
    design = add_intercept_column(features)
    gram = design.T @ design
    moment = design.T @ targets
    return np.linalg.solve(gram, moment)


def fit_lstsq(features, targets):
    design = add_intercept_column(features)
    coefficients, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    return coefficients


def predict(features, coefficients):
    return add_intercept_column(features) @ coefficients


def r_squared(targets, predictions):
    residual_sum = np.sum((targets - predictions) ** 2)
    total_sum = np.sum((targets - targets.mean()) ** 2)
    return 1 - residual_sum / total_sum


def rmse(targets, predictions):
    return np.sqrt(np.mean((targets - predictions) ** 2))


def main():
    rng = np.random.default_rng(42)
    features, prices = make_house_data(200, rng)
    split = 160
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = prices[:split], prices[split:]

    print("=== Step 1: one feature, the textbook formula ===")
    intercept, slope = fit_simple_line(train_x[:, 0], train_y)
    print(f"price_k = {intercept:.3f} + {slope:.3f} * size_m2")

    print("\n=== Step 2: all features, normal equation ===")
    beta = fit_normal_equation(train_x, train_y)
    names = ["intercept", "size_m2", "bedrooms", "age_years"]
    true_values = [30, 2.5, 12, -0.8]
    for name, value, truth in zip(names, beta, true_values):
        print(f"{name:>10}: learned {value:8.3f}   true {truth:6.2f}")

    print("\n=== Step 3: numerically safer least squares ===")
    beta_lstsq = fit_lstsq(train_x, train_y)
    print(f"max |normal - lstsq| = {np.max(np.abs(beta - beta_lstsq)):.2e}")

    print("\n=== Step 4: evaluate on held-out data ===")
    test_pred = predict(test_x, beta)
    print(f"test R^2  = {r_squared(test_y, test_pred):.4f}")
    print(f"test RMSE = {rmse(test_y, test_pred):.3f} (noise std was 15)")

    print("\n=== Step 5: the residuals are orthogonal to every column ===")
    design = add_intercept_column(train_x)
    residuals = train_y - design @ beta
    print("X^T r =", np.round(design.T @ residuals, 6))

    print("\n=== Use It: scikit-learn ===")
    model = LinearRegression().fit(train_x, train_y)
    print(f"sklearn intercept {model.intercept_:.3f}, coefs {np.round(model.coef_, 3)}")
    print(f"sklearn test R^2  {model.score(test_x, test_y):.4f}")
    same = np.allclose(np.r_[model.intercept_, model.coef_], beta)
    print(f"matches from-scratch solution: {same}")


if __name__ == "__main__":
    main()
