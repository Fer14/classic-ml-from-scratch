import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

NOISE_STD = 0.3


def true_function(x):
    return np.sin(2 * np.pi * x)


def sample_training_set(n_samples, rng):
    x = rng.uniform(0, 1, n_samples)
    return x, true_function(x) + rng.normal(0, NOISE_STD, n_samples)


def fit_polynomial(x, y, degree):
    return np.polyfit(x, y, degree)


def decompose(predictions, test_x):
    mean_prediction = predictions.mean(axis=0)
    bias_squared = np.mean((mean_prediction - true_function(test_x)) ** 2)
    variance = np.mean(predictions.var(axis=0))
    return bias_squared, variance


def expected_test_mse(predictions, test_x, rng):
    noisy_targets = true_function(test_x) + rng.normal(0, NOISE_STD, predictions.shape)
    return np.mean((predictions - noisy_targets) ** 2)


def simulate(model_fn, n_datasets, n_samples, test_x, rng):
    predictions = np.empty((n_datasets, len(test_x)))
    for i in range(n_datasets):
        x, y = sample_training_set(n_samples, rng)
        predictions[i] = model_fn(x, y, test_x)
    return predictions


def polynomial_model(degree):
    def model_fn(x, y, test_x):
        return np.polyval(fit_polynomial(x, y, degree), test_x)
    return model_fn


def tree_model(depth, bagged=False):
    def model_fn(x, y, test_x):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=0)
        model = BaggingRegressor(tree, n_estimators=30, random_state=0) if bagged else tree
        return model.fit(x[:, None], y).predict(test_x[:, None])
    return model_fn


def main():
    rng = np.random.default_rng(0)
    test_x = np.linspace(0.05, 0.95, 50)
    noise_variance = NOISE_STD ** 2

    print("=== Step 1: one test point, many training sets ===")
    point = np.array([0.25])
    for degree in (1, 9):
        predictions = simulate(polynomial_model(degree), 200, 25, point, rng)[:, 0]
        print(f"degree {degree}: true f(0.25)={true_function(0.25):.3f}, mean prediction={predictions.mean():.3f}, std of predictions={predictions.std():.3f}")

    print("\n=== Step 2: decompose expected error for polynomial degrees ===")
    print("degree |  bias^2  | variance |  noise   |   sum    | measured MSE")
    for degree in (0, 1, 3, 5, 7, 9):
        predictions = simulate(polynomial_model(degree), 300, 25, test_x, rng)
        bias_squared, variance = decompose(predictions, test_x)
        total = bias_squared + variance + noise_variance
        measured = expected_test_mse(predictions, test_x, rng)
        print(f"  {degree:2d}   | {bias_squared:8.4f} | {variance:8.4f} | {noise_variance:8.4f} | {total:8.4f} | {measured:8.4f}")

    print("\n=== Step 3: more data shrinks variance, not bias ===")
    for n_samples in (25, 100, 400):
        for degree in (1, 9):
            predictions = simulate(polynomial_model(degree), 200, n_samples, test_x, rng)
            bias_squared, variance = decompose(predictions, test_x)
            print(f"n={n_samples:4d} degree {degree}: bias^2={bias_squared:.4f} variance={variance:.4f}")

    print("\n=== Use It: scikit-learn trees and bagging ===")
    for depth in (1, 3, 6, None):
        predictions = simulate(tree_model(depth), 100, 50, test_x, rng)
        bias_squared, variance = decompose(predictions, test_x)
        print(f"tree depth {str(depth):>4s}: bias^2={bias_squared:.4f} variance={variance:.4f} total={bias_squared + variance + noise_variance:.4f}")
    predictions = simulate(tree_model(None, bagged=True), 100, 50, test_x, rng)
    bias_squared, variance = decompose(predictions, test_x)
    print(f"bagged deep trees: bias^2={bias_squared:.4f} variance={variance:.4f} total={bias_squared + variance + noise_variance:.4f}")


if __name__ == "__main__":
    main()
