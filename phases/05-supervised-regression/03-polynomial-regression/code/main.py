from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def true_function(x):
    return np.sin(1.5 * x) + 0.3 * x


def make_data(n_samples, rng):
    x = rng.uniform(-3, 3, n_samples)
    y = true_function(x) + rng.normal(0, 0.3, n_samples)
    return x, y


def polynomial_features(x, degree):
    return np.column_stack([x ** power for power in range(degree + 1)])


def fit_polynomial(x, y, degree):
    design = polynomial_features(x / 3.0, degree)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def predict_polynomial(x, coefficients):
    degree = len(coefficients) - 1
    return polynomial_features(x / 3.0, degree) @ coefficients


def mse(y, predictions):
    return np.mean((y - predictions) ** 2)


def main():
    rng = np.random.default_rng(7)
    train_x, train_y = make_data(30, rng)
    valid_x, valid_y = make_data(200, rng)

    print("=== Step 1: the Vandermonde design matrix ===")
    print(polynomial_features(np.array([1.0, 2.0, 3.0]), 3))

    print("\n=== Step 2: sweep the degree ===")
    print(f"{'degree':>6} {'train MSE':>10} {'valid MSE':>10} {'max |coef|':>11}")
    results = {}
    for degree in [0, 1, 2, 3, 5, 7, 9, 12, 15]:
        coefficients = fit_polynomial(train_x, train_y, degree)
        train_error = mse(train_y, predict_polynomial(train_x, coefficients))
        valid_error = mse(valid_y, predict_polynomial(valid_x, coefficients))
        results[degree] = (coefficients, train_error, valid_error)
        print(f"{degree:>6} {train_error:>10.4f} {valid_error:>10.4f} {np.max(np.abs(coefficients)):>11.1f}")

    best_degree = min(results, key=lambda d: results[d][2])
    print(f"\nbest degree by validation MSE: {best_degree}")
    print("noise variance (irreducible error): 0.09")

    print("\n=== Step 3: plot underfit, good fit, overfit ===")
    grid = np.linspace(-3, 3, 300)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, degree in zip(axes, [1, best_degree, 15]):
        axis.scatter(train_x, train_y, s=15, color="black", label="train")
        axis.plot(grid, true_function(grid), "--", color="gray", label="truth")
        axis.plot(grid, predict_polynomial(grid, results[degree][0]), color="tab:blue", label=f"degree {degree}")
        axis.set_ylim(-3, 3)
        axis.set_title(f"degree {degree}: valid MSE {results[degree][2]:.3f}")
        axis.legend(loc="upper left")
    figure.tight_layout()
    output_path = Path(__file__).parent / "polynomial_fits.png"
    figure.savefig(output_path, dpi=100)
    print(f"saved {output_path.name}")

    print("\n=== Use It: PolynomialFeatures + LinearRegression ===")
    model = make_pipeline(
        PolynomialFeatures(degree=best_degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(train_x.reshape(-1, 1), train_y)
    sk_valid = mse(valid_y, model.predict(valid_x.reshape(-1, 1)))
    print(f"sklearn degree {best_degree} valid MSE {sk_valid:.4f}  (ours {results[best_degree][2]:.4f})")


if __name__ == "__main__":
    main()
