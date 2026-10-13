import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def true_function(x):
    return np.sin(2 * np.pi * x)


def make_data(n_samples, rng, noise=0.25):
    x = rng.uniform(0, 1, n_samples)
    return x, true_function(x) + rng.normal(0, noise, n_samples)


def polynomial_features(x, degree):
    return np.column_stack([x ** power for power in range(1, degree + 1)])


class RidgePolynomial:
    def __init__(self, degree, penalty):
        self.degree = degree
        self.penalty = penalty

    def fit(self, x, y):
        features = polynomial_features(x, self.degree)
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0)
        scaled = (features - self.feature_mean) / self.feature_std
        self.intercept = y.mean()
        gram = scaled.T @ scaled + self.penalty * np.eye(self.degree)
        self.weights = np.linalg.solve(gram, scaled.T @ (y - self.intercept))
        return self

    def predict(self, x):
        scaled = (polynomial_features(x, self.degree) - self.feature_mean) / self.feature_std
        return scaled @ self.weights + self.intercept


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def main():
    rng = np.random.default_rng(1)
    train_x, train_y = make_data(20, rng)
    test_x, test_y = make_data(1000, rng)

    print("=== Step 1: watch overfitting appear as capacity grows ===")
    for degree in (1, 3, 5, 9, 15):
        model = RidgePolynomial(degree, penalty=1e-10).fit(train_x, train_y)
        print(f"degree {degree:2d} | train MSE {mse(train_y, model.predict(train_x)):.4f} | test MSE {mse(test_y, model.predict(test_x)):9.4f} | max |w| {np.abs(model.weights).max():12.2f}")

    print("\n=== Step 2: L2 regularization tames a degree-15 polynomial ===")
    best_penalty, best_test = None, np.inf
    for penalty in (1e-10, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        model = RidgePolynomial(15, penalty).fit(train_x, train_y)
        test_error = mse(test_y, model.predict(test_x))
        if test_error < best_test:
            best_penalty, best_test = penalty, test_error
        print(f"lambda {penalty:8.0e} | train MSE {mse(train_y, model.predict(train_x)):.4f} | test MSE {test_error:9.4f} | ||w|| {np.linalg.norm(model.weights):12.2f}")
    print(f"Best lambda by test MSE (for illustration only): {best_penalty:g}")

    print("\n=== Step 3: more data is also a regularizer ===")
    for n_samples in (20, 50, 200, 1000):
        big_x, big_y = make_data(n_samples, np.random.default_rng(n_samples))
        model = RidgePolynomial(15, penalty=1e-10).fit(big_x, big_y)
        print(f"n={n_samples:5d} | degree 15, no penalty | train MSE {mse(big_y, model.predict(big_x)):.4f} | test MSE {mse(test_y, model.predict(test_x)):.4f}")

    print("\n=== Step 4: early stopping with gradient descent ===")
    features = polynomial_features(train_x, 15)
    mean, std = features.mean(axis=0), features.std(axis=0)
    scaled_train = (features - mean) / std
    scaled_test = (polynomial_features(test_x, 15) - mean) / std
    weights = np.zeros(15)
    intercept = train_y.mean()
    for epoch in range(1, 200001):
        residual = scaled_train @ weights + intercept - train_y
        weights -= 0.05 * scaled_train.T @ residual / len(train_y)
        if epoch in (10, 100, 1000, 10000, 100000, 200000):
            test_error = mse(test_y, scaled_test @ weights + intercept)
            print(f"epoch {epoch:6d} | train MSE {np.mean(residual ** 2):.4f} | test MSE {test_error:.4f}")

    print("\n=== Use It: scikit-learn ===")
    for alpha in (1e-10, 1e-2, 1.0):
        pipeline = make_pipeline(PolynomialFeatures(15, include_bias=False), StandardScaler(), Ridge(alpha=alpha))
        pipeline.fit(train_x[:, None], train_y)
        ours = RidgePolynomial(15, alpha).fit(train_x, train_y)
        sk_test = mse(test_y, pipeline.predict(test_x[:, None]))
        print(f"alpha {alpha:6.0e} | sklearn test MSE {sk_test:.4f} | ours {mse(test_y, ours.predict(test_x)):.4f}")


if __name__ == "__main__":
    main()
