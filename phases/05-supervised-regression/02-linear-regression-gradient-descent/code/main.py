import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler


def make_data(n_samples, rng):
    size_m2 = rng.uniform(40, 220, n_samples)
    bedrooms = rng.integers(1, 6, n_samples).astype(float)
    age_years = rng.uniform(0, 60, n_samples)
    price_k = 30 + 2.5 * size_m2 + 12 * bedrooms - 0.8 * age_years + rng.normal(0, 15, n_samples)
    return np.column_stack([size_m2, bedrooms, age_years]), price_k


def standardize(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    return (train - mean) / std, (test - mean) / std, mean, std


def mse(features, targets, weights, bias):
    errors = features @ weights + bias - targets
    return np.mean(errors ** 2)


def gradients(features, targets, weights, bias):
    n_samples = features.shape[0]
    errors = features @ weights + bias - targets
    grad_weights = (2 / n_samples) * features.T @ errors
    grad_bias = (2 / n_samples) * errors.sum()
    return grad_weights, grad_bias


def batch_gradient_descent(features, targets, learning_rate, epochs):
    weights = np.zeros(features.shape[1])
    bias = 0.0
    history = []
    for _ in range(epochs):
        grad_weights, grad_bias = gradients(features, targets, weights, bias)
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
        history.append(mse(features, targets, weights, bias))
    return weights, bias, history


def minibatch_gradient_descent(features, targets, learning_rate, epochs, batch_size, rng):
    weights = np.zeros(features.shape[1])
    bias = 0.0
    history = []
    n_samples = features.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch = order[start:start + batch_size]
            grad_weights, grad_bias = gradients(features[batch], targets[batch], weights, bias)
            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias
        history.append(mse(features, targets, weights, bias))
    return weights, bias, history


def unscale(weights, bias, mean, std):
    raw_weights = weights / std
    raw_bias = bias - np.sum(weights * mean / std)
    return raw_weights, raw_bias


def main():
    rng = np.random.default_rng(42)
    features, prices = make_data(400, rng)
    train_x, test_x = features[:320], features[320:]
    train_y, test_y = prices[:320], prices[320:]

    print("=== Step 1: raw features with a tiny learning rate ===")
    raw_weights, raw_bias, raw_history = batch_gradient_descent(train_x, train_y, 1e-5, 200)
    print(f"MSE after 200 epochs on raw features: {raw_history[-1]:.1f}")
    _, _, exploding = batch_gradient_descent(train_x, train_y, 1e-3, 5)
    print(f"With lr=1e-3 on raw features MSE after 5 epochs: {exploding[-1]:.3e} (diverging)")

    print("\n=== Step 2: standardize, then batch gradient descent ===")
    train_s, test_s, mean, std = standardize(train_x, test_x)
    for learning_rate in [0.01, 0.1, 0.5, 1.05]:
        _, _, history = batch_gradient_descent(train_s, train_y, learning_rate, 200)
        print(f"lr={learning_rate:<5} MSE after 10: {history[9]:12.2f}   after 200: {history[-1]:12.2f}")

    print("\n=== Step 3: mini-batch gradient descent ===")
    mb_weights, mb_bias, mb_history = minibatch_gradient_descent(train_s, train_y, 0.05, 50, 32, rng)
    print(f"mini-batch (32) MSE after 50 epochs: {mb_history[-1]:.2f}")

    print("\n=== Step 4: compare with the closed form ===")
    weights, bias, history = batch_gradient_descent(train_s, train_y, 0.1, 500)
    design = np.hstack([np.ones((train_s.shape[0], 1)), train_s])
    exact = np.linalg.lstsq(design, train_y, rcond=None)[0]
    print(f"GD     bias {bias:8.3f} weights {np.round(weights, 3)}")
    print(f"exact  bias {exact[0]:8.3f} weights {np.round(exact[1:], 3)}")
    raw_w, raw_b = unscale(weights, bias, mean, std)
    print(f"in original units: intercept {raw_b:.3f}, coefs {np.round(raw_w, 3)}")
    test_mse = mse(test_s, test_y, weights, bias)
    print(f"test MSE {test_mse:.2f}")

    print("\n=== Use It: scikit-learn SGDRegressor ===")
    scaler = StandardScaler().fit(train_x)
    sgd = SGDRegressor(max_iter=1000, tol=1e-6, eta0=0.01, random_state=0)
    sgd.fit(scaler.transform(train_x), train_y)
    ols = LinearRegression().fit(scaler.transform(train_x), train_y)
    print(f"SGDRegressor     bias {sgd.intercept_[0]:8.3f} weights {np.round(sgd.coef_, 3)}")
    print(f"LinearRegression bias {ols.intercept_:8.3f} weights {np.round(ols.coef_, 3)}")
    print(f"SGD test R^2 {sgd.score(scaler.transform(test_x), test_y):.4f}")


if __name__ == "__main__":
    main()
