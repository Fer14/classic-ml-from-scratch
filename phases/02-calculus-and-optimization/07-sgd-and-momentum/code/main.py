from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor


def make_data(n_samples=20_000, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, 5))
    true_weights = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    targets = features @ true_weights + 4.0 + rng.normal(scale=1.0, size=n_samples)
    design = np.column_stack([np.ones(n_samples), features])
    return features, design, targets


def mse(weights, design, targets):
    return np.mean((design @ weights - targets) ** 2)


def batch_gradient(weights, design, targets):
    return 2 * design.T @ (design @ weights - targets) / len(targets)


def minibatch_sgd(design, targets, batch_size, learning_rate, epochs, decay=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = len(targets)
    weights = np.zeros(design.shape[1])
    history, samples_seen, step = [], 0, 0
    for _ in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch = order[start:start + batch_size]
            rate = learning_rate / (1 + decay * step)
            weights = weights - rate * batch_gradient(weights, design[batch], targets[batch])
            samples_seen += len(batch)
            step += 1
            if step % max(1, 2000 // batch_size) == 0:
                history.append((samples_seen, mse(weights, design, targets)))
    return weights, np.array(history)


def momentum_descent(gradient_fn, start, learning_rate, beta, steps, nesterov=False):
    position = np.array(start, dtype=float)
    velocity = np.zeros_like(position)
    path = [position.copy()]
    for _ in range(steps):
        lookahead = position + beta * velocity if nesterov else position
        velocity = beta * velocity - learning_rate * gradient_fn(lookahead)
        position = position + velocity
        path.append(position.copy())
    return np.array(path)


def ravine_gradient(point):
    return np.array([0.02 * point[0], 2.0 * point[1]])


def ravine(point):
    return 0.01 * point[0] ** 2 + point[1] ** 2


def steps_to_reach(path, tolerance=1e-6):
    values = np.array([ravine(p) for p in path])
    hits = np.nonzero(values < tolerance)[0]
    return int(hits[0]) if hits.size else None


def main():
    features, design, targets = make_data()
    best_loss = mse(np.linalg.lstsq(design, targets, rcond=None)[0], design, targets)
    print(f"Dataset: {len(targets)} samples, 5 features. Best possible MSE = {best_loss:.5f}")

    print("\n=== Step 1: Batch GD vs mini-batch SGD vs pure SGD, same data budget ===")
    budget_epochs = 2
    runs = {
        "batch (n=20000)": (len(targets), 0.3),
        "mini-batch (32)": (32, 0.05),
        "pure SGD (1)": (1, 0.005),
    }
    histories = {}
    for name, (batch_size, rate) in runs.items():
        weights, history = minibatch_sgd(design, targets, batch_size, rate, budget_epochs)
        histories[name] = history
        print(f"  {name:<16} updates = {2 * int(np.ceil(len(targets) / batch_size)):6d}  final MSE = {mse(weights, design, targets):.5f}")

    print("\n=== Step 2: Constant step size leaves a noise floor ===")
    for label, decay in [("constant eta", 0.0), ("decaying eta", 0.002)]:
        weights, history = minibatch_sgd(design, targets, 8, 0.05, 3, decay=decay)
        tail = history[-20:, 1]
        print(f"  {label:<13} last 20 checkpoints: mean excess = {tail.mean() - best_loss:.2e}, std = {tail.std():.2e}")

    print("\n=== Step 3: Momentum on a ravine f = 0.01 x^2 + y^2 (condition number 100) ===")
    start = [10.0, 1.0]
    configurations = [
        ("plain GD", 0.5, 0.0, False),
        ("plain GD", 0.9, 0.0, False),
        ("momentum 0.9", 0.5, 0.9, False),
        ("Nesterov 0.9", 0.5, 0.9, True),
    ]
    for label, rate, beta, nesterov in configurations:
        path = momentum_descent(ravine_gradient, start, rate, beta, 3000, nesterov)
        print(f"  {label:<13} eta = {rate}: f after 50 steps = {ravine(path[50]):.2e}, steps to f < 1e-6 = {steps_to_reach(path)}")
    path = momentum_descent(ravine_gradient, start, 0.5, 0.9, 8)
    print("  momentum x-coordinate:", " ".join(f"{p[0]:.2f}" for p in path))
    path = momentum_descent(ravine_gradient, start, 0.5, 0.0, 8)
    print("  plain GD x-coordinate:", " ".join(f"{p[0]:.2f}" for p in path))

    print("\n=== Step 4: With SGD, momentum multiplies the effective step by 1 / (1 - beta) ===")
    for beta, rate in [(0.0, 0.01), (0.9, 0.01), (0.9, 0.001)]:
        rng = np.random.default_rng(1)
        weights = np.zeros(design.shape[1])
        velocity = np.zeros_like(weights)
        tail = []
        for step in range(1500):
            batch = rng.integers(0, len(targets), size=16)
            velocity = beta * velocity - rate * batch_gradient(weights, design[batch], targets[batch])
            weights = weights + velocity
            if step >= 1000:
                tail.append(mse(weights, design, targets))
        print(f"  beta = {beta}, eta = {rate:<5} effective eta = {rate / (1 - beta):.3f}: mean excess MSE over last 500 steps = {np.mean(tail) - best_loss:.2e}")

    print("\n=== Use It: scikit-learn SGDRegressor ===")
    reference = LinearRegression().fit(features, targets)
    model = SGDRegressor(learning_rate="invscaling", eta0=0.01, max_iter=5, tol=None, random_state=0).fit(features, targets)
    print(f"  LinearRegression coef = {np.round(reference.coef_, 3)}, intercept = {reference.intercept_:.3f}")
    print(f"  SGDRegressor     coef = {np.round(model.coef_, 3)}, intercept = {model.intercept_[0]:.3f}")
    streaming = SGDRegressor(learning_rate="invscaling", eta0=0.01, random_state=0)
    for chunk in np.array_split(np.arange(len(targets)), 20):
        streaming.partial_fit(features[chunk], targets[chunk])
    print(f"  partial_fit on 20 chunks, MSE = {np.mean((streaming.predict(features) - targets) ** 2):.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        axes[0].semilogy(history[:, 0], history[:, 1] - best_loss + 1e-6, label=name)
    axes[0].set_xlabel("samples processed")
    axes[0].set_ylabel("MSE - best MSE")
    axes[0].legend()
    xs, ys = np.meshgrid(np.linspace(-11, 11, 200), np.linspace(-2, 2, 200))
    axes[1].contour(xs, ys, 0.01 * xs**2 + ys**2, levels=20, cmap="viridis")
    for label, beta in [("plain GD", 0.0), ("momentum 0.9", 0.9)]:
        path = momentum_descent(ravine_gradient, start, 0.5, beta, 60)
        axes[1].plot(path[:, 0], path[:, 1], "o-", markersize=2, label=label)
    axes[1].legend()
    output_path = Path(__file__).parent / "sgd_and_momentum.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {output_path.name}")


if __name__ == "__main__":
    main()
