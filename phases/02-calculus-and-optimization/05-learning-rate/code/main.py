import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor


def quadratic_trajectory(curvature, learning_rate, start=5.0, steps=12):
    positions = [start]
    for _ in range(steps):
        positions.append(positions[-1] - learning_rate * curvature * positions[-1])
    return np.array(positions)


def make_regression_data(seed=0, n_samples=400):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, 3)) * np.array([1.0, 3.0, 0.5])
    design = np.column_stack([np.ones(n_samples), features])
    true_weights = np.array([1.0, 2.0, -0.5, 3.0])
    targets = design @ true_weights + rng.normal(scale=0.3, size=n_samples)
    return features, design, targets


def mse(weights, design, targets):
    return np.mean((design @ weights - targets) ** 2)


def mse_gradient(weights, design, targets):
    return 2 * design.T @ (design @ weights - targets) / len(targets)


def run_gradient_descent(design, targets, schedule, steps=300):
    weights = np.zeros(design.shape[1])
    losses = []
    for step in range(steps):
        losses.append(mse(weights, design, targets))
        if not np.isfinite(losses[-1]) or losses[-1] > 1e12:
            break
        weights = weights - schedule(step) * mse_gradient(weights, design, targets)
    return weights, np.array(losses)


def constant(rate):
    return lambda step: rate


def step_decay(rate, drop=0.5, every=50):
    return lambda step: rate * drop ** (step // every)


def exponential_decay(rate, decay=0.99):
    return lambda step: rate * decay**step


def inverse_time_decay(rate, decay=0.05):
    return lambda step: rate / (1 + decay * step)


def backtracking_line_search(design, targets, steps=300, initial_rate=1.0, shrink=0.5, armijo=1e-4):
    weights = np.zeros(design.shape[1])
    losses, rates = [], []
    for _ in range(steps):
        loss = mse(weights, design, targets)
        gradient = mse_gradient(weights, design, targets)
        losses.append(loss)
        rate = initial_rate
        while mse(weights - rate * gradient, design, targets) > loss - armijo * rate * gradient @ gradient:
            rate *= shrink
        rates.append(rate)
        weights = weights - rate * gradient
    return weights, np.array(losses), np.array(rates)


def main():
    print("=== Step 1: Five regimes on f(x) = (L / 2) x^2 with L = 4 ===")
    curvature = 4.0
    for label, rate in [("too small", 0.02), ("good", 0.15), ("optimal 1/L", 0.25), ("oscillating", 0.45), ("diverging", 0.55)]:
        trajectory = quadratic_trajectory(curvature, rate)
        factor = 1 - rate * curvature
        shown = " ".join(f"{value:7.3f}" for value in trajectory[:6])
        print(f"  {label:<12} eta = {rate:<5} factor = {factor:+.2f}  x: {shown} ... {trajectory[-1]:.3g}")

    print("\n=== Step 2: The stability limit eta < 2 / L for linear regression ===")
    features, design, targets = make_regression_data()
    hessian = 2 * design.T @ design / len(targets)
    eigenvalues = np.linalg.eigvalsh(hessian)
    largest, smallest = eigenvalues[-1], eigenvalues[0]
    print(f"  Hessian eigenvalues: {np.round(eigenvalues, 3)}")
    print(f"  L = {largest:.3f}, stability limit 2/L = {2 / largest:.4f}, condition number = {largest / smallest:.1f}")
    for multiple in [0.1, 0.5, 1.0, 1.9, 2.1]:
        rate = multiple / largest
        _, losses = run_gradient_descent(design, targets, constant(rate))
        print(f"  eta = {multiple:>3}/L = {rate:.4f}: steps run = {len(losses):3d}, final loss = {losses[-1]:.4g}")

    print("\n=== Step 3: Learning rate schedules ===")
    base_rate = 1.9 / largest
    schedules = {
        "constant": constant(base_rate),
        "step decay": step_decay(base_rate),
        "exponential": exponential_decay(base_rate),
        "inverse time": inverse_time_decay(base_rate),
    }
    curves = {}
    for name, schedule in schedules.items():
        _, losses = run_gradient_descent(design, targets, schedule)
        curves[name] = losses
        print(f"  {name:<13} loss at step 10 = {losses[10]:.4f}, step 100 = {losses[100]:.6f}, step 299 = {losses[-1]:.6f}")

    print("\n=== Step 4: Backtracking line search picks the rate for you ===")
    weights, losses, rates = backtracking_line_search(design, targets)
    print(f"  final loss = {losses[-1]:.6f}")
    print(f"  chosen rates (first 8): {np.round(rates[:8], 4)}")
    print(f"  learned weights = {np.round(weights, 4)}")

    print("\n=== Use It: SGDRegressor learning_rate options ===")
    for strategy, eta0 in [("constant", 0.001), ("constant", 0.5), ("invscaling", 0.01), ("adaptive", 0.01)]:
        model = SGDRegressor(learning_rate=strategy, eta0=eta0, max_iter=200, tol=None, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                model.fit(features, targets)
                loss = np.mean((model.predict(features) - targets) ** 2)
                summary = f"loss = {loss:.4g}, coef = {np.round(model.coef_, 3)}" if loss < 1e6 else f"diverged, loss = {loss:.2e}"
            except ValueError as error:
                summary = f"failed: {str(error).splitlines()[0][:60]}"
        print(f"  {strategy:<10} eta0 = {eta0:<6} {summary}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, losses in curves.items():
        ax.semilogy(losses - losses.min() + 1e-12, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("loss - best loss")
    ax.set_title("Schedules starting from eta = 1.9 / L")
    ax.legend()
    output_path = Path(__file__).parent / "learning_rate_schedules.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {output_path.name}")


if __name__ == "__main__":
    main()
