from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping, minimize


def chord_test(f, low, high, dimension, trials=20_000, seed=0):
    rng = np.random.default_rng(seed)
    worst_gap = np.inf
    for _ in range(trials):
        a = rng.uniform(low, high, size=dimension)
        b = rng.uniform(low, high, size=dimension)
        t = rng.uniform()
        gap = t * f(a) + (1 - t) * f(b) - f(t * a + (1 - t) * b)
        worst_gap = min(worst_gap, gap)
    return worst_gap


def squared(point):
    return float(np.sum(point**2))


def absolute(point):
    return float(np.sum(np.abs(point)))


def log_sum_exp(point):
    return float(np.log(np.sum(np.exp(point))))


def wavy(point):
    return float(np.sum(point**2 + 3 * np.sin(2 * point)))


def double_well(x):
    return x**4 - 3 * x**2 + x


def double_well_gradient(x):
    return 4 * x**3 - 6 * x + 1


def descend(gradient_fn, start, learning_rate=0.01, steps=2000):
    x = start
    for _ in range(steps):
        x = x - learning_rate * gradient_fn(x)
    return x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_hessian(weights, features):
    probabilities = sigmoid(features @ weights)
    return features.T @ (features * (probabilities * (1 - probabilities))[:, None]) / len(features)


def main():
    print("=== Step 1: Chord test, f(t a + (1-t) b) <= t f(a) + (1-t) f(b) ===")
    candidates = {"sum of squares": squared, "sum of |x|": absolute, "log-sum-exp": log_sum_exp, "x^2 + 3 sin(2x)": wavy}
    for name, f in candidates.items():
        gap = chord_test(f, -3, 3, dimension=2)
        verdict = "convex (no violation found)" if gap >= -1e-12 else "NOT convex"
        print(f"  {name:<16} smallest chord gap = {gap:+.2e}  -> {verdict}")

    print("\n=== Step 2: Second derivative test in 1D ===")
    grid = np.linspace(-2, 2, 9)
    second_derivative = 12 * grid**2 - 6
    for x, value in zip(grid, second_derivative):
        print(f"  x = {x:+.1f}  f''(x) = {value:+6.2f}  {'curves up' if value >= 0 else 'curves DOWN'}")

    print("\n=== Step 3: Non-convex means the start point matters ===")
    for start in [-2.0, -0.5, 0.5, 2.0]:
        end = descend(double_well_gradient, start)
        print(f"  start {start:+.1f} -> x = {end:+.4f}, f = {double_well(end):+.4f}")

    print("\n=== Step 4: Hessian eigenvalues certify convexity of ML losses ===")
    rng = np.random.default_rng(1)
    features = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    mse_hessian = 2 * features.T @ features / len(features)
    print(f"  MSE Hessian eigenvalues: {np.round(np.linalg.eigvalsh(mse_hessian), 4)}")
    for trial in range(3):
        weights = rng.normal(scale=3, size=4)
        eigenvalues = np.linalg.eigvalsh(logistic_hessian(weights, features))
        print(f"  logistic Hessian at random w #{trial}: min eigenvalue = {eigenvalues.min():.5f}")
    duplicated = np.column_stack([features, features[:, 1]])
    eigenvalues = np.linalg.eigvalsh(2 * duplicated.T @ duplicated / len(duplicated))
    print(f"  MSE Hessian with a duplicated feature: smallest eigenvalue = {eigenvalues.min():.2e} (convex, not strictly)")

    print("\n=== Step 5: Convex loss, every start reaches the same minimum ===")
    targets = features @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(scale=0.2, size=200)
    for trial in range(3):
        weights = rng.normal(scale=10, size=4)
        for _ in range(500):
            weights = weights - 0.1 * 2 * features.T @ (features @ weights - targets) / len(targets)
        print(f"  random start #{trial} -> {np.round(weights, 4)}")

    print("\n=== Use It: SciPy local vs global optimizers ===")
    for start in [-2.0, 2.0]:
        result = minimize(lambda v: double_well(v[0]), [start], method="BFGS")
        print(f"  minimize (BFGS) from {start:+.1f}: x = {result.x[0]:+.4f}, f = {result.fun:+.4f}")
    result = basinhopping(lambda v: double_well(v[0]), [2.0], niter=100, stepsize=1.5, seed=0)
    print(f"  basinhopping from +2.0:  x = {result.x[0]:+.4f}, f = {result.fun:+.4f}")

    xs = np.linspace(-2.2, 2.2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, xs**2)
    axes[0].plot([-2, 1.5], [4, 2.25], "--", color="gray")
    axes[0].set_title("Convex: every chord lies above the curve")
    axes[1].plot(xs, double_well(xs))
    axes[1].plot([-1.3, 1.13], [double_well(-1.3), double_well(1.13)], "--", color="gray")
    axes[1].set_title("Non-convex: some chords dip below")
    output_path = Path(__file__).parent / "convexity.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {output_path.name}")


if __name__ == "__main__":
    main()
