from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def gradient_descent(gradient_fn, start, learning_rate, max_steps=10_000, tolerance=1e-8):
    position = np.array(start, dtype=float)
    path = [position.copy()]
    for step in range(max_steps):
        gradient = gradient_fn(position)
        if np.linalg.norm(gradient) < tolerance:
            break
        position = position - learning_rate * gradient
        path.append(position.copy())
    return position, np.array(path), step


def parabola(x):
    return (x - 3) ** 2 + 1


def parabola_gradient(x):
    return 2 * (x - 3)


def elongated_bowl(point):
    x, y = point
    return x**2 + 10 * y**2


def elongated_bowl_gradient(point):
    x, y = point
    return np.array([2 * x, 20 * y])


def add_bias_column(features):
    return np.column_stack([np.ones(len(features)), features])


def mse(weights, design, targets):
    return np.mean((design @ weights - targets) ** 2)


def mse_gradient(weights, design, targets):
    return 2 * design.T @ (design @ weights - targets) / len(targets)


def main():
    print("=== Step 1: Gradient descent on f(x) = (x - 3)^2 + 1 ===")
    x = np.array([-4.0])
    for step in range(8):
        gradient = parabola_gradient(x)
        print(f"  step {step}: x = {x[0]:8.4f}  f(x) = {parabola(x)[0]:8.4f}  gradient = {gradient[0]:8.4f}")
        x = x - 0.2 * gradient
    final, _, steps = gradient_descent(parabola_gradient, [-4.0], 0.2)
    print(f"  converged to x = {final[0]:.6f} after {steps} steps")

    print("\n=== Step 2: Gradient descent in 2D on x^2 + 10y^2 ===")
    final, path, steps = gradient_descent(elongated_bowl_gradient, [8.0, 2.0], 0.04)
    print(f"  start (8, 2) -> end ({final[0]:.2e}, {final[1]:.2e}) in {steps} steps")
    for index in [0, 1, 2, 3, 10, 50, 100]:
        point = path[index]
        print(f"  step {index:3d}: ({point[0]:7.4f}, {point[1]:7.4f}) f = {elongated_bowl(point):.6f}")

    print("\n=== Step 3: Linear regression by gradient descent ===")
    rng = np.random.default_rng(42)
    n_samples = 300
    features = rng.uniform(-2, 2, size=(n_samples, 2))
    true_weights = np.array([4.0, 2.5, -1.5])
    design = add_bias_column(features)
    targets = design @ true_weights + rng.normal(scale=0.5, size=n_samples)

    weights = np.zeros(3)
    for epoch in range(101):
        if epoch in (0, 5, 10, 20, 50, 100):
            print(f"  epoch {epoch:3d}: loss = {mse(weights, design, targets):.5f} weights = {np.round(weights, 4)}")
        weights = weights - 0.1 * mse_gradient(weights, design, targets)

    print("\n=== Step 4: Stopping when the gradient is small ===")
    learned, _, steps = gradient_descent(lambda w: mse_gradient(w, design, targets), np.zeros(3), 0.1, tolerance=1e-6)
    print(f"  stopped after {steps} steps with |gradient| < 1e-6")
    print(f"  learned weights = {np.round(learned, 4)}")
    closed_form = np.linalg.solve(design.T @ design, design.T @ targets)
    print(f"  closed form     = {np.round(closed_form, 4)}")
    print(f"  max difference  = {np.max(np.abs(learned - closed_form)):.2e}")

    print("\n=== Use It: scikit-learn and SciPy ===")
    model = LinearRegression().fit(features, targets)
    print(f"  LinearRegression: intercept = {model.intercept_:.4f}, coef = {np.round(model.coef_, 4)}")
    result = minimize(mse, np.zeros(3), args=(design, targets), jac=mse_gradient, method="BFGS")
    print(f"  scipy BFGS: {np.round(result.x, 4)} in {result.nit} iterations ({result.njev} gradient calls)")
    result_cg = minimize(mse, np.zeros(3), args=(design, targets), jac=mse_gradient, method="CG")
    print(f"  scipy CG:   {np.round(result_cg.x, 4)} in {result_cg.nit} iterations")

    xs, ys = np.meshgrid(np.linspace(-9, 9, 200), np.linspace(-3, 3, 200))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.contour(xs, ys, xs**2 + 10 * ys**2, levels=25, cmap="viridis")
    _, path, _ = gradient_descent(elongated_bowl_gradient, [8.0, 2.0], 0.04, max_steps=60)
    ax.plot(path[:, 0], path[:, 1], "o-", color="crimson", markersize=3)
    ax.set_title("Gradient descent path on x^2 + 10y^2")
    output_path = Path(__file__).parent / "gradient_descent_path.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {output_path.name}")


if __name__ == "__main__":
    main()
