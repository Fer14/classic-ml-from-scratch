from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import approx_fprime, check_grad


def bowl(point):
    x, y = point
    return x**2 + 3 * y**2 + x * y


def bowl_gradient(point):
    x, y = point
    return np.array([2 * x + y, 6 * y + x])


def partial_derivative(f, point, index, h=1e-5):
    step = np.zeros_like(point, dtype=float)
    step[index] = h
    return (f(point + step) - f(point - step)) / (2 * h)


def numerical_gradient(f, point, h=1e-5):
    point = np.asarray(point, dtype=float)
    return np.array([partial_derivative(f, point, i, h) for i in range(point.size)])


def directional_derivative(gradient, direction):
    unit = direction / np.linalg.norm(direction)
    return gradient @ unit


def mse_loss(weights, features, targets):
    residuals = features @ weights - targets
    return np.mean(residuals**2)


def mse_gradient(weights, features, targets):
    residuals = features @ weights - targets
    return 2 * features.T @ residuals / len(targets)


def main():
    point = np.array([1.0, 2.0])

    print("=== Step 1: Partial derivatives of f(x, y) = x^2 + 3y^2 + xy ===")
    df_dx = partial_derivative(bowl, point, 0)
    df_dy = partial_derivative(bowl, point, 1)
    print(f"  at (1, 2): df/dx numerical = {df_dx:.6f}, analytic 2x + y = {2 * 1 + 2}")
    print(f"  at (1, 2): df/dy numerical = {df_dy:.6f}, analytic 6y + x = {6 * 2 + 1}")

    print("\n=== Step 2: The gradient vector ===")
    grad = numerical_gradient(bowl, point)
    print(f"  numerical gradient = {grad}")
    print(f"  analytic gradient  = {bowl_gradient(point)}")
    print(f"  gradient norm (steepness) = {np.linalg.norm(grad):.4f}")

    print("\n=== Step 3: Directional derivatives, which way is steepest? ===")
    angles = np.deg2rad(np.arange(0, 360, 45))
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        rate = directional_derivative(grad, direction)
        print(f"  direction {np.rad2deg(angle):5.0f} deg -> rate of change {rate:+8.4f}")
    best_angle = np.rad2deg(np.arctan2(grad[1], grad[0]))
    print(f"  gradient points at {best_angle:.1f} deg with rate {np.linalg.norm(grad):.4f}")
    print(f"  negative gradient points at {best_angle - 180:.1f} deg with rate {-np.linalg.norm(grad):.4f}")

    print("\n=== Step 4: Gradient is perpendicular to level curves ===")
    tangent_direction = np.array([-grad[1], grad[0]])
    small_step = 1e-4 * tangent_direction / np.linalg.norm(tangent_direction)
    print(f"  f at point                = {bowl(point):.8f}")
    print(f"  f after tiny tangent step = {bowl(point + small_step):.8f}")
    print(f"  gradient . tangent = {grad @ tangent_direction:.2e}")

    print("\n=== Step 5: Gradient of the MSE loss for a linear model ===")
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 3
    features = rng.normal(size=(n_samples, n_features))
    true_weights = np.array([2.0, -1.0, 0.5])
    targets = features @ true_weights + rng.normal(scale=0.1, size=n_samples)
    weights = np.zeros(n_features)
    analytic = mse_gradient(weights, features, targets)
    numerical = numerical_gradient(lambda w: mse_loss(w, features, targets), weights)
    print(f"  analytic  = {np.round(analytic, 6)}")
    print(f"  numerical = {np.round(numerical, 6)}")
    relative_error = np.linalg.norm(analytic - numerical) / np.linalg.norm(analytic + numerical)
    print(f"  relative error = {relative_error:.2e}")
    print(f"  at the true weights, gradient = {np.round(mse_gradient(true_weights, features, targets), 4)}")

    print("\n=== Use It: SciPy ===")
    scipy_grad = approx_fprime(point, bowl, 1e-6)
    print(f"  scipy.optimize.approx_fprime = {scipy_grad}")
    print(f"  scipy.optimize.check_grad error = {check_grad(bowl, bowl_gradient, point):.2e}")
    mse_check = check_grad(mse_loss, mse_gradient, weights, features, targets)
    print(f"  check_grad on MSE gradient = {mse_check:.2e}")

    xs, ys = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    zs = xs**2 + 3 * ys**2 + xs * ys
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(xs, ys, zs, levels=20, cmap="viridis")
    grid_points = np.array([[a, b] for a in np.linspace(-2.5, 2.5, 8) for b in np.linspace(-2.5, 2.5, 8)])
    arrows = np.array([bowl_gradient(p) for p in grid_points])
    ax.quiver(grid_points[:, 0], grid_points[:, 1], -arrows[:, 0], -arrows[:, 1], color="crimson")
    ax.set_title("Negative gradient arrows cross level curves at right angles")
    ax.set_aspect("equal")
    output_path = Path(__file__).parent / "gradient_field.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {output_path.name}")


if __name__ == "__main__":
    main()
