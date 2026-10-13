from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.differentiate import derivative


def forward_difference(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def cubic(x):
    return x**3 - 2 * x + 1


def cubic_derivative(x):
    return 3 * x**2 - 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tangent_line(f, df, x0):
    slope = df(x0)
    intercept = f(x0) - slope * x0
    return slope, intercept


def secant_slopes(f, x0, step_sizes):
    return [(h, (f(x0 + h) - f(x0)) / h) for h in step_sizes]


def main():
    print("=== Step 1: Slope of a secant line shrinking to a tangent ===")
    x0 = 1.5
    for h, slope in secant_slopes(cubic, x0, [1.0, 0.1, 0.01, 0.001, 0.0001]):
        print(f"  h = {h:<8} secant slope = {slope:.6f}")
    print(f"  exact derivative f'(1.5) = {cubic_derivative(x0):.6f}")

    print("\n=== Step 2: Numerical vs analytic derivatives ===")
    functions = {
        "x^3 - 2x + 1": (cubic, cubic_derivative),
        "sin(x)": (np.sin, np.cos),
        "exp(x)": (np.exp, np.exp),
        "log(x)": (np.log, lambda x: 1 / x),
        "sigmoid(x)": (sigmoid, sigmoid_derivative),
    }
    x_eval = 0.8
    print(f"  {'function':<14} {'analytic':>12} {'forward':>12} {'central':>12}")
    for name, (f, df) in functions.items():
        exact = df(x_eval)
        fwd = forward_difference(f, x_eval)
        ctr = central_difference(f, x_eval)
        print(f"  {name:<14} {exact:12.8f} {fwd:12.8f} {ctr:12.8f}")

    print("\n=== Step 3: How the step size h affects the error ===")
    exact = cubic_derivative(x0)
    print(f"  {'h':>8} {'forward error':>15} {'central error':>15}")
    for exponent in range(1, 13, 2):
        h = 10.0**-exponent
        fwd_err = abs(forward_difference(cubic, x0, h) - exact)
        ctr_err = abs(central_difference(cubic, x0, h) - exact)
        print(f"  {h:8.0e} {fwd_err:15.3e} {ctr_err:15.3e}")

    print("\n=== Step 4: Tangent line as a local linear approximation ===")
    slope, intercept = tangent_line(cubic, cubic_derivative, x0)
    print(f"  tangent at x = {x0}: y = {slope:.3f}x {intercept:+.3f}")
    for dx in [0.01, 0.1, 0.5, 1.0]:
        true_value = cubic(x0 + dx)
        approx_value = slope * (x0 + dx) + intercept
        print(f"  dx = {dx:<5} f = {true_value:9.4f} tangent = {approx_value:9.4f} gap = {abs(true_value - approx_value):.4f}")

    print("\n=== Step 5: Where is the derivative zero? ===")
    critical_points = np.array([-np.sqrt(2 / 3), np.sqrt(2 / 3)])
    for c in critical_points:
        print(f"  x = {c:+.4f}  f'(x) = {cubic_derivative(c):+.2e}  f(x) = {cubic(c):.4f}")

    print("\n=== Use It: NumPy and SciPy ===")
    grid = np.linspace(-2, 2, 401)
    numpy_slopes = np.gradient(cubic(grid), grid)
    max_gap = np.max(np.abs(numpy_slopes[1:-1] - cubic_derivative(grid[1:-1])))
    print(f"  np.gradient on a 401-point grid, max interior error: {max_gap:.2e}")
    result = derivative(cubic, x0)
    print(f"  scipy.differentiate.derivative at x = {x0}: {result.df:.10f} (exact {exact:.10f})")
    poly = np.polynomial.Polynomial([1, -2, 0, 1])
    print(f"  np.polynomial derivative coefficients: {poly.deriv().coef}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.linspace(-2, 2.5, 300)
    ax.plot(xs, cubic(xs), label="f(x) = x^3 - 2x + 1")
    ax.plot(xs, slope * xs + intercept, "--", label=f"tangent at x = {x0}")
    for h in [1.0, 0.5]:
        secant_slope = (cubic(x0 + h) - cubic(x0)) / h
        ax.plot(xs, cubic(x0) + secant_slope * (xs - x0), ":", label=f"secant, h = {h}")
    ax.scatter([x0], [cubic(x0)], color="black", zorder=3)
    ax.set_ylim(-4, 12)
    ax.legend()
    ax.set_title("Secants converge to the tangent")
    output_path = Path(__file__).parent / "derivatives.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {output_path.name}")


if __name__ == "__main__":
    main()
