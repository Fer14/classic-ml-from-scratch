import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = Path(__file__).parent / "out.png"


def determinant_2x2(matrix):
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]


def shoelace_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def determinant_cofactor(matrix):
    n = matrix.shape[0]
    if n == 1:
        return matrix[0, 0]
    if n == 2:
        return determinant_2x2(matrix)
    total = 0.0
    for column in range(n):
        minor = np.delete(np.delete(matrix, 0, axis=0), column, axis=1)
        sign = (-1) ** column
        total += sign * matrix[0, column] * determinant_cofactor(minor)
    return total


def determinant_elimination(matrix):
    working = matrix.astype(float).copy()
    n = working.shape[0]
    sign = 1.0
    for column in range(n):
        best_row = column + int(np.argmax(np.abs(working[column:, column])))
        if abs(working[best_row, column]) < 1e-12:
            return 0.0
        if best_row != column:
            working[[column, best_row]] = working[[best_row, column]]
            sign = -sign
        for row in range(column + 1, n):
            working[row] -= (working[row, column] / working[column, column]) * working[column]
    return sign * float(np.prod(np.diag(working)))


def rotation(angle_degrees):
    angle = math.radians(angle_degrees)
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


EXAMPLES = {
    "scale x3, y2": np.array([[3.0, 0.0], [0.0, 2.0]]),
    "shear": np.array([[1.0, 1.0], [0.0, 1.0]]),
    "rotate 40 deg": rotation(40),
    "reflect + stretch": np.array([[0.0, 2.0], [1.0, 0.0]]),
    "squash to a line": np.array([[1.0, 2.0], [0.5, 1.0]]),
    "general": np.array([[1.5, -0.5], [0.5, 1.0]]),
}
UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
TRIANGLE = np.array([[0.2, 0.1], [0.9, 0.3], [0.4, 0.8]])

print("Step 1: the determinant is the area scale factor")
for name, matrix in EXAMPLES.items():
    square_image = UNIT_SQUARE @ matrix.T
    triangle_ratio = shoelace_area(TRIANGLE @ matrix.T) / shoelace_area(TRIANGLE)
    print(f"  {name:<18} det = {determinant_2x2(matrix):7.3f}   unit square -> signed area "
          f"{shoelace_area(square_image):7.3f}   triangle area ratio {triangle_ratio:7.3f}")

print()
print("Step 2: the sign is orientation")
flip = EXAMPLES["reflect + stretch"]
print(f"  i_hat -> {flip[:, 0]}, j_hat -> {flip[:, 1]}: j_hat is now clockwise from i_hat, det = {determinant_2x2(flip)}")

print()
print("Step 3: det = 0 means space is squashed and the matrix has no inverse")
squash = EXAMPLES["squash to a line"]
print(f"  columns {squash[:, 0]} and {squash[:, 1]} are parallel, det = {determinant_2x2(squash)}")
print(f"  rank = {np.linalg.matrix_rank(squash)}")

print()
print("Step 4: 3x3 by cofactor expansion and by elimination")
matrix_3 = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
print(f"  cofactor expansion: {determinant_cofactor(matrix_3):.6f}")
print(f"  product of pivots:  {determinant_elimination(matrix_3):.6f}")
print("  |det| = volume of the parallelepiped spanned by the columns")

print()
print("Step 5: rules that follow from 'area scale factor'")
generator = np.random.default_rng(0)
a, b = generator.normal(size=(4, 4)), generator.normal(size=(4, 4))
det_a, det_b = determinant_elimination(a), determinant_elimination(b)
print(f"  det(AB) = {determinant_elimination(a @ b):.6f}, det(A) det(B) = {det_a * det_b:.6f}")
print(f"  det(A^-1) = {determinant_elimination(np.linalg.inv(a)):.6f}, 1 / det(A) = {1 / det_a:.6f}")
print(f"  det(A^T) = {determinant_elimination(a.T):.6f}, det(A) = {det_a:.6f}")
print(f"  det(2A) = {determinant_elimination(2 * a):.6f} = 2^4 det(A) = {16 * det_a:.6f}")

print()
print("Step 6: cofactor expansion explodes, elimination does not")
for size in (6, 8):
    big = generator.normal(size=(size, size))
    start = time.perf_counter()
    cofactor_value = determinant_cofactor(big)
    cofactor_seconds = time.perf_counter() - start
    start = time.perf_counter()
    elimination_value = determinant_elimination(big)
    elimination_seconds = time.perf_counter() - start
    print(f"  n={size}: cofactor {cofactor_seconds:.4f} s (~{math.factorial(size):,} terms), "
          f"elimination {elimination_seconds * 1000:.3f} ms, agree: {math.isclose(cofactor_value, elimination_value, rel_tol=1e-8)}")

print()
print("Use It: np.linalg.det and slogdet")
all_match = all(math.isclose(np.linalg.det(matrix), determinant_2x2(matrix), abs_tol=1e-12) for matrix in EXAMPLES.values())
print(f"  np.linalg.det matches all 2x2 examples: {all_match}")
print(f"  np.linalg.det matches 3x3: {math.isclose(np.linalg.det(matrix_3), determinant_elimination(matrix_3))}")
covariance = np.diag(np.full(200, 0.01))
print(f"  det of a 200-dim covariance with variances 0.01: {np.linalg.det(covariance)} (underflow)")
sign, log_determinant = np.linalg.slogdet(covariance)
print(f"  slogdet: sign {sign}, log|det| = {log_determinant:.3f} = 200 * log(0.01) = {200 * math.log(0.01):.3f}")

figure, axes_grid = plt.subplots(2, 3, figsize=(12, 8))
for axes, (name, matrix) in zip(axes_grid.flat, EXAMPLES.items()):
    axes.fill(UNIT_SQUARE[:, 0], UNIT_SQUARE[:, 1], color="lightgray", label="unit square (area 1)")
    image = UNIT_SQUARE @ matrix.T
    axes.fill(image[:, 0], image[:, 1], color="tab:blue", alpha=0.5, label="image")
    axes.quiver(0, 0, *matrix[:, 0], color="tab:green", angles="xy", scale_units="xy", scale=1, width=0.015)
    axes.quiver(0, 0, *matrix[:, 1], color="tab:red", angles="xy", scale_units="xy", scale=1, width=0.015)
    axes.set(xlim=(-1.5, 3.5), ylim=(-1.5, 3.5), title=f"{name}: det = {determinant_2x2(matrix):.2f}")
    axes.axhline(0, color="black", linewidth=0.5)
    axes.axvline(0, color="black", linewidth=0.5)
    axes.set_aspect("equal")
axes_grid[0, 0].legend(loc="upper left", fontsize=8)
figure.suptitle("The determinant is the signed area of the image of the unit square")
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
print(f"  saved {OUTPUT_PATH.name}")
