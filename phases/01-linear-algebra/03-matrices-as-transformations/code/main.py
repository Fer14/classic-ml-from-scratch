import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

OUTPUT_PATH = Path(__file__).parent / "out.png"


def matrix_from_basis_images(i_hat_lands_at, j_hat_lands_at):
    return np.column_stack([i_hat_lands_at, j_hat_lands_at]).astype(float)


def apply_column_picture(matrix, vector):
    result = np.zeros(matrix.shape[0])
    for column_index, coefficient in enumerate(vector):
        result = result + coefficient * matrix[:, column_index]
    return result


def apply_row_picture(matrix, vector):
    return np.array([sum(row[k] * vector[k] for k in range(len(vector))) for row in matrix])


def apply_to_points(matrix, points):
    return np.array([apply_column_picture(matrix, point) for point in points])


def rotation(angle_degrees):
    angle = math.radians(angle_degrees)
    return matrix_from_basis_images([math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)])


def translate(vector):
    return vector + np.array([1.0, 1.0])


def square_each(vector):
    return vector ** 2


def is_linear(transform, first, second, scalar=2.5):
    additive = np.allclose(transform(first + second), transform(first) + transform(second))
    homogeneous = np.allclose(transform(scalar * first), scalar * transform(first))
    return additive and homogeneous


TRANSFORMATIONS = {
    "identity": matrix_from_basis_images([1, 0], [0, 1]),
    "rotate 30 deg": rotation(30),
    "scale x2, y0.5": matrix_from_basis_images([2, 0], [0, 0.5]),
    "shear": matrix_from_basis_images([1, 0], [1, 1]),
    "reflect over y=x": matrix_from_basis_images([0, 1], [1, 0]),
    "project onto x-axis": matrix_from_basis_images([1, 0], [0, 0]),
}

print("Step 1: a matrix is where the basis vectors land")
shear = TRANSFORMATIONS["shear"]
print(f"  shear sends i_hat -> {shear[:, 0]}, j_hat -> {shear[:, 1]}")
print(f"  so the shear matrix is\n{shear}")

print()
print("Step 2: matrix times vector, two pictures")
vector = np.array([2.0, 3.0])
column_result = apply_column_picture(shear, vector)
row_result = apply_row_picture(shear, vector)
print(f"  column picture: 2 * {shear[:, 0]} + 3 * {shear[:, 1]} = {column_result}")
print(f"  row picture:    [row0 . v, row1 . v] = {row_result}")

print()
print("Step 3: the catalogue of 2-D transformations")
probe = np.array([2.0, 1.0])
for name, matrix in TRANSFORMATIONS.items():
    image = apply_column_picture(matrix, probe)
    print(f"  {name:<20} columns {np.round(matrix[:, 0], 3)} {np.round(matrix[:, 1], 3)}  "
          f"(2, 1) -> {np.round(image, 3) + 0.0}")

print()
print("Step 4: linear means grid lines stay parallel and evenly spaced, origin stays put")
first = np.array([1.0, -2.0])
second = np.array([0.5, 4.0])
for name, matrix in TRANSFORMATIONS.items():
    print(f"  {name:<20} linear: {is_linear(lambda x: apply_column_picture(matrix, x), first, second)}")
print(f"  {'translate by (1,1)':<20} linear: {is_linear(translate, first, second)}  (origin moves to {translate(np.zeros(2))})")
print(f"  {'square each entry':<20} linear: {is_linear(square_each, first, second)}  (grid lines curve)")

print()
print("Step 5: rotations keep lengths, projections destroy information")
rotated = apply_column_picture(TRANSFORMATIONS["rotate 30 deg"], vector)
print(f"  |v| = {np.linalg.norm(vector):.4f}, |R v| = {np.linalg.norm(rotated):.4f}")
projector = TRANSFORMATIONS["project onto x-axis"]
print(f"  P [2, 3] = {apply_column_picture(projector, [2, 3])}, P [2, -7] = {apply_column_picture(projector, [2, -7])}: different inputs, same output")

print()
print("Use It: the @ operator and a transformation you already used")
print(f"  shear @ v = {shear @ vector}, matches: {np.allclose(shear @ vector, column_result)}")
many_points = np.random.default_rng(0).normal(size=(1000, 2))
print(f"  transform 1000 points at once with points @ M.T: matches loop: "
      f"{np.allclose(many_points @ shear.T, apply_to_points(shear, many_points))}")
features = np.random.default_rng(1).normal([50.0, 3.0], [20.0, 1.0], size=(200, 2))
scaler = StandardScaler().fit(features)
scaling_matrix = np.diag(1.0 / scaler.scale_)
print(f"  StandardScaler is a shift then the diagonal matrix\n{np.round(scaling_matrix, 4)}")
print(f"  (X - mean) @ D matches scaler.transform: {np.allclose((features - scaler.mean_) @ scaling_matrix, scaler.transform(features))}")

grid_lines = []
for value in np.linspace(-2, 2, 9):
    grid_lines.append(np.column_stack([np.full(50, value), np.linspace(-2, 2, 50)]))
    grid_lines.append(np.column_stack([np.linspace(-2, 2, 50), np.full(50, value)]))
letter_f = np.array([[0, 0], [0, 1.5], [1, 1.5], [1, 1.25], [0.25, 1.25], [0.25, 0.85], [0.75, 0.85],
                     [0.75, 0.6], [0.25, 0.6], [0.25, 0], [0, 0]])

figure, axes_grid = plt.subplots(2, 3, figsize=(12, 8))
for axes, (name, matrix) in zip(axes_grid.flat, TRANSFORMATIONS.items()):
    for line in grid_lines:
        moved = line @ matrix.T
        axes.plot(moved[:, 0], moved[:, 1], color="lightgray", linewidth=0.8)
    moved_f = letter_f @ matrix.T
    axes.fill(moved_f[:, 0], moved_f[:, 1], color="tab:purple", alpha=0.6)
    axes.quiver(0, 0, *matrix[:, 0], color="tab:green", angles="xy", scale_units="xy", scale=1, width=0.015)
    axes.quiver(0, 0, *matrix[:, 1], color="tab:red", angles="xy", scale_units="xy", scale=1, width=0.015)
    axes.set(xlim=(-3, 3), ylim=(-3, 3), title=name)
    axes.set_aspect("equal")
figure.suptitle("Each matrix moves the whole grid; green = where i_hat lands, red = where j_hat lands")
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
print(f"  saved {OUTPUT_PATH.name}")
