from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = Path(__file__).parent / "out.png"


def add(first, second):
    return np.array([a + b for a, b in zip(first, second)], dtype=float)


def scale(scalar, vector):
    return np.array([scalar * component for component in vector], dtype=float)


def linear_combination(coefficients, vectors):
    result = np.zeros(len(vectors[0]))
    for coefficient, vector in zip(coefficients, vectors):
        result = add(result, scale(coefficient, vector))
    return result


def signed_area_2d(first, second):
    return first[0] * second[1] - first[1] * second[0]


def are_collinear_2d(first, second, tolerance=1e-12):
    return abs(signed_area_2d(first, second)) < tolerance


def coordinates_in_basis_2d(basis_one, basis_two, target):
    area = signed_area_2d(basis_one, basis_two)
    if abs(area) < 1e-12:
        raise ValueError("basis vectors are collinear: they only span a line")
    first_coordinate = signed_area_2d(target, basis_two) / area
    second_coordinate = signed_area_2d(basis_one, target) / area
    return np.array([first_coordinate, second_coordinate])


print("Step 1: vectors as lists of numbers")
v = np.array([3.0, 1.0])
w = np.array([-1.0, 2.0])
print(f"  v = {v}, w = {w}")
print(f"  v + w     = {add(v, w)}")
print(f"  2.5 * v   = {scale(2.5, v)}")
print(f"  -1 * w    = {scale(-1, w)}")
print(f"  v - w     = {add(v, scale(-1, w))}")

print()
print("Step 2: linear combinations")
i_hat = np.array([1.0, 0.0])
j_hat = np.array([0.0, 1.0])
print(f"  3 * i_hat + 1 * j_hat = {linear_combination([3, 1], [i_hat, j_hat])}  (this IS v)")
combination = linear_combination([2, -1], [v, w])
print(f"  2 * v - 1 * w = {combination}")
for a, b in [(1, 1), (0.5, 0.5), (-1, 2)]:
    print(f"  {a:>4} * v + {b:>4} * w = {linear_combination([a, b], [v, w])}")

print()
print("Step 3: span and collinearity")
u = np.array([-6.0, -2.0])
print(f"  signed area of (v, w) = {signed_area_2d(v, w)} -> collinear? {are_collinear_2d(v, w)}")
print(f"  signed area of (v, u) = {signed_area_2d(v, u)} -> collinear? {are_collinear_2d(v, u)}")
print("  v and w span the whole plane; v and u = -2v only span a line")

print()
print("Step 4: coordinates in a different basis")
target = np.array([4.0, 7.0])
coordinates = coordinates_in_basis_2d(v, w, target)
print(f"  target {target} = {coordinates[0]:.3f} * v + {coordinates[1]:.3f} * w")
print(f"  check: {linear_combination(coordinates, [v, w])}")
try:
    coordinates_in_basis_2d(v, u, target)
except ValueError as error:
    print(f"  with basis (v, u): {error}")

print()
print("Step 5: the same ideas in 3D")
a = np.array([1.0, 0.0, 2.0])
b = np.array([0.0, 1.0, 1.0])
c = linear_combination([2, 3], [a, b])
print(f"  c = 2a + 3b = {c}: c lies in the plane spanned by a and b")
d = np.array([0.0, 0.0, 1.0])
print(f"  rank of [a, b, c] = {np.linalg.matrix_rank(np.column_stack([a, b, c]))} (c adds no new direction)")
print(f"  rank of [a, b, d] = {np.linalg.matrix_rank(np.column_stack([a, b, d]))} (d escapes the plane)")

print()
print("Use It: NumPy does all of this with operators")
print(f"  v + w matches: {np.allclose(v + w, add(v, w))}")
print(f"  2.5 * v matches: {np.allclose(2.5 * v, scale(2.5, v))}")
print(f"  2 * v - w matches: {np.allclose(2 * v - w, combination)}")
basis_matrix = np.column_stack([v, w])
numpy_coordinates = np.linalg.solve(basis_matrix, target)
print(f"  np.linalg.solve coordinates {numpy_coordinates} match: {np.allclose(numpy_coordinates, coordinates)}")
print(f"  basis_matrix @ coordinates = {basis_matrix @ coordinates}")

figure, (sum_axes, grid_axes) = plt.subplots(1, 2, figsize=(11, 5))
arrow_style = {"angles": "xy", "scale_units": "xy", "scale": 1, "width": 0.008}
sum_axes.quiver(0, 0, *v, color="tab:blue", **arrow_style)
sum_axes.quiver(*v, *w, color="tab:orange", **arrow_style)
sum_axes.quiver(0, 0, *(v + w), color="tab:green", **arrow_style)
sum_axes.text(*(v / 2 + [0, -0.4]), "v")
sum_axes.text(*(v + w / 2 + [0.15, 0]), "w")
sum_axes.text(*((v + w) / 2 + [-0.6, 0.2]), "v + w")
sum_axes.set(xlim=(-1, 4), ylim=(-1, 4), title="Addition: tip to tail")

for a_coefficient in np.arange(-2, 3):
    for b_coefficient in np.arange(-2, 3):
        point = a_coefficient * v + b_coefficient * w
        grid_axes.plot(*point, "o", color="gray", markersize=3)
for value in np.arange(-2, 3):
    line_along_w = np.array([value * v + t * w for t in (-2, 2)])
    line_along_v = np.array([t * v + value * w for t in (-2, 2)])
    grid_axes.plot(line_along_w[:, 0], line_along_w[:, 1], color="lightgray", linewidth=0.8)
    grid_axes.plot(line_along_v[:, 0], line_along_v[:, 1], color="lightgray", linewidth=0.8)
grid_axes.quiver(0, 0, *v, color="tab:blue", **arrow_style)
grid_axes.quiver(0, 0, *w, color="tab:orange", **arrow_style)
grid_axes.quiver(0, 0, *target, color="tab:red", **arrow_style)
grid_axes.text(*(target + [0.3, -0.2]), f"{coordinates[0]:.2f} v + {coordinates[1]:.2f} w")
grid_axes.set(xlim=(-9, 10), ylim=(-7, 8), title="Span of v and w: every point is a v + b w")
for axes in (sum_axes, grid_axes):
    axes.axhline(0, color="black", linewidth=0.5)
    axes.axvline(0, color="black", linewidth=0.5)
    axes.set_aspect("equal")
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
print(f"  saved {OUTPUT_PATH.name}")
