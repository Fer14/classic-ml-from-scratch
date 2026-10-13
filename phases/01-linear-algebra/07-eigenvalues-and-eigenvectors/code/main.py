import cmath
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = Path(__file__).parent / "out.png"


def eigenvalues_2x2(matrix):
    trace = matrix[0, 0] + matrix[1, 1]
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    discriminant = trace * trace - 4 * determinant
    root = cmath.sqrt(discriminant)
    values = [(trace + root) / 2, (trace - root) / 2]
    if discriminant >= 0:
        return [value.real for value in values]
    return values


def format_numbers(numbers):
    parts = []
    for number in numbers:
        number = complex(number)
        real_part = number.real if abs(number.real) > 1e-12 else 0.0
        if abs(number.imag) < 1e-12:
            parts.append(f"{real_part:.4g}")
        else:
            parts.append(f"{real_part:.4g}{number.imag:+.4g}i")
    return "[" + ", ".join(parts) + "]"


def eigenvector_2x2(matrix, eigenvalue):
    shifted = matrix - eigenvalue * np.eye(2)
    row = shifted[0] if np.abs(shifted[0]).sum() > np.abs(shifted[1]).sum() else shifted[1]
    if np.abs(row).sum() < 1e-12:
        return np.array([1.0, 0.0])
    vector = np.array([-row[1], row[0]])
    return vector / np.linalg.norm(vector)


def stays_on_span_scan(matrix, n_angles=3600):
    found = []
    angles = np.linspace(0, math.pi, n_angles, endpoint=False)
    cross_values = []
    for angle in angles:
        direction = np.array([math.cos(angle), math.sin(angle)])
        image = matrix @ direction
        cross_values.append(direction[0] * image[1] - direction[1] * image[0])
    for index in range(n_angles - 1):
        if cross_values[index] == 0 or cross_values[index] * cross_values[index + 1] < 0:
            found.append(math.degrees(angles[index]))
    return found


def power_iteration(matrix, n_steps=200, seed=0):
    vector = np.random.default_rng(seed).normal(size=matrix.shape[0])
    vector = vector / np.linalg.norm(vector)
    for _ in range(n_steps):
        image = matrix @ vector
        vector = image / np.linalg.norm(image)
    eigenvalue = vector @ matrix @ vector
    return eigenvalue, vector


def symmetric_eigendecomposition(matrix, n_steps=500):
    remaining = matrix.astype(float).copy()
    values, vectors = [], []
    for index in range(matrix.shape[0]):
        eigenvalue, eigenvector = power_iteration(remaining, n_steps=n_steps, seed=index)
        values.append(eigenvalue)
        vectors.append(eigenvector)
        remaining = remaining - eigenvalue * np.outer(eigenvector, eigenvector)
    return np.array(values), np.column_stack(vectors)


def rotation(angle_degrees):
    angle = math.radians(angle_degrees)
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


matrix = np.array([[3.0, 1.0], [0.0, 2.0]])

print("Step 1: find directions that stay on their own span")
print(f"  A = {matrix.tolist()}")
print(f"  scanning 3600 directions, A v is parallel to v at angles: {[round(a, 2) for a in stays_on_span_scan(matrix)]} deg")
for test in (np.array([1.0, 0.0]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])):
    print(f"  A {test} = {matrix @ test}")

print()
print("Step 2: solve det(A - lambda I) = 0")
values = eigenvalues_2x2(matrix)
print(f"  lambda^2 - (trace) lambda + det = lambda^2 - {np.trace(matrix)} lambda + {np.linalg.det(matrix):.0f} = 0")
print(f"  eigenvalues: {format_numbers(values)}")
for value in values:
    vector = eigenvector_2x2(matrix, value)
    print(f"  lambda = {value}: v = {np.round(vector, 4)}, A v = {np.round(matrix @ vector, 4)} = {value} * v")

print()
print("Step 3: a rotation has no real eigenvectors")
print(f"  rotation(90) eigenvalues: {format_numbers(eigenvalues_2x2(rotation(90)))}")
print(f"  directions kept on their span: {stays_on_span_scan(rotation(90))}")
shear = np.array([[1.0, 1.0], [0.0, 1.0]])
print(f"  shear eigenvalues: {format_numbers(eigenvalues_2x2(shear))}, only one eigen-direction: {np.abs(eigenvector_2x2(shear, 1.0))}")

print()
print("Step 4: in the eigenbasis, the matrix is just scaling")
eigenvector_matrix = np.column_stack([eigenvector_2x2(matrix, value) for value in values])
diagonal = np.linalg.inv(eigenvector_matrix) @ matrix @ eigenvector_matrix
print(f"  V^-1 A V =\n{np.round(diagonal, 10) + 0.0}")
power = 10
fast_power = eigenvector_matrix @ np.diag(np.array(values) ** power) @ np.linalg.inv(eigenvector_matrix)
print(f"  A^10 via V D^10 V^-1 matches repeated multiplication: {np.allclose(fast_power, np.linalg.matrix_power(matrix, power))}")

print()
print("Step 5: power iteration finds the dominant direction")
dominant_value, dominant_vector = power_iteration(matrix)
print(f"  dominant eigenvalue {dominant_value:.6f}, eigenvector {np.round(dominant_vector, 6) + 0.0} (sign is arbitrary)")

print()
print("Step 6: eigenvectors of a covariance matrix are the data's main axes")
generator = np.random.default_rng(42)
stretched = generator.normal(size=(500, 2)) * [3.0, 0.7]
data = stretched @ rotation(30).T
covariance = np.cov(data, rowvar=False)
covariance_values, covariance_vectors = symmetric_eigendecomposition(covariance)
main_axis_angle = math.degrees(math.atan2(covariance_vectors[1, 0], covariance_vectors[0, 0])) % 180
print(f"  covariance =\n{np.round(covariance, 3)}")
print(f"  eigenvalues (variances along the axes): {np.round(covariance_values, 3)}")
print(f"  main axis angle: {main_axis_angle:.1f} deg (data was rotated by 30 deg)")
print(f"  eigenvectors are perpendicular: dot = {covariance_vectors[:, 0] @ covariance_vectors[:, 1]:.2e}")

print()
print("Step 7: a Markov chain's long-run distribution is an eigenvector with eigenvalue 1")
transition = np.array([[0.9, 0.5, 0.2], [0.075, 0.4, 0.3], [0.025, 0.1, 0.5]])
_, stationary = power_iteration(transition, n_steps=500)
stationary = stationary / stationary.sum()
print(f"  stationary distribution (sunny, cloudy, rainy): {np.round(stationary, 4)}")
print(f"  P @ pi = {np.round(transition @ stationary, 4)}")

print()
print("Use It: np.linalg.eig and np.linalg.eigh")
numpy_values, numpy_vectors = np.linalg.eig(matrix)
print(f"  eig values {format_numbers(numpy_values)} match: {np.allclose(sorted(numpy_values), sorted(values))}")
symmetric_values, symmetric_vectors = np.linalg.eigh(covariance)
print(f"  eigh (ascending) {np.round(symmetric_values, 4)} match ours: {np.allclose(symmetric_values[::-1], covariance_values)}")
same_up_to_sign = np.allclose(np.abs(symmetric_vectors[:, ::-1]), np.abs(covariance_vectors), atol=1e-6)
print(f"  eigh vectors match ours up to sign: {same_up_to_sign}")
print(f"  rotation(90) via eig: {format_numbers(np.linalg.eig(rotation(90))[0])}")

figure, (map_axes, data_axes) = plt.subplots(1, 2, figsize=(12, 5.5))
angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)
for angle in angles:
    direction = np.array([math.cos(angle), math.sin(angle)])
    image = matrix @ direction
    is_eigen = abs(direction[0] * image[1] - direction[1] * image[0]) < 1e-9
    color = "tab:red" if is_eigen else "lightgray"
    map_axes.annotate("", xy=image, xytext=direction, arrowprops={"arrowstyle": "->", "color": color})
circle = np.array([[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, 200)])
ellipse = circle @ matrix.T
map_axes.plot(circle[:, 0], circle[:, 1], color="gray", linewidth=0.8)
map_axes.plot(ellipse[:, 0], ellipse[:, 1], color="tab:blue")
for value in values:
    vector = eigenvector_2x2(matrix, value)
    line = np.array([-4 * vector, 4 * vector])
    map_axes.plot(line[:, 0], line[:, 1], color="tab:red", alpha=0.4, linestyle="--")
map_axes.set(xlim=(-4, 4), ylim=(-3, 3), title="Unit circle -> A (blue). Red: eigen-directions stay on their line")
data_axes.scatter(data[:, 0], data[:, 1], s=6, alpha=0.4)
for value, vector in zip(covariance_values, covariance_vectors.T):
    data_axes.annotate("", xy=2 * math.sqrt(value) * vector, xytext=(0, 0),
                       arrowprops={"arrowstyle": "->", "color": "tab:red", "linewidth": 2})
data_axes.set(xlim=(-10, 10), ylim=(-7, 7), title="Covariance eigenvectors = principal axes of the data")
for axes in (map_axes, data_axes):
    axes.set_aspect("equal")
    axes.axhline(0, color="black", linewidth=0.5)
    axes.axvline(0, color="black", linewidth=0.5)
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
print(f"  saved {OUTPUT_PATH.name}")
