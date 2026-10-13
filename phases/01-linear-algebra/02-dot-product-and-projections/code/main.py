import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_PATH = Path(__file__).parent / "out.png"


def dot(first, second):
    total = 0.0
    for a, b in zip(first, second):
        total += a * b
    return total


def length(vector):
    return math.sqrt(dot(vector, vector))


def angle_degrees(first, second):
    cosine = dot(first, second) / (length(first) * length(second))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def cosine_similarity_pair(first, second):
    return dot(first, second) / (length(first) * length(second))


def project(vector, onto):
    return (dot(vector, onto) / dot(onto, onto)) * np.asarray(onto, dtype=float)


def gram_schmidt(vectors):
    orthonormal = []
    for vector in vectors:
        residual = np.asarray(vector, dtype=float)
        for basis_vector in orthonormal:
            residual = residual - project(residual, basis_vector)
        if length(residual) > 1e-10:
            orthonormal.append(residual / length(residual))
    return np.array(orthonormal)


print("Step 1: the dot product two ways")
v = np.array([4.0, 1.0])
w = np.array([2.0, 3.0])
algebraic = dot(v, w)
geometric = length(v) * length(w) * math.cos(math.radians(angle_degrees(v, w)))
print(f"  v = {v}, w = {w}")
print(f"  sum of products      v . w = {algebraic:.4f}")
print(f"  |v| |w| cos(theta)         = {geometric:.4f}  (theta = {angle_degrees(v, w):.2f} deg)")

print()
print("Step 2: the sign tells you the direction")
for other in ([1.0, 2.0], [-1.0, 4.0], [-2.0, -1.0]):
    other = np.array(other)
    value = dot(v, other)
    relation = "same side" if value > 0 else ("perpendicular" if value == 0 else "opposite side")
    print(f"  v . {other} = {value:6.2f}  angle {angle_degrees(v, other):6.2f} deg  -> {relation}")

print()
print("Step 3: projection = shadow of one vector on another")
shadow = project(w, v)
perpendicular_part = w - shadow
print(f"  proj of w onto v      = {np.round(shadow, 4)}")
print(f"  w - proj (the rest)   = {np.round(perpendicular_part, 4)}")
print(f"  (w - proj) . v        = {dot(perpendicular_part, v):.2e}  (perpendicular)")
print(f"  scalar projection     = (w . v) / |v| = {dot(w, v) / length(v):.4f} = signed length of the shadow")

print()
print("Step 4: a linear model is a dot product")
weights = np.array([3000.0, 15000.0, -800.0])
bias = 50000.0
house = np.array([120.0, 3.0, 15.0])
print(f"  price = w . x + b = {dot(weights, house) + bias:,.0f}")
perpendicular_step = np.array([5.0, -1.0, 0.0])
other_house = house + perpendicular_step
print(f"  w . {perpendicular_step} = {dot(weights, perpendicular_step):.0f}: a step perpendicular to w")
print(f"  house {other_house} has the same price {dot(weights, other_house) + bias:,.0f}")

print()
print("Step 5: cosine similarity between documents")
vocabulary = ["cat", "dog", "model", "data", "pet"]
documents = np.array([
    [3.0, 1.0, 0.0, 0.0, 2.0],
    [1.0, 3.0, 0.0, 0.0, 2.0],
    [0.0, 0.0, 4.0, 3.0, 0.0],
    [6.0, 2.0, 0.0, 0.0, 4.0],
])
ours_similarity = np.array([[cosine_similarity_pair(a, b) for b in documents] for a in documents])
print("  pairwise cosine similarity:")
for row in ours_similarity:
    print("   " + " ".join(f"{value:6.3f}" for value in row))
print("  doc 3 is doc 0 doubled: similarity 1.0, the length does not matter, only the direction")

print()
print("Step 6: Gram-Schmidt builds perpendicular axes")
raw_vectors = np.array([[3.0, 1.0, 0.0], [2.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
orthonormal_basis = gram_schmidt(raw_vectors)
print(f"  orthonormal basis:\n{np.round(orthonormal_basis, 4)}")
print(f"  Q Q^T (should be identity):\n{np.round(orthonormal_basis @ orthonormal_basis.T, 10) + 0.0}")

print()
print("Use It: NumPy and scikit-learn")
print(f"  np.dot(v, w) = {np.dot(v, w)}, v @ w = {v @ w}, matches: {math.isclose(np.dot(v, w), algebraic)}")
print(f"  np.linalg.norm(v) = {np.linalg.norm(v):.4f}, matches: {math.isclose(np.linalg.norm(v), length(v))}")
print(f"  sklearn cosine_similarity matches: {np.allclose(cosine_similarity(documents), ours_similarity)}")
q_matrix, _ = np.linalg.qr(raw_vectors.T)
signs = np.sign(np.sum(q_matrix.T * orthonormal_basis, axis=1))
print(f"  np.linalg.qr gives the same axes up to sign: {np.allclose(q_matrix.T * signs[:, None], orthonormal_basis)}")

figure, axes = plt.subplots(figsize=(6, 5))
arrow_style = {"angles": "xy", "scale_units": "xy", "scale": 1, "width": 0.01}
axes.quiver(0, 0, *v, color="tab:blue", **arrow_style)
axes.quiver(0, 0, *w, color="tab:orange", **arrow_style)
axes.quiver(0, 0, *shadow, color="tab:green", **arrow_style)
axes.plot([w[0], shadow[0]], [w[1], shadow[1]], linestyle="--", color="gray")
line_points = np.array([-0.5 * v, 1.3 * v])
axes.plot(line_points[:, 0], line_points[:, 1], color="tab:blue", alpha=0.2, linewidth=6)
axes.text(*(v + [0.1, -0.3]), "v")
axes.text(*(w + [0.1, 0.1]), "w")
axes.text(*(shadow / 2 + [0.1, -0.5]), "proj of w on v")
axes.set(xlim=(-1, 5.5), ylim=(-1, 4), title="Projection: drop a perpendicular from w onto the line of v")
axes.set_aspect("equal")
axes.axhline(0, color="black", linewidth=0.5)
axes.axvline(0, color="black", linewidth=0.5)
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
print(f"  saved {OUTPUT_PATH.name}")
