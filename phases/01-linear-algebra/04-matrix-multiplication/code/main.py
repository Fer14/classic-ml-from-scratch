import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = Path(__file__).parent / "out.png"


def matmul_dot_products(left, right):
    rows, inner = left.shape
    inner_right, columns = right.shape
    if inner != inner_right:
        raise ValueError(f"shapes {left.shape} and {right.shape} do not align: {inner} != {inner_right}")
    result = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            total = 0.0
            for k in range(inner):
                total += left[i, k] * right[k, j]
            result[i, j] = total
    return result


def matmul_columns(left, right):
    return np.column_stack([sum(right[k, j] * left[:, k] for k in range(left.shape[1])) for j in range(right.shape[1])])


def matmul_outer_products(left, right):
    return sum(np.outer(left[:, k], right[k, :]) for k in range(left.shape[1]))


def rotation(angle_degrees):
    angle = math.radians(angle_degrees)
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])
ROTATE_90 = rotation(90)

print("Step 1: the product as a grid of dot products")
left = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
right = np.array([[3.0, 1.0], [0.0, 2.0], [1.0, 1.0]])
product = matmul_dot_products(left, right)
print(f"  ({left.shape[0]}x{left.shape[1]}) @ ({right.shape[0]}x{right.shape[1]}) -> {product.shape}")
print(f"  entry [0, 1] = row 0 of left . column 1 of right = {left[0]} . {right[:, 1]} = {product[0, 1]}")
print(f"  result:\n{product}")
try:
    matmul_dot_products(left, left)
except ValueError as error:
    print(f"  left @ left fails: {error}")

print()
print("Step 2: other ways to see the same product")
print(f"  column view  (each result column = left @ that column of right): {np.allclose(matmul_columns(left, right), product)}")
print(f"  outer products (sum of column_k(left) x row_k(right)):            {np.allclose(matmul_outer_products(left, right), product)}")

print()
print("Step 3: multiplication is composition, read right to left")
vector = np.array([1.0, 2.0])
step_by_step = SHEAR @ (ROTATE_90 @ vector)
composed = matmul_dot_products(SHEAR, ROTATE_90)
print(f"  rotate then shear, one step at a time: {np.round(step_by_step, 6) + 0.0}")
print(f"  (SHEAR @ ROTATE_90) @ v:               {np.round(composed @ vector, 6) + 0.0}")
print(f"  SHEAR @ ROTATE_90 columns are where i_hat, j_hat end up:\n{np.round(composed, 6) + 0.0}")

print()
print("Step 4: order matters, grouping does not")
other_order = matmul_dot_products(ROTATE_90, SHEAR)
print(f"  ROTATE_90 @ SHEAR:\n{np.round(other_order, 6) + 0.0}")
print(f"  commutative? {np.allclose(composed, other_order)}")
generator = np.random.default_rng(0)
a, b, c = generator.normal(size=(3, 4)), generator.normal(size=(4, 5)), generator.normal(size=(5, 2))
print(f"  associative (AB)C == A(BC)? {np.allclose(matmul_dot_products(matmul_dot_products(a, b), c), matmul_dot_products(a, matmul_dot_products(b, c)))}")
print(f"  transpose rule (AB)^T == B^T A^T? {np.allclose(matmul_dot_products(a, b).T, matmul_dot_products(b.T, a.T))}")
print(f"  identity: A I == A? {np.allclose(matmul_dot_products(a, np.eye(4)), a)}")

print()
print("Step 5: in ML, one matmul predicts a whole batch")
samples = generator.normal(size=(6, 3))
weights = np.array([[0.5, -1.0], [2.0, 0.0], [-0.5, 1.5]])
predictions = matmul_dot_products(samples, weights)
print(f"  X (6 samples x 3 features) @ W (3 x 2 outputs) -> {predictions.shape}")
first_hidden = generator.normal(size=(3, 4))
second_hidden = generator.normal(size=(4, 2))
two_layers = matmul_dot_products(matmul_dot_products(samples, first_hidden), second_hidden)
collapsed = matmul_dot_products(samples, matmul_dot_products(first_hidden, second_hidden))
print(f"  two stacked linear layers == one layer with W1 @ W2: {np.allclose(two_layers, collapsed)}")

print()
print("Use It: @, np.matmul, np.einsum, and why you never write the loop")
print(f"  left @ right matches: {np.allclose(left @ right, product)}")
print(f"  np.einsum('ik,kj->ij') matches: {np.allclose(np.einsum('ik,kj->ij', left, right), product)}")
size = 80
big_left, big_right = generator.normal(size=(size, size)), generator.normal(size=(size, size))
start = time.perf_counter()
slow = matmul_dot_products(big_left, big_right)
loop_seconds = time.perf_counter() - start
start = time.perf_counter()
fast = big_left @ big_right
numpy_seconds = time.perf_counter() - start
print(f"  {size}x{size}: triple loop {loop_seconds:.3f} s, @ {numpy_seconds * 1000:.3f} ms, "
      f"~{loop_seconds / max(numpy_seconds, 1e-9):,.0f}x faster, same result: {np.allclose(slow, fast)}")

letter_f = np.array([[0, 0], [0, 1.5], [1, 1.5], [1, 1.25], [0.25, 1.25], [0.25, 0.85], [0.75, 0.85],
                     [0.75, 0.6], [0.25, 0.6], [0.25, 0], [0, 0]])
panels = [("original", np.eye(2)), ("rotate 90, then shear", composed), ("shear, then rotate 90", other_order)]
figure, axes_row = plt.subplots(1, 3, figsize=(12, 4.5))
for axes, (title, matrix) in zip(axes_row, panels):
    moved = letter_f @ matrix.T
    axes.fill(moved[:, 0], moved[:, 1], color="tab:purple", alpha=0.6)
    axes.quiver(0, 0, *matrix[:, 0], color="tab:green", angles="xy", scale_units="xy", scale=1, width=0.015)
    axes.quiver(0, 0, *matrix[:, 1], color="tab:red", angles="xy", scale_units="xy", scale=1, width=0.015)
    axes.axhline(0, color="black", linewidth=0.5)
    axes.axvline(0, color="black", linewidth=0.5)
    axes.set(xlim=(-3, 3), ylim=(-3, 3), title=title)
    axes.set_aspect("equal")
figure.suptitle("Matrix products are compositions, and order matters")
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
print(f"  saved {OUTPUT_PATH.name}")
