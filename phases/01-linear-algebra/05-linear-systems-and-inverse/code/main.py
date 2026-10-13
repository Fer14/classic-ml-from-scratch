from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

OUTPUT_PATH = Path(__file__).parent / "out.png"
PIVOT_TOLERANCE = 1e-12


def row_reduce(augmented):
    matrix = augmented.astype(float).copy()
    n_rows, n_columns = matrix.shape
    pivot_columns = []
    pivot_row = 0
    for column in range(n_columns):
        if pivot_row == n_rows:
            break
        best_row = pivot_row + int(np.argmax(np.abs(matrix[pivot_row:, column])))
        if abs(matrix[best_row, column]) < PIVOT_TOLERANCE:
            continue
        matrix[[pivot_row, best_row]] = matrix[[best_row, pivot_row]]
        for row in range(pivot_row + 1, n_rows):
            factor = matrix[row, column] / matrix[pivot_row, column]
            matrix[row] = matrix[row] - factor * matrix[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
    return matrix, pivot_columns


def rank(matrix):
    _, pivot_columns = row_reduce(matrix)
    return len(pivot_columns)


def back_substitute(upper, right_hand_side):
    n = len(right_hand_side)
    solution = np.zeros(n)
    for row in range(n - 1, -1, -1):
        already_known = upper[row, row + 1:] @ solution[row + 1:]
        solution[row] = (right_hand_side[row] - already_known) / upper[row, row]
    return solution


def solve(matrix, right_hand_side):
    n = matrix.shape[0]
    augmented = np.column_stack([matrix, right_hand_side])
    reduced, pivot_columns = row_reduce(augmented)
    if pivot_columns[:n] != list(range(n)):
        if n in pivot_columns:
            raise np.linalg.LinAlgError("no solution: the equations contradict each other")
        raise np.linalg.LinAlgError("infinitely many solutions: the matrix is singular")
    return back_substitute(reduced[:, :n], reduced[:, n])


def inverse(matrix):
    n = matrix.shape[0]
    augmented = np.column_stack([matrix.astype(float), np.eye(n)])
    for column in range(n):
        best_row = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[best_row, column]) < PIVOT_TOLERANCE:
            raise np.linalg.LinAlgError("matrix is singular: it squashes space, nothing can undo it")
        augmented[[column, best_row]] = augmented[[best_row, column]]
        augmented[column] = augmented[column] / augmented[column, column]
        for row in range(n):
            if row != column:
                augmented[row] = augmented[row] - augmented[row, column] * augmented[column]
    return augmented[:, n:]


print("Step 1: a 2x2 system, two pictures")
matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
target = np.array([5.0, 10.0])
solution = solve(matrix, target)
print("  2x + 1y = 5")
print("  1x + 3y = 10")
print(f"  solution x = {solution[0]:.4f}, y = {solution[1]:.4f}")
print(f"  row picture:    the two lines cross at {solution}")
print(f"  column picture: {solution[0]:.1f} * {matrix[:, 0]} + {solution[1]:.1f} * {matrix[:, 1]} = {matrix @ solution}")

print()
print("Step 2: Gaussian elimination on a 3x3 system")
matrix_3 = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
target_3 = np.array([8.0, -11.0, -3.0])
reduced, pivots = row_reduce(np.column_stack([matrix_3, target_3]))
print(f"  upper-triangular form after elimination:\n{np.round(reduced, 4) + 0.0}")
solution_3 = solve(matrix_3, target_3)
print(f"  back substitution: {np.round(solution_3, 6) + 0.0}")
print(f"  check A x = {np.round(matrix_3 @ solution_3, 6) + 0.0}")

print()
print("Step 3: the inverse undoes the transformation")
matrix_inverse = inverse(matrix)
print(f"  A^-1 =\n{np.round(matrix_inverse, 4)}")
print(f"  A^-1 A =\n{np.round(matrix_inverse @ matrix, 10) + 0.0}")
print(f"  A^-1 b = {matrix_inverse @ target} (same as solving)")

print()
print("Step 4: when there is no inverse")
singular = np.array([[1.0, 2.0], [2.0, 4.0]])
print(f"  rank of {singular.tolist()} = {rank(singular)}: both columns lie on one line")
for right_side in (np.array([3.0, 6.0]), np.array([3.0, 5.0])):
    try:
        solve(singular, right_side)
    except np.linalg.LinAlgError as error:
        print(f"  b = {right_side}: {error}")
try:
    inverse(singular)
except np.linalg.LinAlgError as error:
    print(f"  inverse: {error}")

print()
print("Step 5: nearly singular means unstable")
nearly_parallel = np.array([[1.0, 1.0], [1.0, 1.0001]])
for right_side in (np.array([2.0, 2.0001]), np.array([2.0, 2.0002])):
    print(f"  b = {right_side} -> x = {np.round(solve(nearly_parallel, right_side), 4)}")
print(f"  a change of 0.0001 in b moved x by 1.0; condition number = {np.linalg.cond(nearly_parallel):,.0f}")

print()
print("Step 6: fitting a line is a linear system")
generator = np.random.default_rng(0)
sizes = generator.uniform(40, 150, size=30)
prices = 3.0 * sizes + 50 + generator.normal(0, 15, size=30)
design = np.column_stack([sizes, np.ones_like(sizes)])
coefficients = solve(design.T @ design, design.T @ prices)
print(f"  normal equations (X^T X) w = X^T y -> slope {coefficients[0]:.3f}, intercept {coefficients[1]:.3f}")

print()
print("Use It: NumPy and SciPy")
print(f"  np.linalg.solve matches (2x2): {np.allclose(np.linalg.solve(matrix, target), solution)}")
print(f"  np.linalg.solve matches (3x3): {np.allclose(np.linalg.solve(matrix_3, target_3), solution_3)}")
print(f"  np.linalg.inv matches: {np.allclose(np.linalg.inv(matrix), matrix_inverse)}")
print(f"  np.linalg.matrix_rank matches: {np.linalg.matrix_rank(singular) == rank(singular)}")
lu_factors = scipy.linalg.lu_factor(matrix_3)
print(f"  scipy LU factor once, solve many: {np.allclose(scipy.linalg.lu_solve(lu_factors, target_3), solution_3)}")
lstsq_coefficients = np.linalg.lstsq(design, prices, rcond=None)[0]
print(f"  np.linalg.lstsq line fit matches: {np.allclose(lstsq_coefficients, coefficients)}")
try:
    np.linalg.solve(singular, np.array([3.0, 6.0]))
except np.linalg.LinAlgError as error:
    print(f"  np.linalg.solve on the singular matrix raises: {error}")

figure, (row_axes, column_axes) = plt.subplots(1, 2, figsize=(11, 5))
x_values = np.linspace(-1, 4, 50)
row_axes.plot(x_values, 5 - 2 * x_values, label="2x + y = 5")
row_axes.plot(x_values, (10 - x_values) / 3, label="x + 3y = 10")
row_axes.plot(*solution, "ko")
row_axes.annotate(f"({solution[0]:.0f}, {solution[1]:.0f})", solution + [0.15, 0.15])
row_axes.set(xlim=(-1, 4), ylim=(-1, 5), title="Row picture: lines intersect at the solution")
row_axes.legend()
arrow_style = {"angles": "xy", "scale_units": "xy", "scale": 1, "width": 0.01}
first_part = solution[0] * matrix[:, 0]
column_axes.quiver(0, 0, *matrix[:, 0], color="tab:green", **arrow_style)
column_axes.quiver(0, 0, *matrix[:, 1], color="tab:red", **arrow_style)
column_axes.quiver(0, 0, *first_part, color="tab:green", alpha=0.35, **arrow_style)
column_axes.quiver(*first_part, *(solution[1] * matrix[:, 1]), color="tab:red", alpha=0.35, **arrow_style)
column_axes.plot(*target, "ko")
column_axes.annotate("b = (5, 10)", target + [0.2, 0])
column_axes.text(0.9, 0.2, "column 1")
column_axes.text(-0.4, 3.2, "column 2")
column_axes.set(xlim=(-1, 8), ylim=(-1, 11), title="Column picture: 1 * col1 + 3 * col2 = b")
for axes in (row_axes, column_axes):
    axes.axhline(0, color="black", linewidth=0.5)
    axes.axvline(0, color="black", linewidth=0.5)
    axes.set_aspect("equal")
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
print(f"  saved {OUTPUT_PATH.name}")
