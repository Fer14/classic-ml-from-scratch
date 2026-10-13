# Linear Systems and the Inverse

> Solving `A x = b` means asking which vector the transformation `A` sends onto `b`. The inverse is the transformation that plays the movie backwards, and it only exists if `A` did not squash space.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 Lesson 03 (Matrices as Transformations), Phase 01 Lesson 04 (Matrix Multiplication)
**Time:** ~75 minutes

## Learning Objectives

- Interpret `A x = b` with the row picture (intersecting lines or planes) and the column picture (combining columns to reach `b`)
- Implement Gaussian elimination with partial pivoting and back substitution, and Gauss-Jordan inversion
- Explain singular matrices geometrically and distinguish "no solution" from "infinitely many solutions" using rank
- Recognize ill-conditioned systems, and know why `np.linalg.solve` is preferred over `np.linalg.inv(A) @ b`

## The Problem

In Phase 05 you will fit linear regression in closed form. The answer comes from solving a system of linear equations, `(X^T X) w = X^T y`, where `w` holds the model's weights. Whether that system has a unique answer, no answer, or infinitely many decides whether your model is well defined. Whether it is *nearly* singular decides whether your weights are meaningful or numerical garbage.

Linear systems also show up in Gaussian processes, in Newton's method, in Kalman filters, and inside every library call that says "solve". You will write the solver yourself, because seeing elimination happen is the best way to understand when it breaks.

## The Concept

### Two pictures of `A x = b`

Take

```
2x + 1y = 5
1x + 3y = 10          A = [[2, 1], [1, 3]],  b = [5, 10]
```

**Row picture.** Each equation is a line in the plane. The solution is where the lines cross: `(1, 3)`. In 3-D each equation is a plane, and the solution is where three planes meet.

**Column picture (3Blue1Brown's view).** `A` is a transformation. You want the input vector `x` that `A` moves onto `b`:

```
x * [2, 1] + y * [1, 3] = [5, 10]
     col 1        col 2
```

Which combination of the columns reaches `b`? `1 * col1 + 3 * col2`. Open `code/out.png` to see both pictures side by side.

### When there is no unique answer

If `A` squashes space onto a lower dimension (its columns are dependent), it is *singular*:

```
A = [[1, 2], [2, 4]]        both columns lie on the line through [1, 2]
```

Every output of `A` lies on that line (the *column space*). Then:

- If `b` is on the line (e.g. `[3, 6]`), infinitely many `x` reach it. A whole line of inputs (the *null space*, shifted) gets squashed onto the same point.
- If `b` is off the line (e.g. `[3, 5]`), nothing reaches it. No solution.

In the row picture: the two lines are the same line (infinitely many intersections) or parallel (none).

The *rank* is the dimension of the column space: the number of independent columns. An `n x n` matrix is invertible exactly when its rank is `n`.

### Gaussian elimination

Elimination turns `A x = b` into an upper-triangular system with the same solution, using operations that do not change the solution: swap two equations, or subtract a multiple of one equation from another.

```
[ 2  1 -1 |   8 ]        [ -3  -1     2     | -11    ]
[-3 -1  2 | -11 ]  --->  [  0   1.667 0.667 |  4.333 ]
[-2  1  2 |  -3 ]        [  0   0     0.2   | -0.2   ]
```

Then *back substitution* solves from the bottom up: `0.2 z = -0.2` gives `z = -1`, then the second row gives `y`, then the first gives `x`.

**Partial pivoting**: before eliminating a column, swap up the row with the largest absolute value in that column. Dividing by a tiny pivot amplifies rounding errors; dividing by the largest available one keeps them small. A pivot that is (numerically) zero means that column adds no new direction: the matrix is singular.

### The inverse

If `A` does not squash space, you can undo it. The inverse `A^-1` is the transformation that sends every output back to its input:

```
A^-1 A = A A^-1 = I        and then        x = A^-1 b
```

To compute it, solve `A X = I` for all columns at once (Gauss-Jordan): augment `[A | I]`, reduce the left half to `I`, and the right half becomes `A^-1`.

### Solve, do not invert

`x = A^-1 b` is the right formula and the wrong algorithm. Computing the inverse costs about three times as much as solving directly and loses more precision. `np.linalg.solve(A, b)` factors `A` (into triangular pieces, called LU) and back-substitutes. If you need to solve with the same `A` for many right-hand sides, factor once and reuse the factors (`scipy.linalg.lu_factor` / `lu_solve`).

### Conditioning

Two lines that are almost parallel cross at a point that moves enormously when one line shifts slightly:

```
x + y       = 2              b = [2, 2.0001] -> x = [1, 1]
x + 1.0001y = 2.0001         b = [2, 2.0002] -> x = [0, 2]
```

A change of 0.0001 in `b` moved the answer by 1. The *condition number* (about 40,000 here) measures this amplification. In regression, nearly duplicate features (multicollinearity) produce exactly this: weights that swing wildly with tiny changes in the data. Ridge regression (Phase 05) fixes it.

## Build It

### Step 1: Row reduction with partial pivoting

```python
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
```

Columns without a usable pivot are skipped, so the number of pivots found is the rank.

### Step 2: Back substitution and the solver

```python
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
```

The diagnosis reads the pivots. If every column of `A` has a pivot, the solution is unique. If not, look at the extra column `b`: a pivot there means a row reduced to `0 = nonzero`, a contradiction.

```
  back substitution: [ 2.  3. -1.]
  b = [3. 6.]: infinitely many solutions: the matrix is singular
  b = [3. 5.]: no solution: the equations contradict each other
```

### Step 3: Gauss-Jordan inverse

```python
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
```

```
  A^-1 =
[[ 0.6 -0.2]
 [-0.2  0.4]]
  A^-1 b = [1. 3.] (same as solving)
```

### Step 4: Fitting a line is a linear system

```python
design = np.column_stack([sizes, np.ones_like(sizes)])
coefficients = solve(design.T @ design, design.T @ prices)
```

The column of ones lets the system learn an intercept (the bias that a purely linear map cannot express, Lesson 03). You will derive these *normal equations* in Phase 05; for now notice that training a linear regression is literally one call to `solve`.

## Use It

```python
np.linalg.solve(matrix, target)
np.linalg.inv(matrix)
np.linalg.matrix_rank(singular)
np.linalg.cond(nearly_parallel)

lu_factors = scipy.linalg.lu_factor(matrix_3)
scipy.linalg.lu_solve(lu_factors, target_3)

np.linalg.lstsq(design, prices, rcond=None)
```

All match the from-scratch results. Three notes:

- `np.linalg.solve` raises `LinAlgError: Singular matrix` where your solver gives a more specific diagnosis. Libraries check only for numerical failure; deciding "no solution vs. infinitely many" is your job.
- `np.linalg.matrix_rank` uses the SVD (Phase 09) instead of elimination, because counting pivots with a tolerance is fragile for nearly singular matrices.
- `np.linalg.lstsq` solves the line fit without ever forming `X^T X`, whose condition number is the *square* of `X`'s. For regression on real data, prefer it (or scikit-learn's `LinearRegression`, which does the same) over the normal equations.

## Exercises

1. **Easy.** Solve `x + 2y + z = 2`, `3x + 8y + z = 12`, `4y + z = 2` with your `solve`, and inspect the upper-triangular matrix from `row_reduce`. Verify with `np.linalg.solve`.
2. **Medium.** Show that `inverse(rotation(theta))` equals `rotation(-theta)` and also equals the transpose of `rotation(theta)`. Explain both facts geometrically.
3. **Hard.** Measure numerical error: for Hilbert matrices `H[i, j] = 1 / (i + j + 1)` of size 4 to 12, set `x_true = np.ones(n)`, `b = H @ x_true`, and compare `max abs error` of `solve(H, b)` and of `inverse(H) @ b` against `np.linalg.cond(H)`. Plot error versus condition number on log-log axes. What rule of thumb relates them?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Linear system | "Simultaneous equations" | A x = b: find the input x that the transformation A sends to b |
| Gaussian elimination | "Row reduction" | Using row swaps and row subtractions to turn A into upper-triangular form without changing the solution |
| Pivot | "Leading entry" | The entry used to eliminate the rest of its column; a missing pivot means a dependent column |
| Partial pivoting | "Pick the biggest" | Swapping in the row with the largest absolute value in the current column to reduce rounding error |
| Inverse | "Undo" | The matrix A^-1 with A^-1 A = I; exists only when A does not collapse any dimension |
| Singular matrix | "Not invertible" | A square matrix whose columns are dependent; rank below n, determinant zero |
| Column space / null space | "Reachable outputs / squashed inputs" | All vectors A x can reach / all x with A x = 0 |
| Condition number | "How sensitive" | The factor by which relative errors in b can be amplified in x; large means ill-conditioned |

## Further Reading

- [3Blue1Brown, Inverse matrices, column space and null space (Essence of Linear Algebra, chapter 7)](https://www.3blue1brown.com/lessons/inverse-matrices) -- the "play the transformation backwards" intuition used here
- [Gilbert Strang, MIT 18.06 Lectures 2-4](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- elimination, inverses, and LU factorization
- [numpy.linalg.solve documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) -- the LAPACK routine behind the library call
- *Numerical Linear Algebra* by Trefethen and Bau -- the standard reference on pivoting, stability, and conditioning
