# Matrices as Transformations

> Stop reading a matrix as a grid of numbers. Read its columns: they tell you where the basis vectors land, and that determines where everything else goes.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 Lesson 01 (Vectors), Phase 01 Lesson 02 (Dot Product and Projections)
**Time:** ~60 minutes

## Learning Objectives

- Build the matrix of any 2-D linear transformation by writing down where `i_hat` and `j_hat` land
- Compute a matrix-vector product with both the column picture (linear combination of columns) and the row picture (dot products with rows)
- Recognize rotation, scaling, shear, reflection, and projection matrices, and predict what each does to a grid
- Test whether a function is linear, and explain why translation and element-wise squaring are not

## The Problem

A dataset with `n` samples and `d` features is an `n x d` matrix, and the first instinct is to see it as a spreadsheet. Then a model multiplies it by a weight matrix, and nothing about "spreadsheet" helps you understand what happened.

Here is the view that does help: a matrix is a *function* that moves every point in space, in a very constrained way. Standardizing features is a matrix (scaling each axis). PCA is a matrix (rotate onto new axes, then drop some). A linear layer in a neural network is a matrix. Once you can picture what a matrix does to a grid, you can predict what these operations do to your data before running any code. This is the central idea of 3Blue1Brown's *Essence of Linear Algebra*, chapter 3, and the rest of this phase builds on it.

## The Concept

### Linear transformations

A transformation `T` takes a vector in and returns a vector. It is *linear* if

```
T(v + w) = T(v) + T(w)        adding then moving = moving then adding
T(c v)   = c T(v)             scaling then moving = moving then scaling
```

Visually: **grid lines stay straight, parallel, and evenly spaced, and the origin stays fixed.** Rotations, stretches, shears, reflections, and flattenings qualify. Translations (the origin moves) and anything that curves grid lines do not.

### Where the basis vectors go determines everything

Every vector is a linear combination of the basis: `v = x * i_hat + y * j_hat`. Apply a linear `T` and use the two rules:

```
T(v) = x * T(i_hat) + y * T(j_hat)
```

So you only need to know two vectors: where `i_hat` lands and where `j_hat` lands. Write them as the columns of a matrix:

```
                T(i_hat)   T(j_hat)
                   |          |
                   v          v
       M  =  [   a          b   ]
             [   c          d   ]

       M [x]  =  x [a]  +  y [b]  =  [a x + b y]
         [y]       [c]       [d]     [c x + d y]
```

That is the whole definition of matrix-vector multiplication. It is not an arbitrary rule; it is "take `x` copies of where `i_hat` went and `y` copies of where `j_hat` went".

### Two pictures of the same product

- **Column picture**: `M v` is a linear combination of the columns of `M`, weighted by the entries of `v`. Best for geometry.
- **Row picture**: entry `i` of `M v` is the dot product of row `i` with `v`. Best for computing, and for ML: each row of a weight matrix is one "detector" measuring how much `v` points along it (Lesson 02).

Both give the same numbers; `[[1, 1], [0, 1]] @ [2, 3]` is `2*[1,0] + 3*[1,1] = [5, 3]`, or `[[1,1].[2,3], [0,1].[2,3]] = [5, 3]`.

### A catalogue of 2-D transformations

```
name                 i_hat lands at       j_hat lands at       matrix
identity             [1, 0]               [0, 1]               [[1, 0], [0, 1]]
rotate by t          [cos t, sin t]       [-sin t, cos t]      [[cos t, -sin t], [sin t, cos t]]
scale                [sx, 0]              [0, sy]              [[sx, 0], [0, sy]]
shear                [1, 0]               [1, 1]               [[1, 1], [0, 1]]
reflect over y = x   [0, 1]               [1, 0]               [[0, 1], [1, 0]]
project onto x-axis  [1, 0]               [0, 0]               [[1, 0], [0, 0]]
```

Read each row by asking "where do the two arrows go?" and you can derive the matrix without memorizing it. Open `code/out.png` to see each one applied to a grid and a letter F:

- The rotation keeps every length and angle.
- The scaling stretches x and squashes y.
- The shear slides each horizontal line sideways in proportion to its height.
- The reflection flips the F into its mirror image (orientation reverses, which the determinant in Lesson 06 will detect).
- The projection flattens the whole plane onto a line. Many different inputs land on the same output, so the information is gone and cannot be undone (Lesson 05).

### In higher dimensions

Nothing changes. A `3 x 3` matrix has three columns: where `i_hat`, `j_hat`, `k_hat` land. An `m x n` matrix takes `n`-dimensional vectors (one coefficient per column) to `m`-dimensional vectors (the length of each column). A `2 x 3` matrix squashes 3-D space onto a plane.

## Build It

### Step 1: Matrices from basis images

```python
def matrix_from_basis_images(i_hat_lands_at, j_hat_lands_at):
    return np.column_stack([i_hat_lands_at, j_hat_lands_at]).astype(float)


def rotation(angle_degrees):
    angle = math.radians(angle_degrees)
    return matrix_from_basis_images([math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)])
```

Every matrix in the script is built this way, so you never have to remember a formula.

### Step 2: Column picture and row picture

```python
def apply_column_picture(matrix, vector):
    result = np.zeros(matrix.shape[0])
    for column_index, coefficient in enumerate(vector):
        result = result + coefficient * matrix[:, column_index]
    return result


def apply_row_picture(matrix, vector):
    return np.array([sum(row[k] * vector[k] for k in range(len(vector))) for row in matrix])
```

```
  column picture: 2 * [1. 0.] + 3 * [1. 1.] = [5. 3.]
  row picture:    [row0 . v, row1 . v] = [5. 3.]
```

### Step 3: The catalogue

```python
TRANSFORMATIONS = {
    "identity": matrix_from_basis_images([1, 0], [0, 1]),
    "rotate 30 deg": rotation(30),
    "scale x2, y0.5": matrix_from_basis_images([2, 0], [0, 0.5]),
    "shear": matrix_from_basis_images([1, 0], [1, 1]),
    "reflect over y=x": matrix_from_basis_images([0, 1], [1, 0]),
    "project onto x-axis": matrix_from_basis_images([1, 0], [0, 0]),
}
```

The script sends the point `(2, 1)` through each one, for example the reflection swaps it to `(1, 2)` and the projection drops it to `(2, 0)`.

### Step 4: Test linearity

```python
def is_linear(transform, first, second, scalar=2.5):
    additive = np.allclose(transform(first + second), transform(first) + transform(second))
    homogeneous = np.allclose(transform(scalar * first), scalar * transform(first))
    return additive and homogeneous


def translate(vector):
    return vector + np.array([1.0, 1.0])


def square_each(vector):
    return vector ** 2
```

```
  shear                linear: True
  translate by (1,1)   linear: False  (origin moves to [1. 1.])
  square each entry    linear: False  (grid lines curve)
```

Translation fails because `T(0) = [1, 1]`, but every linear map must send `0` to `0`. That is why a linear model needs a separate bias term `b`: `w . x` alone can only describe planes through the origin. (Checking two vectors is a quick test, not a proof; a non-linear function could pass by luck on specific inputs.)

### Step 5: What is preserved

```
  |v| = 3.6056, |R v| = 3.6056
  P [2, 3] = [2. 0.], P [2, -7] = [2. 0.]: different inputs, same output
```

Rotations preserve length. Projections collapse a whole line of inputs (everything above and below `(2, 0)`) onto one output.

## Use It

In NumPy, `M @ v` is the matrix-vector product. To transform many points stored as *rows* of an `n x 2` array, use `points @ M.T`:

```python
shear @ vector
many_points @ shear.T
```

The transpose appears because each row `p` must become `M p`, and `(M p)^T = p^T M^T`. You will see `X @ W.T` or `X @ W` constantly in ML code for this reason.

A transformation you already used is a matrix in disguise:

```python
scaler = StandardScaler().fit(features)
scaling_matrix = np.diag(1.0 / scaler.scale_)
np.allclose((features - scaler.mean_) @ scaling_matrix, scaler.transform(features))
```

`StandardScaler` is a translation (subtract the mean, not linear) followed by a diagonal scaling matrix, which stretches or squashes each axis independently. In picture terms: it moves the data cloud's center to the origin, then rescales the grid so every axis has the same spread.

## Exercises

1. **Easy.** Write the matrix that rotates by 90 degrees counter-clockwise by asking where `i_hat` and `j_hat` land. Apply it four times to `[1, 2]` and explain the result.
2. **Medium.** Build the matrix that reflects across the x-axis, and the one that projects onto the line `y = x` (hint: `i_hat` lands at `[0.5, 0.5]`; why?). Add both to `TRANSFORMATIONS` and check the figure.
3. **Hard.** Represent 2-D translation as a *linear* map in 3-D using homogeneous coordinates: write points as `[x, y, 1]` and find a `3 x 3` matrix whose effect on the first two coordinates is `+[1, 1]`. Use it to rotate the letter F by 30 degrees about the point `(1, 1)` instead of the origin.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Linear transformation | "A matrix" | A function T with T(v + w) = T(v) + T(w) and T(cv) = cT(v); grid lines stay straight, parallel, evenly spaced, origin fixed |
| Matrix | "A grid of numbers" | A record of a linear transformation: column k is where the k-th basis vector lands |
| Matrix-vector product | "Multiply the matrix by the vector" | A linear combination of the matrix's columns, weighted by the vector's entries |
| Column picture | "Combine the columns" | Reading M v as x1 * column1 + x2 * column2 + ... |
| Row picture | "Dot with the rows" | Reading entry i of M v as the dot product of row i with v |
| Rotation matrix | "Turn the plane" | A matrix whose columns are perpendicular unit vectors with positive orientation; preserves lengths and angles |
| Shear | "Slant" | A transformation that shifts points parallel to an axis by an amount proportional to their other coordinate |
| Projection matrix | "Flatten" | A matrix P with P P = P that sends every vector to its shadow on a subspace, discarding the perpendicular part |

## Further Reading

- [3Blue1Brown, Linear transformations and matrices (Essence of Linear Algebra, chapter 3)](https://www.3blue1brown.com/lessons/linear-transformations) -- the animation of grids moving that this lesson reproduces
- [3Blue1Brown, Nonsquare matrices as transformations between dimensions (chapter 8)](https://www.3blue1brown.com/lessons/nonsquare-matrices) -- what `m x n` matrices do when `m != n`
- [Mathematics for Machine Learning, section 2.7: Linear Mappings](https://mml-book.github.io/) -- formal treatment of transformation matrices and change of basis
- [Gilbert Strang, Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) -- chapter 1 develops the column picture and row picture side by side
