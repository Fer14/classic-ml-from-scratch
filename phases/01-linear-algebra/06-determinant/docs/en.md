# Determinant

> The determinant answers one question: by what factor does this transformation scale areas (or volumes)? Zero means it flattens space. Negative means it flips it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 Lesson 03 (Matrices as Transformations), Phase 01 Lesson 05 (Linear Systems and the Inverse)
**Time:** ~50 minutes

## Learning Objectives

- Interpret the determinant as the signed factor by which a transformation scales area (2-D) or volume (3-D)
- Explain why `det = 0` means the matrix is singular, and what a negative determinant says about orientation
- Compute determinants by cofactor expansion and by elimination (product of pivots), and explain why only the second scales
- Use `np.linalg.slogdet` to avoid underflow when determinants appear in probability densities

## The Problem

You will meet the determinant in three places later in this course. In Gaussian mixture models (Phase 08) and LDA/QDA (Phase 06), the probability density of a multivariate normal contains `det(covariance)`. If you compute it naively for 200 features, it underflows to exactly 0 and your log-likelihood becomes `-inf`. In the inverse and linear systems (Lesson 05), `det(A) = 0` is the test for "no unique solution". And in change-of-variables formulas (normalizing flows, density transforms), the determinant measures how much a transformation stretches probability mass.

The usual definition (an alternating sum of products of entries) tells you none of this. The geometric one does, and from it every property follows in one line.

## The Concept

### Area scale factor

Apply a linear transformation to the unit square. It becomes a parallelogram spanned by the two columns of the matrix (where `i_hat` and `j_hat` landed). The area of that parallelogram is the determinant.

```
     j_hat                       M j_hat = [b, d]
      |                            /----------/
      |____                       /          /    area = ad - bc
      |    |  area 1    --M-->   /          /
      |____|___ i_hat           /__________/
                                       M i_hat = [a, c]
```

```
det [[a, b], [c, d]] = a d - b c
```

Because the transformation is linear, every small square of the grid is scaled by the same factor. So *any* region's area is multiplied by `det(M)`: a triangle, a circle, the shape of your data cloud. The script checks this with a triangle.

In 3-D, `det` is the factor by which volumes scale: the volume of the parallelepiped spanned by the three columns.

### The sign is orientation

Before the transformation, `j_hat` is counter-clockwise from `i_hat`. If after the transformation the image of `j_hat` is clockwise from the image of `i_hat`, space has been flipped over, like turning a sheet of paper. The determinant is then negative. A reflection always has a negative determinant; a rotation always has `det = +1`.

### Zero means squashed

If `det(M) = 0`, the unit square is flattened to a line segment (or a point). The columns are parallel: the matrix has rank below `n`. A flattened space cannot be un-flattened, so `M` has no inverse, and `M x = b` has either no solution or infinitely many (Lesson 05).

### Properties, all from "scale factor"

```
det(I) = 1                       doing nothing scales nothing
det(A B) = det(A) det(B)         scale by B, then by A
det(A^-1) = 1 / det(A)           undoing a scale-by-k scales by 1/k
det(A^T) = det(A)                (less obvious geometrically, but true)
det(c A) = c^n det(A)            each of n directions scaled by c
```

None of these needs to be memorized once you picture areas.

### Computing it

**Cofactor expansion** (the textbook formula): expand along the first row,

```
det(A) = sum over j of (-1)^j * A[0, j] * det(minor_0j)
```

where `minor_0j` is `A` without row 0 and column j. It recurses down to 2x2 blocks. It is correct and takes about `n!` operations: 8! = 40,320, 20! is about 2.4 * 10^18.

**Elimination**: row-reduce to an upper-triangular `U` (Lesson 05). Subtracting a multiple of one row from another is a shear, which does not change area. Swapping two rows is a reflection, which flips the sign. The determinant of a triangular matrix is the product of its diagonal (the triangle just stretches each axis). So

```
det(A) = (-1)^(number of swaps) * product of pivots
```

About `n^3 / 3` operations. That is what every library does.

### Determinants in probability

The density of a `d`-dimensional normal distribution has `1 / sqrt((2 pi)^d det(Sigma))` in front. `det(Sigma)` is the volume of the data cloud's covariance ellipsoid: a spread-out cloud has a large determinant and a low peak density. With 200 features of variance 0.01, `det = 0.01^200 = 10^-400`, smaller than the smallest float64 (about `10^-308`). You must work with `log det(Sigma)` directly, which `np.linalg.slogdet` computes from the pivots without ever forming the tiny product.

## Build It

### Step 1: 2x2 determinant and the shoelace area

```python
def determinant_2x2(matrix):
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]


def shoelace_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
```

The shoelace formula computes the signed area of any polygon from its corners (positive when the corners go counter-clockwise). Transforming shapes and measuring their areas directly confirms the claim:

```
  scale x3, y2       det =   6.000   unit square -> signed area   6.000   triangle area ratio   6.000
  shear              det =   1.000   unit square -> signed area   1.000   triangle area ratio   1.000
  rotate 40 deg      det =   1.000   unit square -> signed area   1.000   triangle area ratio   1.000
  reflect + stretch  det =  -2.000   unit square -> signed area  -2.000   triangle area ratio  -2.000
  squash to a line   det =   0.000   unit square -> signed area   0.000   triangle area ratio   0.000
```

Open `code/out.png`: each panel shows the unit square (gray) and its image (blue). The shear slants the square without changing its area.

### Step 2: Cofactor expansion

```python
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
```

### Step 3: Elimination

```python
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
```

Both give `-1` for the 3x3 system from Lesson 05, and they agree on random matrices. The timing shows the difference:

```
  n=6: cofactor 0.0030 s (~720 terms), elimination 0.081 ms
  n=8: cofactor 0.1603 s (~40,320 terms), elimination 0.259 ms   (your timings will differ)
```

Going from 6 to 8 multiplied the cofactor time by about 56 (= 8 * 7). At n = 12 it would take minutes; elimination would still take under a millisecond.

### Step 4: Verify the properties

```
  det(AB) = 1.842646, det(A) det(B) = 1.842646
  det(A^-1) = -0.530571, 1 / det(A) = -0.530571
  det(2A) = -30.156201 = 2^4 det(A) = -30.156201
```

## Use It

```python
np.linalg.det(matrix)

sign, log_determinant = np.linalg.slogdet(covariance)
```

`np.linalg.det` uses LU factorization, exactly the elimination approach. On the 200-dimensional covariance it returns `0.0`, which is wrong (the matrix is perfectly invertible). `slogdet` returns the sign and `log |det|` separately, summing the logs of the pivots instead of multiplying them:

```
  det of a 200-dim covariance with variances 0.01: 0.0 (underflow)
  slogdet: sign 1.0, log|det| = -921.034 = 200 * log(0.01) = -921.034
```

Two practical rules:

- Never test invertibility with `det(A) == 0` or `abs(det(A)) < eps`. Scaling a matrix by 0.1 multiplies its determinant by `0.1^n` without making it any less invertible. Use `np.linalg.matrix_rank` or `np.linalg.cond`.
- In any likelihood, use `slogdet`. scikit-learn's `GaussianMixture` works with log-determinants for this reason.

## Exercises

1. **Easy.** Without computing, predict the determinant of `rotation(25) @ [[2, 0], [0, 3]] @ rotation(-60)`. Then check with `determinant_elimination`.
2. **Medium.** Compute the area of the triangle with corners `(1, 1)`, `(4, 2)`, `(2, 5)` as half the determinant of the two edge vectors from `(1, 1)`. Verify with `shoelace_area`.
3. **Hard.** Estimate a determinant by Monte Carlo: sample 200,000 points uniformly in the unit cube, map them through a 3x3 matrix `M`, and estimate the volume of the image by counting how many points of a bounding box land inside it (test membership by solving `M x = p` and checking `0 <= x <= 1`). Compare with `abs(det(M))`.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Determinant | "ad - bc" | The signed factor by which a linear transformation scales areas (2-D) or volumes (n-D) |
| Orientation | "Handedness" | Whether the transformed basis keeps the same rotational order; flipped orientation gives a negative determinant |
| Singular | "det is zero" | The transformation collapses at least one dimension; no inverse exists |
| Cofactor expansion | "Laplace expansion" | Recursive definition of the determinant via minors along a row; O(n!) work |
| Pivot product | "Determinant via LU" | det(A) = (-1)^swaps times the product of the pivots from elimination; O(n^3) work |
| Shoelace formula | "Polygon area" | Signed area of a polygon from its ordered corner coordinates |
| Log-determinant | "slogdet" | log of the absolute determinant, computed as a sum of logs to avoid underflow and overflow |

## Further Reading

- [3Blue1Brown, The determinant (Essence of Linear Algebra, chapter 6)](https://www.3blue1brown.com/lessons/determinant) -- the area-scaling view with animations
- [Gilbert Strang, MIT 18.06 Lectures 18-19](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- properties of determinants and the cofactor formula
- [numpy.linalg.slogdet documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html) -- why and how to compute log-determinants
- [Mathematics for Machine Learning, section 4.1](https://mml-book.github.io/) -- determinant and trace, with the volume interpretation
