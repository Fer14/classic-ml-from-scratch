# Matrix Multiplication

> Multiplying two matrices means doing one transformation and then the other. The famous row-times-column rule is just bookkeeping for that.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 Lesson 03 (Matrices as Transformations)
**Time:** ~60 minutes

## Learning Objectives

- Implement matrix multiplication with a triple loop and state the shape rule `(m x n) @ (n x p) -> (m x p)`
- Explain matrix multiplication as composition of transformations, read right to left
- Show that matrix multiplication is associative but not commutative, and use `(AB)^T = B^T A^T`
- Use one matrix product to compute predictions for a whole batch, and explain why stacked linear layers collapse into one

## The Problem

A model with 3 features and 2 outputs has a `3 x 2` weight matrix. You have 10,000 samples. You could loop over samples and compute dot products one by one, or you could write `X @ W` and get all 20,000 numbers in one call that runs a thousand times faster. Every model in this course, from linear regression to PCA, is a handful of matrix products.

Matrix multiplication is usually taught as a rule to memorize ("row times column"). That rule hides what is happening, and it makes properties like "order matters" seem arbitrary. When you see a product as two transformations applied in sequence, every property becomes obvious, and you will be able to look at `A @ B @ x` and know what it does to your data.

## The Concept

### Composition

In Lesson 03 you saw that a matrix is a transformation. Apply a rotation `R`, then a shear `S`, to a vector `v`:

```
S (R v)
```

The result is a new linear transformation (grid lines are still straight, parallel, evenly spaced). So it has its own matrix. Call it `S R`. Its columns are where `i_hat` and `j_hat` end up after *both* steps:

```
first column of S R  =  S (R i_hat)  =  S @ (first column of R)
second column of S R =  S (R j_hat)  =  S @ (second column of R)
```

That is the definition of matrix multiplication: **apply the left matrix to each column of the right matrix.** And it is read right to left, like function composition `f(g(x))`: in `S R`, `R` happens first.

### The row-times-column rule falls out

Applying `S` to a column is a matrix-vector product, and each entry of that is a dot product of a row of `S` with the column (the row picture). So:

```
(A B)[i, j] = row i of A  .  column j of B = sum over k of A[i, k] * B[k, j]
```

```
          B (n x p)
          [ . b . ]
          [ . b . ]
          [ . b . ]
A (m x n)
[ a a a ]  [ . * . ]     * = a . b  (row i of A dot column j of B)
[ . . . ]  [ . . . ]
           (m x p)
```

### The shape rule

```
(m x n) @ (n x p)  ->  (m x p)
     ^     ^
     must match
```

The inner dimensions must agree: `B` outputs `n`-dimensional vectors and `A` must accept `n`-dimensional vectors. The outer dimensions give the result's shape. When NumPy says `matmul: Input operand 1 has a mismatch in its core dimension`, check this first.

### Four views of the same product

| View | Formula | When it helps |
|------|---------|---------------|
| Dot products | entry (i, j) = row i of A . column j of B | computing by hand |
| Columns | column j of AB = A @ (column j of B) | geometry, composition |
| Rows | row i of AB = (row i of A) @ B | ML: each sample row times the weights |
| Outer products | AB = sum over k of (column k of A)(row k of B) | low-rank approximations, SVD (Phase 09) |

### Properties

- **Not commutative**: `AB != BA` in general. Rotating then shearing is not the same as shearing then rotating. Look at `code/out.png`: the two Fs end up in different places.
- **Associative**: `(AB)C = A(BC)`. Applying C, then B, then A is the same sequence whichever pair you pre-combine. This is what lets you collapse a chain of transformations into one matrix.
- **Transpose reverses order**: `(AB)^T = B^T A^T`.
- **Identity**: `A I = I A = A`, where `I` leaves every basis vector in place.

### Why it matters in ML

- **Batch prediction**: if `X` is `n x d` (one sample per row) and `W` is `d x k`, then `X @ W` is `n x k`: every sample's `k` outputs, using the row view.
- **Linear layers collapse**: `(X W1) W2 = X (W1 W2)`. Two linear layers without a non-linearity between them are exactly one linear layer. Neural networks insert non-linear activation functions precisely to break this.

## Build It

### Step 1: The triple loop

```python
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
```

```
  (2x3) @ (3x2) -> (2, 2)
  entry [0, 1] = row 0 of left . column 1 of right = [1. 2. 0.] . [1. 2. 1.] = 5.0
  left @ left fails: shapes (2, 3) and (2, 3) do not align: 3 != 2
```

### Step 2: Column view and outer-product view

```python
def matmul_columns(left, right):
    return np.column_stack([sum(right[k, j] * left[:, k] for k in range(left.shape[1])) for j in range(right.shape[1])])


def matmul_outer_products(left, right):
    return sum(np.outer(left[:, k], right[k, :]) for k in range(left.shape[1]))
```

`matmul_columns` builds each result column as a linear combination of the columns of `left`, which is exactly `left @ (column j of right)` in the column picture. `matmul_outer_products` adds up `n` rank-1 matrices. All three agree.

### Step 3: Composition

```python
SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])
ROTATE_90 = rotation(90)

step_by_step = SHEAR @ (ROTATE_90 @ vector)
composed = matmul_dot_products(SHEAR, ROTATE_90)
```

```
  rotate then shear, one step at a time: [-1.  1.]
  (SHEAR @ ROTATE_90) @ v:               [-1.  1.]
  SHEAR @ ROTATE_90 columns are where i_hat, j_hat end up:
[[ 1. -1.]
 [ 1.  0.]]
```

Check the first column by hand: rotating `i_hat` by 90 degrees gives `[0, 1]`; shearing `[0, 1]` gives `[1, 1]`. That is column one.

### Step 4: Properties

```
  ROTATE_90 @ SHEAR:
[[ 0. -1.]
 [ 1.  1.]]
  commutative? False
  associative (AB)C == A(BC)? True
  transpose rule (AB)^T == B^T A^T? True
```

### Step 5: Batches and layers

```python
predictions = matmul_dot_products(samples, weights)
two_layers = matmul_dot_products(matmul_dot_products(samples, first_hidden), second_hidden)
collapsed = matmul_dot_products(samples, matmul_dot_products(first_hidden, second_hidden))
```

`two_layers` and `collapsed` are equal: depth without non-linearity buys nothing.

## Use It

```python
left @ right
np.matmul(left, right)
np.einsum("ik,kj->ij", left, right)
```

`@` is the operator you will use; `np.einsum` spells out the index formula `sum over k of A[i,k] B[k,j]` literally, which is handy for unusual contractions such as batched products.

The speed difference is the reason you never write the loop:

```
  80x80: triple loop 0.100 s, @ 0.089 ms, ~1,121x faster, same result: True
```

(Your timings will differ.) NumPy's `@` calls BLAS, a library tuned for decades: it uses SIMD instructions, cache-friendly blocking, and multiple cores. The loop is `O(n^3)` and so is BLAS, but with a constant factor about a thousand times smaller.

One practical warning: `*` on two NumPy arrays is *element-wise* multiplication, not matrix multiplication. `A * B` with two `2 x 2` arrays silently returns a different `2 x 2` array. Use `@`.

## Exercises

1. **Easy.** Compute `[[2, 0], [0, 3]] @ [[1, 4], [2, 5]]` by hand with the dot-product rule, then with the column view. Then reverse the order. Describe geometrically what the diagonal matrix does in each case (hint: in one case it scales rows, in the other columns).
2. **Medium.** Show numerically that `rotation(30) @ rotation(45) == rotation(75)`, and that rotations commute with each other even though matrices generally do not. Why does this make geometric sense?
3. **Hard.** Count floating-point operations. For `(m x n) @ (n x p)`, the triple loop does `m * n * p` multiplications. Given `A` (1000 x 2), `B` (2 x 1000), and `x` (1000 x 1), compare the cost of `(A @ B) @ x` and `A @ (B @ x)`. Time both with NumPy. Why does associativity matter for performance?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Matrix multiplication | "Row times column" | The matrix of the composed transformation: AB applies B first, then A |
| Composition | "Do one, then the other" | Applying one function to the output of another; for matrices, multiplication |
| Inner dimension | "The shapes must match" | In (m x n) @ (n x p), the shared n; columns of the left must equal rows of the right |
| Commutative | "Order does not matter" | AB = BA; false for matrices in general |
| Associative | "Grouping does not matter" | (AB)C = A(BC); true for matrix multiplication |
| Transpose | "Flip rows and columns" | A^T[i, j] = A[j, i]; satisfies (AB)^T = B^T A^T |
| Outer product | "Column times row" | u v^T, a rank-1 matrix whose (i, j) entry is u_i * v_j |
| BLAS | "The fast backend" | Basic Linear Algebra Subprograms, optimized low-level routines that NumPy calls for @ |

## Further Reading

- [3Blue1Brown, Matrix multiplication as composition (Essence of Linear Algebra, chapter 4)](https://www.3blue1brown.com/lessons/matrix-multiplication) -- the composition view, animated
- [3Blue1Brown, Three-dimensional linear transformations (chapter 5)](https://www.3blue1brown.com/lessons/3d-transformations) -- the same ideas in 3-D
- [Gilbert Strang, MIT 18.06 Lecture 3: Multiplication and inverse matrices](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- the four ways to multiply matrices
- [numpy.matmul documentation](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) -- broadcasting and stacking rules for `@`
