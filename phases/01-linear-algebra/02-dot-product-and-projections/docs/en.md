# Dot Product and Projections

> The dot product asks one question: how much does one vector go in the direction of another? Similarity, projection, and every linear model are answers to it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 Lesson 01 (Vectors)
**Time:** ~60 minutes

## Learning Objectives

- Compute the dot product algebraically (sum of products) and interpret it geometrically (`|v| |w| cos theta`)
- Project one vector onto another and split a vector into parallel and perpendicular parts
- Use cosine similarity to compare documents, and explain why it ignores vector length
- Build an orthonormal basis with Gram-Schmidt and verify it against `np.linalg.qr`

## The Problem

Three questions come up constantly in machine learning:

1. **How similar are two items?** Two documents, two users' purchase histories, two embeddings.
2. **What is the prediction?** A linear model computes `w1 x1 + w2 x2 + ... + b` for every sample.
3. **What part of this vector lies along that direction?** The core of PCA (Phase 09), of least squares (Phase 05), and of removing a component you do not want.

All three are the same operation: the dot product. It is a single line of code, but it has a geometric meaning that makes all three questions obvious once you see it.

## The Concept

### Two formulas, one number

Algebraically, multiply matching components and add:

```
v . w = v1*w1 + v2*w2 + ... + vn*wn
```

Geometrically, it is the product of the lengths times the cosine of the angle between them:

```
v . w = |v| * |w| * cos(theta)
```

For `v = [4, 1]` and `w = [2, 3]`: `4*2 + 1*3 = 11`, and `sqrt(17) * sqrt(13) * cos(42.27 deg) = 11`. Same number.

Why are these equal? 3Blue1Brown's explanation: the operation "take the dot product with `v`" is itself a linear transformation from 2-D to the number line. A linear map to 1-D is described by a 1x2 matrix, and that matrix's entries are exactly `v`'s coordinates. This *duality* (a vector is also a little "measure along me" machine) is why the sum-of-products formula carries geometric meaning.

### The sign tells you the side

```
              w (same side: v.w > 0, angle < 90)
             /
            /
  ---------o--------->  v
           |\
           | \
           |  w' (opposite side: v.w' < 0, angle > 90)
           w'' (perpendicular: v.w'' = 0)
```

- `v . w > 0`: they point roughly the same way (angle below 90 degrees).
- `v . w = 0`: perpendicular, also called *orthogonal*.
- `v . w < 0`: roughly opposite.

A linear classifier (Phase 06) is exactly this: predict class 1 if `w . x + b > 0`.

### Length is a dot product with itself

```
|v| = sqrt(v . v) = sqrt(v1^2 + v2^2 + ... + vn^2)
```

This is the Pythagorean theorem in n dimensions.

### Projection: the shadow

Shine a light perpendicular to the line through `v`. The shadow that `w` casts on that line is its *projection*:

```
proj_v(w) = (w . v / v . v) * v

             w
            /|
           / |   <- perpendicular part: w - proj_v(w)
          /  |
  -------o---*-------->  v
         |___|
       proj_v(w)
```

If `v` has length 1, this simplifies to `(w . v) v`, and `w . v` is the *signed length* of the shadow. That is the other way to read the dot product: **project `w` onto the direction of `v`, then multiply by the length of `v`.**

Every vector splits into a part along `v` and a part perpendicular to `v`:

```
w = proj_v(w) + (w - proj_v(w))        and        (w - proj_v(w)) . v = 0
```

### Cosine similarity

Divide the dot product by both lengths and you are left with only the angle:

```
cos_sim(v, w) = (v . w) / (|v| |w|)        in [-1, 1]
```

For word-count vectors (all entries non-negative) it ranges from 0 (no shared words) to 1 (same proportions). A document repeated twice has double the counts but the same direction, so its cosine similarity with the original is exactly 1. That is usually what you want: long and short documents about the same topic should look alike.

### Gram-Schmidt: perpendicular axes from any basis

Given independent vectors `a1, a2, a3`, build perpendicular unit vectors:

```
q1 = a1 / |a1|
q2 = (a2 - proj_q1(a2))                  normalized
q3 = (a3 - proj_q1(a3) - proj_q2(a3))    normalized
```

Each step removes the shadows on all previous axes, leaving only the genuinely new direction. Perpendicular unit axes (an *orthonormal basis*) make coordinates trivial: the coordinate along `qi` is just `x . qi`.

## Build It

### Step 1: Dot product, length, angle

```python
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
```

The `max(-1, min(1, ...))` clamp protects `acos` from rounding errors like `1.0000000000000002`.

### Step 2: Projection and decomposition

```python
def project(vector, onto):
    return (dot(vector, onto) / dot(onto, onto)) * np.asarray(onto, dtype=float)


shadow = project(w, v)
perpendicular_part = w - shadow
```

```
  proj of w onto v      = [2.5882 0.6471]
  w - proj (the rest)   = [-0.5882  2.3529]
  (w - proj) . v        = -4.44e-16  (perpendicular)
```

`-4.44e-16` is zero up to floating-point rounding. Open `code/out.png` to see the shadow (green) and the dashed perpendicular drop.

### Step 3: A linear model is a dot product

```python
weights = np.array([3000.0, 15000.0, -800.0])
bias = 50000.0
house = np.array([120.0, 3.0, 15.0])
print(dot(weights, house) + bias)

perpendicular_step = np.array([5.0, -1.0, 0.0])
other_house = house + perpendicular_step
```

`weights . perpendicular_step = 3000*5 - 15000 = 0`. Moving a house in a direction perpendicular to `w` (5 more square meters, one fewer room) does not change its predicted price. All houses with the same price form a plane perpendicular to `w`; the model only "sees" the component of `x` along `w`.

### Step 4: Cosine similarity

```python
def cosine_similarity_pair(first, second):
    return dot(first, second) / (length(first) * length(second))
```

On four word-count vectors over `[cat, dog, model, data, pet]`:

```
    1.000  0.714  0.000  1.000
    0.714  1.000  0.000  0.714
    0.000  0.000  1.000  0.000
    1.000  0.714  0.000  1.000
```

Documents 0 and 1 (pets) are similar, document 2 (ML) shares no words with them, and document 3 is document 0 doubled, so their similarity is 1.

### Step 5: Gram-Schmidt

```python
def gram_schmidt(vectors):
    orthonormal = []
    for vector in vectors:
        residual = np.asarray(vector, dtype=float)
        for basis_vector in orthonormal:
            residual = residual - project(residual, basis_vector)
        if length(residual) > 1e-10:
            orthonormal.append(residual / length(residual))
    return np.array(orthonormal)
```

Subtracting from the running `residual` rather than from the original vector is the *modified* Gram-Schmidt variant; it is mathematically identical but loses less precision. The result satisfies `Q Q^T = I`: every pair of rows has dot product 0 and every row has length 1.

## Use It

```python
np.dot(v, w)
v @ w
np.linalg.norm(v)

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(documents)

q_matrix, r_matrix = np.linalg.qr(raw_vectors.T)
```

All match the from-scratch versions. `np.linalg.qr` performs the same job as Gram-Schmidt (using the more stable Householder reflections) and returns the orthonormal vectors as the columns of `Q`. It may flip the sign of some axes, since `-q` is as valid an axis as `q`, so the script compares up to sign. You will use QR again when solving least squares in Phase 05.

In practice: use `@` for dot products and matrix products, `np.linalg.norm` for lengths, and `cosine_similarity` for comparing rows of a matrix at once.

## Exercises

1. **Easy.** Find a vector perpendicular to `[3, 4]` and verify with `dot`. Then find the angle between `[1, 1, 0]` and `[0, 1, 1]` by hand and with `angle_degrees`.
2. **Medium.** Write `project_onto_plane(x, a, b)` that projects a 3-D vector onto the plane spanned by `a` and `b`. Hint: run `gram_schmidt([a, b])` first, then add the projections onto each orthonormal axis. Check that `x - projection` is perpendicular to both `a` and `b`.
3. **Hard.** Feed `gram_schmidt` three vectors where the third is nearly in the plane of the first two, such as `[1, 0, 0]`, `[1, 1e-8, 0]`, `[1, 0, 1e-8]`. Implement the *classical* variant (subtract projections of the original vector) and compare `Q Q^T` for both. Which one loses orthogonality, and why?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Dot product | "Multiply and add" | `v . w = sum(vi * wi)`, equal to length(v) * length(w) * cos(theta): how much two vectors point the same way, scaled by their lengths |
| Norm (L2) | "Length" | `sqrt(v . v)`, the Euclidean length of a vector |
| Orthogonal | "Perpendicular" | Two vectors whose dot product is zero |
| Projection | "Shadow" | `proj_v(w) = (w . v / v . v) v`, the component of `w` along the line of `v` |
| Unit vector | "Direction" | A vector of length 1, obtained by dividing a vector by its norm |
| Cosine similarity | "Angle similarity" | Dot product divided by both norms; depends only on direction, not length |
| Orthonormal basis | "Clean axes" | A basis of mutually perpendicular unit vectors; coordinates are simple dot products |
| Gram-Schmidt | "Orthogonalization" | Turning independent vectors into an orthonormal basis by repeatedly subtracting projections onto earlier axes |

## Further Reading

- [3Blue1Brown, Dot products and duality (Essence of Linear Algebra, chapter 9)](https://www.3blue1brown.com/lessons/dot-products) -- why the sum-of-products formula equals projection times length
- [Mathematics for Machine Learning, chapter 3: Analytic Geometry](https://mml-book.github.io/) -- inner products, angles, orthogonal projections, and Gram-Schmidt
- [scikit-learn: cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) -- the library function used in "Use It"
- [Gilbert Strang, MIT 18.06 Lectures 15-17](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- projections onto subspaces, least squares, and Gram-Schmidt
