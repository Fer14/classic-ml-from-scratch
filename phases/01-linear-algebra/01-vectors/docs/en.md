# Vectors

> A vector is an arrow and a list of numbers at the same time. Machine learning works because you can switch between the two views whenever it helps.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 00 (Python Foundations), especially Lesson 05 (NumPy)
**Time:** ~60 minutes

## Learning Objectives

- Interpret a vector both as an arrow from the origin and as a list of coordinates, and move between the two views
- Compute vector addition, scalar multiplication, and linear combinations, and draw each one geometrically
- Explain span, linear dependence, and basis, and test whether two 2-D vectors are collinear
- Find the coordinates of a vector in a non-standard basis, and verify with `np.linalg.solve`

## The Problem

A house is 120 square meters, has 3 rooms, and is 15 years old. A customer bought products 4, 9, and 17. An image is 784 pixel intensities. In machine learning every one of these becomes the same thing: a list of numbers, a *vector*. Every model you will build takes vectors in and does arithmetic on them.

If vectors are just lists, why learn any geometry? Because the questions you care about are geometric. "Which houses are similar?" is a question about distance between arrows. "Which direction in feature space separates spam from ham?" is a question about directions. "Can these three features be compressed into two?" is a question about span. The list view lets you compute; the arrow view lets you think. This phase follows the approach of 3Blue1Brown's *Essence of Linear Algebra*: always ask what the operation does to the picture.

## The Concept

### Two views of the same object

```
     y
     ^
   2 |
     |
   1 |              * <- tip of v, the point (3, 1)
     |        ___/
     |  ___/
   0 +----+----+----+----> x
     0    1    2    3

  arrow view:   start at the origin, go 3 right and 1 up
  list view:    v = [3, 1]
```

In ML the list view comes first (a row of a dataset), but the arrow view tells you what the operations mean. A vector in this course is always rooted at the origin; its coordinates are where its tip lands.

### Addition: tip to tail

```
v + w = [v1 + w1, v2 + w2]
```

Geometrically: walk along `v`, then from its tip walk along `w`. Where you end up is `v + w`. Addition is "do one movement, then the other", which is why the order does not matter (`v + w = w + v`, the two paths form a parallelogram).

### Scalar multiplication: stretch and flip

```
c * v = [c * v1, c * v2]
```

Multiplying by a number *scales* the arrow: `2v` is twice as long, `0.5v` half as long, `-v` points the opposite way. That is why plain numbers are called *scalars*.

### Linear combinations and span

Combine both operations and you get a *linear combination*:

```
a * v + b * w        a, b any real numbers
```

The *span* of `v` and `w` is the set of every point you can reach this way. Picture two dials, one for `a` and one for `b`. Turn them and the tip of `a v + b w` sweeps out:

- the whole plane, if `v` and `w` point in different directions;
- only a line, if they are *collinear* (one is a multiple of the other);
- only the origin, if both are zero.

In 3-D, two non-collinear vectors span a plane through the origin. A third vector either lies in that plane (adding nothing new) or escapes it (and the three span all of space).

### Linear dependence and basis

Vectors are *linearly dependent* if one of them is a linear combination of the others: it is redundant and does not enlarge the span. A *basis* is a set of linearly independent vectors that spans the whole space. The standard basis of 2-D is

```
i_hat = [1, 0]      j_hat = [0, 1]
```

and the coordinates `[3, 1]` really mean `3 * i_hat + 1 * j_hat`. Coordinates are always "how much of each basis vector". Change the basis and the same arrow gets different coordinates.

### Testing collinearity in 2-D

Two vectors `v` and `w` define a parallelogram. Its signed area is

```
area(v, w) = v1 * w2 - v2 * w1
```

If the area is zero, the parallelogram is flat, so `v` and `w` lie on the same line. You will meet this quantity again as the *determinant* in Lesson 06.

### Coordinates in another basis

To write `target = a v + b w`, solve two equations for `a` and `b`. Using the area trick (Cramer's rule):

```
a = area(target, w) / area(v, w)
b = area(v, target) / area(v, w)
```

Replacing `v` by `target` in the area tells you how much of `v` is needed. If `area(v, w) = 0` there is no unique answer, because `v` and `w` do not span the plane.

## Build It

### Step 1: Addition and scaling, element by element

```python
def add(first, second):
    return np.array([a + b for a, b in zip(first, second)], dtype=float)


def scale(scalar, vector):
    return np.array([scalar * component for component in vector], dtype=float)


v = np.array([3.0, 1.0])
w = np.array([-1.0, 2.0])
print(add(v, w))
print(add(v, scale(-1, w)))
```

Subtraction is not a new operation: `v - w = v + (-1) w`. Geometrically, `v - w` is the arrow from the tip of `w` to the tip of `v`, which is why distance between two points is the length of their difference (Lesson 08).

### Step 2: Linear combinations

```python
def linear_combination(coefficients, vectors):
    result = np.zeros(len(vectors[0]))
    for coefficient, vector in zip(coefficients, vectors):
        result = add(result, scale(coefficient, vector))
    return result


i_hat = np.array([1.0, 0.0])
j_hat = np.array([0.0, 1.0])
print(linear_combination([3, 1], [i_hat, j_hat]))
print(linear_combination([2, -1], [v, w]))
```

The first line prints `[3. 1.]`: `v` is literally three `i_hat` plus one `j_hat`.

### Step 3: Span and collinearity

```python
def signed_area_2d(first, second):
    return first[0] * second[1] - first[1] * second[0]


def are_collinear_2d(first, second, tolerance=1e-12):
    return abs(signed_area_2d(first, second)) < tolerance
```

```
  signed area of (v, w) = 7.0 -> collinear? False
  signed area of (v, u) = 0.0 -> collinear? True
```

`u = [-6, -2] = -2v`, so `v` and `u` only span a line.

### Step 4: Coordinates in a new basis

```python
def coordinates_in_basis_2d(basis_one, basis_two, target):
    area = signed_area_2d(basis_one, basis_two)
    if abs(area) < 1e-12:
        raise ValueError("basis vectors are collinear: they only span a line")
    first_coordinate = signed_area_2d(target, basis_two) / area
    second_coordinate = signed_area_2d(basis_one, target) / area
    return np.array([first_coordinate, second_coordinate])
```

```
  target [4. 7.] = 2.143 * v + 2.429 * w
  check: [4. 7.]
```

The right panel of `code/out.png` shows this: the gray grid is built from copies of `v` and `w` instead of the usual squares. The red arrow `[4, 7]` sits 2.14 steps along `v` and 2.43 steps along `w` in that tilted grid. Same arrow, different coordinates.

### Step 5: The same ideas in 3-D

```python
a = np.array([1.0, 0.0, 2.0])
b = np.array([0.0, 1.0, 1.0])
c = linear_combination([2, 3], [a, b])
d = np.array([0.0, 0.0, 1.0])
print(np.linalg.matrix_rank(np.column_stack([a, b, c])))
print(np.linalg.matrix_rank(np.column_stack([a, b, d])))
```

The *rank* counts how many independent directions a set of vectors contains, which is the dimension of their span. `[a, b, c]` has rank 2 (a plane), `[a, b, d]` rank 3 (all of 3-D space). Here you borrow NumPy's `matrix_rank`; you will be able to compute it yourself after Lesson 05.

## Use It

NumPy arrays already implement vector arithmetic with operators:

```python
print(np.allclose(v + w, add(v, w)))
print(np.allclose(2.5 * v, scale(2.5, v)))
print(np.allclose(2 * v - w, combination))

basis_matrix = np.column_stack([v, w])
numpy_coordinates = np.linalg.solve(basis_matrix, target)
print(basis_matrix @ coordinates)
```

Notice the last two lines. Stacking `v` and `w` as the *columns* of a matrix turns "find a, b with `a v + b w = target`" into "solve `M x = target`", and `M @ x` computes the linear combination of the columns. That is the bridge to the next lessons: a matrix is a bundle of vectors, and multiplying a matrix by a vector is a linear combination of its columns.

In ML code you will almost never write `add` or `scale`. You will write `X[i] + X[j]` and `0.1 * gradient`, and you should picture arrows when you do.

## Exercises

1. **Easy.** Take `v = [2, 1]` and `w = [4, 2]`. Compute `signed_area_2d(v, w)`, explain the result geometrically, and show that `coordinates_in_basis_2d(v, w, [1, 1])` raises an error. Is `[6, 3]` in their span?
2. **Medium.** Add a third panel to the figure that shows the span of `v` and `u = -2v`: plot the points `a v + b u` for integer `a, b` in `[-2, 2]` and observe that they all lie on one line.
3. **Hard.** Generalize `coordinates_in_basis_2d` to 3-D using the scalar triple product `det[a, b, c] = a . (b x c)` (implement the cross product yourself). Use it to express `[1, 2, 3]` in the basis `a = [1, 0, 2]`, `b = [0, 1, 1]`, `d = [0, 0, 1]`, and verify with `np.linalg.solve`.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "A list of numbers" | An element of a vector space; in R^n, an ordered list of n numbers, drawn as an arrow from the origin to that point |
| Scalar | "A number" | A single real number used to scale vectors |
| Linear combination | "A mix of vectors" | A sum `c1 v1 + c2 v2 + ... + ck vk` with scalar coefficients |
| Span | "Everything you can reach" | The set of all linear combinations of a given set of vectors |
| Linearly dependent | "Redundant vectors" | At least one vector is a linear combination of the others, so removing it does not shrink the span |
| Basis | "Coordinate axes" | A linearly independent set of vectors whose span is the whole space; every vector has unique coordinates in it |
| Rank | "Number of independent directions" | The dimension of the span of a set of vectors (or of a matrix's columns) |
| Feature vector | "A row of the dataset" | The vector of numeric features describing one sample, a point in feature space |

## Further Reading

- [3Blue1Brown, Essence of Linear Algebra (full series)](https://www.3blue1brown.com/topics/linear-algebra) -- the visual intuition this phase is built on; watch chapters 1 ("Vectors, what even are they?") and 2 ("Linear combinations, span, and basis vectors") alongside this lesson
- [Mathematics for Machine Learning, chapter 2](https://mml-book.github.io/) -- free book by Deisenroth, Faisal, and Ong; rigorous treatment of vector spaces, linear independence, and bases
- [Gilbert Strang, MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- the classic lecture course; lecture 1 covers the column picture of linear combinations
