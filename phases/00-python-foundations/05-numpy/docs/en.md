# NumPy

> NumPy is not a faster list. It is a different way of thinking: you describe the operation on the whole array, and the loop disappears into C.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 00 Lesson 04 (Functions and Classes)
**Time:** ~75 minutes

## Learning Objectives

- Create NumPy arrays and reason about their `shape`, `dtype`, and `ndim`
- Use indexing, slicing, boolean masks, and axis reductions (`mean(axis=0)`) instead of loops
- Apply broadcasting rules to combine arrays of different shapes, including the `np.newaxis` trick for pairwise operations
- Rewrite list-based code as vectorized NumPy code and verify it against scikit-learn and SciPy

## The Problem

In the previous lessons you computed statistics with Python loops. That is fine for 14 temperatures. A real dataset has 100,000 rows and 50 columns. A k-nearest-neighbors model (Phase 06) computes the distance from every test point to every training point: billions of subtractions. In pure Python that takes hours, because every `a - b` goes through the interpreter, checks types, allocates a new float object, and so on.

NumPy stores numbers in a contiguous block of memory with a single type, and runs the loop in compiled C. You write `x - y` on whole arrays and get the speed of C with the readability of math. Every algorithm from Phase 01 onward is written in NumPy, so this lesson is the one you will reuse the most.

## The Concept

### An array is a typed block of memory plus a shape

```
data_as_lists:  [[10.9, 17.2, ...], [ ... ], ...]   400 lists of 5 Python float objects
                                                     each float is a separate heap object

data = np.array(data_as_lists)
       +-------------------------------------------+
       | 10.9 | 17.2 | 27.9 | 41.4 | 44.9 | ...    |   2000 float64 values, contiguous
       +-------------------------------------------+
       shape = (400, 5)   dtype = float64   ndim = 2
```

`shape` tells NumPy how to interpret the flat block as rows and columns. `reshape` changes the shape without copying data.

### Axes

For a 2-D array, axis 0 runs down the rows and axis 1 runs across the columns. A reduction *along* an axis collapses that axis:

```
           axis 1 ->
         [[ a  b  c ]
axis 0    [ d  e  f ]]
  |
  v       sum(axis=0) -> [a+d, b+e, c+f]     one value per column, shape (3,)
          sum(axis=1) -> [a+b+c, d+e+f]      one value per row,    shape (2,)
```

In ML, rows are samples and columns are features, so `X.mean(axis=0)` is the per-feature mean. You will write it hundreds of times.

### Broadcasting

When you combine two arrays of different shapes, NumPy compares shapes from the right. Two dimensions are compatible if they are equal or one of them is 1. A dimension of size 1 is (virtually) stretched to match:

```
(400, 5) - (5,)          -> (400, 5)     subtract the column means from every row
(2, 1)   + (3,)          -> (2, 3)       outer sum
(400, 1, 5) - (1, 400, 5) -> (400, 400, 5) every row minus every other row
```

The last line is the pairwise-difference trick. `matrix[:, np.newaxis, :]` inserts an axis of size 1, turning shape `(400, 5)` into `(400, 1, 5)`. Broadcasting does the rest, without any Python loop.

### Boolean masks

A comparison on an array returns an array of `True`/`False`. Using it as an index selects the matching elements:

```
mask = data[:, 0] > 11          shape (400,) of bools
data[mask, 1]                   feature 1 of only the rows where feature 0 > 11
mask.sum()                      how many rows matched (True counts as 1)
np.where(cond, a, b)            elementwise "if cond then a else b"
```

### Standardization

You will use this transformation constantly (Phase 04, Feature Scaling):

```
z = (x - mean) / std        computed per column
```

After it, every column has mean 0 and standard deviation 1. In NumPy it is one line, thanks to broadcasting.

## Build It

### Step 1: The list-based versions

These are what you could write with Lessons 02-03 alone:

```python
def standardize_lists(rows):
    n_rows = len(rows)
    n_columns = len(rows[0])
    means = [sum(row[j] for row in rows) / n_rows for j in range(n_columns)]
    stds = [math.sqrt(sum((row[j] - means[j]) ** 2 for row in rows) / n_rows) for j in range(n_columns)]
    return [[(row[j] - means[j]) / stds[j] for j in range(n_columns)] for row in rows]


def pairwise_distances_lists(rows):
    distances = []
    for first in rows:
        distance_row = []
        for second in rows:
            squared = 0.0
            for a, b in zip(first, second):
                squared += (a - b) ** 2
            distance_row.append(math.sqrt(squared))
        distances.append(distance_row)
    return distances
```

Three nested loops for the distances. Count them: that is 400 x 400 x 5 = 800,000 interpreted operations.

### Step 2: Arrays, indexing, and reductions

```python
data = np.array(data_as_lists)
print(f"  shape={data.shape} ndim={data.ndim} dtype={data.dtype} size={data.size}")
print(f"  first row        {np.round(data[0], 2)}")
print(f"  column 2, rows 0-3 {np.round(data[:4, 2], 2)}")
print(f"  column means (axis=0)  {np.round(data.mean(axis=0), 2)}")
```

`data[:4, 2]` reads as "rows 0 to 3, column 2". Each axis gets its own slice, separated by a comma.

### Step 3: Boolean masks

```python
mask = data[:, 0] > 11
print(f"  rows with feature 0 > 11: {mask.sum()} of {len(mask)}")
print(f"  mean of feature 1 on those rows: {data[mask, 1].mean():.3f}")
clipped = np.where(data > 40, 40.0, data)
```

### Step 4: Vectorized versions with broadcasting

```python
def standardize_numpy(matrix):
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)


def pairwise_distances_numpy(matrix):
    differences = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
    return np.sqrt((differences ** 2).sum(axis=-1))
```

Read `pairwise_distances_numpy` shape by shape: `(400, 1, 5) - (1, 400, 5)` gives `(400, 400, 5)`, squaring keeps the shape, summing over the last axis gives `(400, 400)`. The math is the same as the triple loop, but it is one expression.

The trade-off: the intermediate array holds 400 x 400 x 5 floats. For 100,000 points that would be 400 GB. In Phase 01 Lesson 08 you will see the memory-friendly identity `|a - b|^2 = |a|^2 + |b|^2 - 2 a.b`.

### Step 5: Measure

```python
def timed(function, *arguments):
    start = time.perf_counter()
    result = function(*arguments)
    return result, time.perf_counter() - start


distances_lists, list_seconds = timed(pairwise_distances_lists, data_as_lists)
distances_numpy, numpy_seconds = timed(pairwise_distances_numpy, data)
print(f"  same result: {np.allclose(distances_lists, distances_numpy)}")
```

On a typical laptop the NumPy version is around 10x faster here, and the gap grows with the number of features and with operations that call optimized libraries (matrix multiplication can be 1000x faster). Timings vary by machine; `np.allclose` checks the results agree up to floating-point rounding.

## Use It

scikit-learn and SciPy provide the same operations, tested and optimized:

```python
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

sklearn_standardized = StandardScaler().fit_transform(data)
scipy_distances = cdist(data, data)
print(np.allclose(sklearn_standardized, standardized_numpy))
print(np.allclose(scipy_distances, distances_numpy))
```

Both print `True`. Note that `StandardScaler` and `np.std` both divide by `n` (population standard deviation), unlike `statistics.stdev` in Lesson 02.

For randomness, use the modern generator API, which is what every later lesson uses:

```python
generator = np.random.default_rng(42)
generator.normal(size=3)
```

Prefer `default_rng` over the legacy `np.random.seed` / `np.random.normal`: each generator carries its own state, so two parts of a program cannot accidentally interfere with each other's random streams.

## Exercises

1. **Easy.** Compute the min-max scaled version of `data`, `(x - min) / (max - min)` per column, in one line. Verify every column now spans exactly [0, 1].
2. **Medium.** Without loops, find for each row the index of its nearest *other* row using `pairwise_distances_numpy`. Hint: the diagonal is 0, so set it to `np.inf` with `np.fill_diagonal` before `argmin(axis=1)`.
3. **Hard.** Implement `pairwise_distances_dot(matrix)` using `|a - b|^2 = |a|^2 + |b|^2 - 2 a.b` with `matrix @ matrix.T` and broadcasting, so it never builds the `(n, n, d)` array. Compare speed and memory for n = 3000. Why might a few entries come out as tiny negative numbers before `sqrt`, and how do you guard against it?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| ndarray | "A NumPy array" | A fixed-type, contiguous block of memory plus a shape and strides that describe how to index it |
| dtype | "The type" | The element type shared by every value in the array, such as `float64` or `int64` |
| Vectorization | "No loops" | Expressing an operation on whole arrays so the loop runs in compiled code instead of the Python interpreter |
| Axis | "Direction" | One dimension of an array; reducing along axis k removes that dimension from the result's shape |
| Broadcasting | "Auto-expanding shapes" | NumPy's rule for combining arrays whose shapes differ, by stretching size-1 dimensions to match |
| Boolean mask | "Filter" | A boolean array used as an index to select the elements where it is `True` |
| Standardization | "z-scoring" | Subtracting each column's mean and dividing by its standard deviation, giving mean 0 and std 1 |

## Further Reading

- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) -- the official starting guide to arrays, shapes, and indexing
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) -- the exact broadcasting rules with diagrams
- [Python Data Science Handbook, chapter 2](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html) -- Jake VanderPlas's free, thorough introduction to NumPy
