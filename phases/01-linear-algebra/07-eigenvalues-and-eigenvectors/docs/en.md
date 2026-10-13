# Eigenvalues and Eigenvectors

> Most vectors get knocked off their line by a transformation. Eigenvectors are the ones that stay on it, only stretched. Find them and the matrix becomes simple.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 Lesson 03 (Matrices as Transformations), Phase 01 Lesson 06 (Determinant)
**Time:** ~75 minutes

## Learning Objectives

- Define eigenvectors geometrically (directions that stay on their own span) and eigenvalues (the stretch factor along them)
- Derive the characteristic equation `det(A - lambda I) = 0` and solve it for 2x2 matrices, including complex and repeated cases
- Implement power iteration and deflation to find eigenpairs of a symmetric matrix
- Connect eigenvectors to PCA (principal axes of a covariance matrix) and to the stationary distribution of a Markov chain

## The Problem

A dataset of 500 points forms a tilted, elongated cloud. Which direction is it stretched along, and by how much? That question is Principal Component Analysis (Phase 09), and the answer is the eigenvectors and eigenvalues of the data's covariance matrix.

The same machinery shows up everywhere: Google's PageRank is an eigenvector of the web's link matrix; the long-run behavior of a Markov chain is an eigenvector with eigenvalue 1; whether gradient descent converges (Phase 02) depends on the largest eigenvalue of the Hessian; spectral clustering uses eigenvectors of a graph. Eigen-analysis is how you find the "natural axes" of a linear transformation.

## The Concept

### Vectors that stay on their line

Apply `A = [[3, 1], [0, 2]]` to many vectors. Most get rotated off their original direction:

```
A [1, 1]  = [4, 2]       different direction
A [1, 0]  = [3, 0]       same direction, 3 times longer
A [-1, 1] = [-2, 2]      same direction, 2 times longer
```

`[1, 0]` and `[-1, 1]` are special: the transformation only stretches them. Every vector on the line through `[1, 0]` is stretched by 3; every vector on the line through `[-1, 1]` by 2. These are the **eigenvectors**, and 3 and 2 are the **eigenvalues**:

```
A v = lambda v
```

Look at the left panel of `code/out.png`: gray arrows go from points on the unit circle to their images. Only the red ones (along the dashed eigen-lines) point straight outward. The circle becomes an ellipse.

What an eigenvalue tells you:

```
lambda > 1        stretch along that line
0 < lambda < 1    shrink
lambda < 0        flip to the other side, then scale
lambda = 0        the line is squashed to the origin (matrix is singular)
complex           no real direction survives; there is rotation
```

### Finding them: the characteristic equation

`A v = lambda v` means `(A - lambda I) v = 0` for a non-zero `v`. A matrix that sends a non-zero vector to zero squashes space, so its determinant is zero (Lesson 06):

```
det(A - lambda I) = 0
```

For 2x2 this is a quadratic:

```
lambda^2 - trace(A) * lambda + det(A) = 0          trace = a + d, det = ad - bc

lambda = (trace +- sqrt(trace^2 - 4 det)) / 2
```

For `A = [[3, 1], [0, 2]]`: `lambda^2 - 5 lambda + 6 = 0`, so `lambda = 3` and `lambda = 2`. Two facts drop out for free: the eigenvalues sum to the trace and multiply to the determinant.

Then each eigenvector is any non-zero vector in the null space of `A - lambda I`. In 2-D, if a row of `A - lambda I` is `[p, q]`, then `[-q, p]` is perpendicular to it and therefore solves the equation.

### Special cases

- **Rotation by 90 degrees**: `trace = 0`, `det = 1`, `lambda = +-i`. Every direction gets rotated, so no real eigenvector exists. Complex eigenvalues signal rotation.
- **Shear** `[[1, 1], [0, 1]]`: `lambda = 1` twice, but only one eigen-direction (the x-axis). Not enough eigenvectors to form a basis; the matrix cannot be diagonalized.
- **Symmetric matrices** (`A = A^T`, such as covariance matrices): eigenvalues are always real, and eigenvectors for different eigenvalues are perpendicular. This is the *spectral theorem*, and it is why PCA works so cleanly.

### The eigenbasis makes everything simple

If you have `n` independent eigenvectors, use them as a basis (3Blue1Brown, chapter 13: change of basis). In that basis the matrix just scales each axis:

```
A = V D V^-1          V: eigenvectors as columns, D: eigenvalues on the diagonal
```

Read right to left: convert into eigen-coordinates (`V^-1`), scale each axis (`D`), convert back (`V`). Powers become trivial, `A^k = V D^k V^-1`, which is how you analyze what happens after many repeated applications (Markov chains, gradient descent iterations).

### Power iteration

Apply `A` over and over to a random vector, normalizing each time. Write the start as a combination of eigenvectors: each application multiplies the component along eigenvector `i` by `lambda_i`. The component with the largest `|lambda|` grows fastest and eventually dominates:

```
v, A v, A^2 v, ...   ->   direction of the dominant eigenvector
lambda ~= v^T A v    (Rayleigh quotient, for unit v)
```

For symmetric matrices, *deflation* finds the rest: subtract `lambda v v^T` to remove the found direction and run power iteration again.

### Covariance eigenvectors are the data's axes

The covariance matrix describes how the data spreads. Its eigenvectors are the perpendicular directions of maximum spread, and each eigenvalue is the variance along its direction. In the right panel of `code/out.png`, the long red arrow is the first principal component. That is PCA in one sentence.

## Build It

### Step 1: Scan for directions that stay on their span

```python
def stays_on_span_scan(matrix, n_angles=3600):
    found = []
    angles = np.linspace(0, math.pi, n_angles, endpoint=False)
    cross_values = []
    for angle in angles:
        direction = np.array([math.cos(angle), math.sin(angle)])
        image = matrix @ direction
        cross_values.append(direction[0] * image[1] - direction[1] * image[0])
    for index in range(n_angles - 1):
        if cross_values[index] == 0 or cross_values[index] * cross_values[index + 1] < 0:
            found.append(math.degrees(angles[index]))
    return found
```

The 2-D cross product `v1 * w2 - v2 * w1` (Lesson 01's signed area) is zero exactly when `A v` is parallel to `v`. Brute force finds 0 and 135 degrees, which are `[1, 0]` and `[-1, 1]`.

### Step 2: The characteristic equation

```python
def eigenvalues_2x2(matrix):
    trace = matrix[0, 0] + matrix[1, 1]
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    discriminant = trace * trace - 4 * determinant
    root = cmath.sqrt(discriminant)
    values = [(trace + root) / 2, (trace - root) / 2]
    if discriminant >= 0:
        return [value.real for value in values]
    return values


def eigenvector_2x2(matrix, eigenvalue):
    shifted = matrix - eigenvalue * np.eye(2)
    row = shifted[0] if np.abs(shifted[0]).sum() > np.abs(shifted[1]).sum() else shifted[1]
    if np.abs(row).sum() < 1e-12:
        return np.array([1.0, 0.0])
    vector = np.array([-row[1], row[0]])
    return vector / np.linalg.norm(vector)
```

```
  eigenvalues: [3, 2]
  lambda = 2.0: v = [-0.7071  0.7071], A v = [-1.4142  1.4142] = 2.0 * v
  rotation(90) eigenvalues: [0+1i, 0-1i]
```

### Step 3: Diagonalize

```python
eigenvector_matrix = np.column_stack([eigenvector_2x2(matrix, value) for value in values])
diagonal = np.linalg.inv(eigenvector_matrix) @ matrix @ eigenvector_matrix
```

`V^-1 A V` comes out as `[[3, 0], [0, 2]]`: in the eigenbasis, `A` is pure scaling.

### Step 4: Power iteration and deflation

```python
def power_iteration(matrix, n_steps=200, seed=0):
    vector = np.random.default_rng(seed).normal(size=matrix.shape[0])
    vector = vector / np.linalg.norm(vector)
    for _ in range(n_steps):
        image = matrix @ vector
        vector = image / np.linalg.norm(image)
    eigenvalue = vector @ matrix @ vector
    return eigenvalue, vector


def symmetric_eigendecomposition(matrix, n_steps=500):
    remaining = matrix.astype(float).copy()
    values, vectors = [], []
    for index in range(matrix.shape[0]):
        eigenvalue, eigenvector = power_iteration(remaining, n_steps=n_steps, seed=index)
        values.append(eigenvalue)
        vectors.append(eigenvector)
        remaining = remaining - eigenvalue * np.outer(eigenvector, eigenvector)
    return np.array(values), np.column_stack(vectors)
```

Convergence speed depends on the ratio `|lambda_2 / lambda_1|`: the smaller it is, the faster the dominant direction takes over. This simple version assumes the eigenvalues differ in magnitude and are positive (true for covariance matrices with distinct variances).

### Step 5: Principal axes of a data cloud

```python
stretched = generator.normal(size=(500, 2)) * [3.0, 0.7]
data = stretched @ rotation(30).T
covariance = np.cov(data, rowvar=False)
covariance_values, covariance_vectors = symmetric_eigendecomposition(covariance)
```

```
  eigenvalues (variances along the axes): [8.862 0.467]
  main axis angle: 32.1 deg (data was rotated by 30 deg)
  eigenvectors are perpendicular: dot = -1.67e-16
```

The data was generated with standard deviations 3 and 0.7 (variances 9 and 0.49), rotated by 30 degrees. The eigen-decomposition recovers all of it from the sample, up to sampling noise.

### Step 6: Markov chain stationary distribution

```python
transition = np.array([[0.9, 0.5, 0.2], [0.075, 0.4, 0.3], [0.025, 0.1, 0.5]])
_, stationary = power_iteration(transition, n_steps=500)
stationary = stationary / stationary.sum()
```

Column `j` holds the probabilities of moving from weather state `j` to each state. Running the chain for many days is power iteration, and the long-run distribution is the eigenvector with eigenvalue 1: `[0.8, 0.133, 0.067]`, unchanged by one more step.

## Use It

```python
numpy_values, numpy_vectors = np.linalg.eig(matrix)
symmetric_values, symmetric_vectors = np.linalg.eigh(covariance)
```

- `np.linalg.eig` handles any square matrix and returns complex values when needed (`rotation(90)` gives `+-i`). Eigenvectors are the *columns* of the second output.
- `np.linalg.eigh` is for symmetric matrices: faster, guaranteed real output, orthonormal eigenvectors, and eigenvalues sorted in *ascending* order. Use it for covariance matrices, and reverse the order to get principal components first.
- Eigenvectors are only defined up to scale (and sign): `v` and `-v` are equally valid. Never compare eigenvectors from two libraries without normalizing the sign.

Production libraries do not use power iteration for full decompositions; they use the QR algorithm (repeated QR factorizations, Lesson 02). Power iteration still matters when you need only the top few eigenvectors of a huge sparse matrix (`scipy.sparse.linalg.eigsh`, PageRank).

## Exercises

1. **Easy.** Find the eigenvalues and eigenvectors of `[[2, 0], [0, 5]]` and of `[[0, 1], [1, 0]]` by thinking geometrically before computing. Check with `np.linalg.eig`.
2. **Medium.** Run `power_iteration` on `[[2, 0], [0, 1.9]]` and on `[[2, 0], [0, 0.5]]` for 1 to 50 steps, recording the angle between the iterate and the true eigenvector. Plot both curves on a log scale. How does convergence relate to `lambda_2 / lambda_1`?
3. **Hard.** Generate 3-D data with standard deviations `[4, 2, 0.3]`, rotate it by a random orthogonal matrix (from `np.linalg.qr` of a random matrix), and recover the three principal axes with `symmetric_eigendecomposition`. Project the data onto the top two eigenvectors and report what fraction of the total variance (sum of eigenvalues) you kept. This is PCA, which you will build properly in Phase 09.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Eigenvector | "A direction the matrix does not rotate" | A non-zero v with A v = lambda v: it stays on its own span under the transformation |
| Eigenvalue | "The stretch factor" | The scalar lambda by which A scales its eigenvector; can be negative or complex |
| Characteristic equation | "det(A - lambda I) = 0" | The polynomial whose roots are the eigenvalues |
| Trace | "Sum of the diagonal" | The sum of diagonal entries, equal to the sum of the eigenvalues |
| Diagonalization | "Eigendecomposition" | Writing A = V D V^-1 with eigenvectors in V and eigenvalues on the diagonal of D |
| Spectral theorem | "Symmetric matrices are nice" | A real symmetric matrix has real eigenvalues and an orthonormal basis of eigenvectors |
| Power iteration | "Multiply until it converges" | Repeatedly applying A and normalizing, which converges to the dominant eigenvector |
| Principal axis | "Principal component" | An eigenvector of the covariance matrix; its eigenvalue is the variance of the data along it |

## Further Reading

- [3Blue1Brown, Eigenvectors and eigenvalues (Essence of Linear Algebra, chapter 14)](https://www.3blue1brown.com/lessons/eigenvalues) -- the "vectors that stay on their span" picture used throughout this lesson
- [3Blue1Brown, Change of basis (chapter 13)](https://www.3blue1brown.com/lessons/change-of-basis) -- why `V^-1 A V` is "the same transformation in eigen-coordinates"
- [Gilbert Strang, MIT 18.06 Lectures 21-22](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- eigenvalues, diagonalization, and powers of a matrix
- [StatQuest: Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) -- how eigenvectors of the covariance become principal components
