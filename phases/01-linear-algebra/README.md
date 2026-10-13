# Phase 01: Linear Algebra

> Every dataset is a matrix and every model is a transformation. Learn to see what matrices do to space, and the rest of machine learning becomes geometry.

This phase builds linear algebra from the geometric point of view popularized by 3Blue1Brown's *Essence of Linear Algebra*: vectors as arrows, matrices as transformations of a grid, determinants as area scaling, eigenvectors as the directions a transformation leaves on their line. Each lesson implements the operation from scratch in NumPy (elimination, Gram-Schmidt, power iteration, pairwise distances), draws what it does, and then checks it against `numpy.linalg`, SciPy, or scikit-learn. Everything here is used directly later: projections and linear systems in regression (Phase 05), distances in k-NN and clustering (Phases 06 and 08), eigenvectors in PCA (Phase 09).

**Prerequisites:** Phase 00 (Python Foundations), especially Lesson 05 (NumPy) and Lesson 07 (Plotting with Matplotlib).

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Vectors](01-vectors/) | Vector addition, scaling, linear combinations, span tests, and coordinates in a new basis, with a plot of the tilted grid |
| 02 | [Dot Product and Projections](02-dot-product-and-projections/) | Dot product, projection, cosine similarity between documents, and Gram-Schmidt, checked against `np.linalg.qr` |
| 03 | [Matrices as Transformations](03-matrices-as-transformations/) | Matrices built from where the basis vectors land, applied to a grid: rotation, shear, reflection, projection |
| 04 | [Matrix Multiplication](04-matrix-multiplication/) | Triple-loop, column, and outer-product matrix multiplication, composition, and a timing race against `@` |
| 05 | [Linear Systems and the Inverse](05-linear-systems-and-inverse/) | Gaussian elimination with pivoting, Gauss-Jordan inverse, rank, and conditioning, checked against `np.linalg.solve` |
| 06 | [Determinant](06-determinant/) | Determinant as area scaling, by cofactors and by pivots, plus `slogdet` for underflow-safe log-densities |
| 07 | [Eigenvalues and Eigenvectors](07-eigenvalues-and-eigenvectors/) | Characteristic equation, power iteration with deflation, principal axes of a data cloud, a Markov chain's steady state |
| 08 | [Norms and Distances](08-norms-and-distances/) | Lp norms and unit balls, Mahalanobis distance, fast pairwise distances, and the curse of dimensionality |

Run the first lesson:

```
uv run python phases/01-linear-algebra/01-vectors/code/main.py
```
