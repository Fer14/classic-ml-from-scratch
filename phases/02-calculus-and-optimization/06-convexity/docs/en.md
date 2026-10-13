# Convexity

> On a convex loss, downhill always leads to the bottom. That single property is why classic ML is so reliable.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra: Eigenvalues and Eigenvectors), Phase 02 Lessons 01-05
**Time:** ~50 minutes

## Learning Objectives

- State the chord definition of a convex function and test it numerically on candidate functions
- Use the second derivative in 1D and Hessian eigenvalues in many dimensions to certify convexity
- Show that MSE and logistic loss are convex, and explain the difference between convex and strictly convex
- Demonstrate that on a non-convex function gradient descent's answer depends on the starting point, and use a global heuristic to escape

## The Problem

Gradient descent stops where the gradient is zero. But a zero gradient only means "flat here". It could be the lowest point of the whole surface, a shallow dip halfway down a slope, or the top of a hill. If you train a model twice from different random weights and get two different answers, which one is right? Should you run it 100 times to be safe?

For most models in this course the answer is reassuring: their losses are convex, every local minimum is a global minimum, and any reasonable optimizer from any start reaches the same loss. You need to know which models enjoy this guarantee (linear regression, logistic regression, ridge, lasso, linear SVMs) and which do not (k-means, Gaussian mixtures, neural networks), because it changes how you train and debug them.

## The Concept

### The chord definition

A function `f` is convex if the straight line segment (the chord) between any two points on its graph never dips below the graph:

```
f(t a + (1 - t) b) <= t f(a) + (1 - t) f(b)     for all a, b and t in [0, 1]
```

```
  convex                        non-convex
  \                  /          \      __      /
   \      chord     /            \    /  \    /
    \  ----------- /              \  / ---\--/----  <- chord dips below
     \_         _/                 \/      \/
       \_______/
```

The left-hand side is the function at a blend of `a` and `b`. The right-hand side is the same blend of the function values. Convexity says "the function of an average is at most the average of the function". This is also Jensen's inequality, which you will meet again in Phase 03.

### Calculus tests

If `f` is twice differentiable:

- **1D:** `f` is convex if and only if `f''(x) >= 0` everywhere. The curve bends upward at every point.
- **Many dimensions:** `f` is convex if and only if the Hessian `H(x)` is positive semidefinite everywhere, meaning all its eigenvalues are `>= 0`. In every direction, the curvature is non-negative.
- If all eigenvalues are strictly positive everywhere, `f` is **strictly convex** and has at most one minimizer.

For `f(x) = x^4 - 3x^2 + x`, `f''(x) = 12x^2 - 6`, which is negative on `(-0.71, 0.71)`. Not convex.

### Why convexity matters

For a convex `f`:

1. Every local minimum is a global minimum. If some lower point existed, the chord from your local minimum to it would lie below the function near you, contradicting "local minimum".
2. Any point with zero gradient is a global minimizer.
3. Gradient descent with a suitable learning rate converges to the global minimum value from any starting point.

If `f` is convex but not strictly convex, the minimum value is unique but the minimizer may not be: a whole line or plane of weights can achieve it. This happens in linear regression when two features are perfectly collinear: you can shift weight between them without changing predictions.

### Which ML losses are convex?

```
convex in the weights                     non-convex
---------------------                     ----------
MSE for linear regression                 k-means objective (Phase 08)
log loss for logistic regression          Gaussian mixture likelihood (Phase 08)
ridge, lasso, elastic net                 neural networks
hinge loss for linear SVMs                t-SNE objective (Phase 09)
```

The MSE Hessian is `(2/n) X^T X`. For any vector `v`, `v^T X^T X v = |X v|^2 >= 0`, so it is always positive semidefinite.

The logistic loss Hessian is `(1/n) X^T S X` with `S = diag(p_i (1 - p_i))`. Every `p_i (1 - p_i)` is positive, so again `v^T X^T S X v >= 0`.

Building blocks that preserve convexity: non-negative weighted sums of convex functions, composition of a convex function with an affine map (`f(Xw + b)`), and pointwise maximum of convex functions. That is why adding an L2 or L1 penalty to a convex loss keeps it convex.

### Non-convex in practice

When the loss is not convex, local optimizers find a local minimum that depends on where they start. Common responses:

- **Multiple restarts:** run from several random starts and keep the best (k-means `n_init`).
- **Global heuristics:** random jumps plus local optimization, as in SciPy's `basinhopping`.
- **Accept it:** for neural networks, many local minima are about equally good, and SGD noise helps avoid bad ones.

## Build It

### Step 1: A numerical chord test

```python
def chord_test(f, low, high, dimension, trials=20_000, seed=0):
    rng = np.random.default_rng(seed)
    worst_gap = np.inf
    for _ in range(trials):
        a = rng.uniform(low, high, size=dimension)
        b = rng.uniform(low, high, size=dimension)
        t = rng.uniform()
        gap = t * f(a) + (1 - t) * f(b) - f(t * a + (1 - t) * b)
        worst_gap = min(worst_gap, gap)
    return worst_gap
```

```
sum of squares   smallest chord gap = +2.53e-05  -> convex (no violation found)
sum of |x|       smallest chord gap = -8.88e-16  -> convex (no violation found)
log-sum-exp      smallest chord gap = +3.19e-09  -> convex (no violation found)
x^2 + 3 sin(2x)  smallest chord gap = -7.46e+00  -> NOT convex
```

The `-8.88e-16` for `|x|` is floating-point rounding (it is exactly zero when `a` and `b` have the same signs). A random test can prove non-convexity by finding one violation, but it can only suggest convexity. For proof you need the calculus tests or the building-block rules.

### Step 2: Second derivative of the double well

```python
grid = np.linspace(-2, 2, 9)
second_derivative = 12 * grid**2 - 6
```

The script prints `curves DOWN` at `x = -0.5, 0, 0.5`. Any region of negative curvature breaks convexity.

### Step 3: The start point decides the answer

```python
def descend(gradient_fn, start, learning_rate=0.01, steps=2000):
    x = start
    for _ in range(steps):
        x = x - learning_rate * gradient_fn(x)
    return x
```

```
start -2.0 -> x = -1.3008, f = -3.5139
start -0.5 -> x = -1.3008, f = -3.5139
start +0.5 -> x = +1.1309, f = -1.0702
start +2.0 -> x = +1.1309, f = -1.0702
```

Starts on the right side end in the shallow local minimum at `f = -1.07`, missing the global minimum at `-3.51`. Gradient descent did its job perfectly; the function is the problem.

### Step 4: Certify ML losses with Hessian eigenvalues

```python
def logistic_hessian(weights, features):
    probabilities = sigmoid(features @ weights)
    return features.T @ (features * (probabilities * (1 - probabilities))[:, None]) / len(features)


mse_hessian = 2 * features.T @ features / len(features)
print(np.linalg.eigvalsh(mse_hessian))
```

The MSE Hessian eigenvalues are `[1.44, 1.61, 1.82, 2.54]`, all positive: strictly convex. The logistic Hessian at three random weight vectors has smallest eigenvalues `0.004, 0.006, 0.002`: small but positive. When a feature is duplicated, the smallest MSE eigenvalue becomes `2.2e-16`, numerically zero: convex, but with a flat valley of equally good solutions.

### Step 5: Same minimum from every start

Three gradient descent runs on the MSE from random starts with scale 10 all print `[0.9739, -2.0193, 0.5021, 3.0308]`. That is what convexity buys you.

## Use It

```python
from scipy.optimize import basinhopping, minimize

minimize(lambda v: double_well(v[0]), [-2.0], method="BFGS")
minimize(lambda v: double_well(v[0]), [2.0], method="BFGS")
basinhopping(lambda v: double_well(v[0]), [2.0], niter=100, stepsize=1.5, seed=0)
```

```
minimize (BFGS) from -2.0: x = -1.3008, f = -3.5139
minimize (BFGS) from +2.0: x = +1.1309, f = -1.0702
basinhopping from +2.0:  x = -1.3008, f = -3.5139
```

`minimize` is a local method: a better algorithm than plain gradient descent, but it still gets trapped on the right. `basinhopping` repeatedly perturbs the point by up to `stepsize`, runs a local minimization, and accepts or rejects the result with a Metropolis rule. With jumps large enough to cross the hump, it finds the global minimum.

In scikit-learn, the convex models (`LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `LinearSVC`) give the same answer every run, up to solver tolerance. Non-convex ones expose knobs for restarts: `KMeans(n_init=...)`, `GaussianMixture(n_init=...)`, `MLPClassifier(random_state=...)`.

## Exercises

1. **Easy.** Use `chord_test` on `f(x) = max(x1, x2)`, `f(x) = x1 * x2`, and `f(x) = exp(x1) + x2^2`. Which ones are convex? Confirm one of your answers with a calculus argument.
2. **Medium.** Show numerically that the squared error of a single sigmoid neuron, `L(w) = mean((sigmoid(w x) - y)^2)`, is not convex in `w`: plot `L(w)` for `w` in `[-10, 10]` on a small dataset and find a chord that dips below. Contrast with log loss on the same data.
3. **Hard.** Implement k-means with `k = 3` on a dataset of 3 Gaussian blobs and run it from 20 random initializations. Record the final objective (sum of squared distances to centers). How many distinct values do you get? Which property of the objective explains this?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Convex function | "Bowl-shaped" | A function whose chords never lie below its graph: `f(ta + (1-t)b) <= t f(a) + (1-t) f(b)` |
| Strictly convex | "One true minimum" | The chord inequality is strict for `a != b`; at most one minimizer exists |
| Positive semidefinite | "All curvatures non-negative" | A symmetric matrix with all eigenvalues `>= 0`, equivalently `v^T H v >= 0` for every `v` |
| Local minimum | "A dip" | A point lower than everything in a small neighborhood around it |
| Global minimum | "The lowest point" | A point lower than or equal to every other point in the domain |
| Jensen's inequality | "Average of f vs f of average" | For convex `f`, `f(E[X]) <= E[f(X)]`; the probabilistic form of the chord definition |
| Basin hopping | "Random restarts, but smarter" | A global heuristic alternating random jumps with local minimization |

## Further Reading

- [Boyd and Vandenberghe, Convex Optimization, chapters 2-3](https://web.stanford.edu/~boyd/cvxbook/) -- the standard free reference on convex sets, convex functions and operations that preserve convexity
- [Stanford CS229, Convex Optimization Overview](https://cs229.stanford.edu/section/cs229-cvxopt.pdf) -- a short section handout tailored to ML students
- [Mathematics for Machine Learning, section 7.3](https://mml-book.github.io/) -- convex optimization in the context of ML models
- [SciPy `basinhopping` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) -- the global optimizer used in Use It
