# Partial Derivatives and Gradients

> A gradient is a list of slopes, one per input. It is an arrow that points straight uphill.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra: Vectors, Dot Product), Phase 02 Lesson 01 (Derivatives)
**Time:** ~60 minutes

## Learning Objectives

- Compute partial derivatives of multivariable functions analytically and with central differences
- Assemble partials into the gradient vector and explain why it points in the direction of steepest ascent
- Use the dot product to compute directional derivatives and show the gradient is perpendicular to level curves
- Derive and verify the vectorized MSE gradient `2/n * X^T (Xw - y)` used by every linear model

## The Problem

A real model does not have one weight. Linear regression on 3 features has 3 weights plus a bias. A small neural network has thousands. The loss is a function of all of them at once: `L(w1, w2, ..., wd)`.

The derivative from the previous lesson answers "how does the loss change if I nudge one number?". You need the same answer for every weight simultaneously, packaged in a form you can use to update all weights in one line of NumPy. That package is the gradient.

## The Concept

### Partial derivatives

A partial derivative asks: if I change only `x` and freeze every other input, how fast does `f` change? You treat the other variables as constants and differentiate as usual.

For `f(x, y) = x^2 + 3y^2 + xy`:

```
df/dx = 2x + y        (3y^2 is a constant w.r.t. x, xy becomes y)
df/dy = 6y + x        (x^2 is a constant w.r.t. y, xy becomes x)
```

At the point `(1, 2)`: `df/dx = 4`, `df/dy = 13`. The function is over three times more sensitive to `y` than to `x` there.

Numerically, a partial derivative is a central difference along one coordinate axis:

```
df/dx_i ~ (f(x + h e_i) - f(x - h e_i)) / (2h)
```

where `e_i` is the unit vector with a 1 in position `i`.

### The gradient

Stack all partial derivatives into a vector:

```
grad f(x) = [ df/dx_1, df/dx_2, ..., df/dx_d ]
```

For the bowl above, `grad f(1, 2) = [4, 13]`. The gradient lives in the same space as the input: it has one entry per weight.

### Why the gradient points uphill

How fast does `f` change if you move in some unit direction `u`? The first-order Taylor expansion in several variables is:

```
f(x + t u) ~ f(x) + t * (grad f(x) . u)
```

So the rate of change along `u`, the directional derivative, is a dot product:

```
D_u f(x) = grad f(x) . u = |grad f| * |u| * cos(theta) = |grad f| cos(theta)
```

This is maximized when `theta = 0`, when `u` points along the gradient. So:

- The gradient direction is the steepest ascent; its length `|grad f|` is that steepest rate.
- The negative gradient is the steepest descent. This is the direction gradient descent walks.
- Any direction perpendicular to the gradient (`theta = 90 deg`) has rate zero: moving that way keeps `f` constant to first order.

```
                 grad f  (steepest uphill, rate = |grad f|)
                   ^
                   |
  level curve -----+------ level curve   (rate 0 along the curve)
                   |
                   v
                -grad f  (steepest downhill)
```

The last point means the gradient is always perpendicular to the level curves (contour lines) of `f`. If you have seen a topographic map, the steepest path down a hill crosses the contour lines at right angles.

### The gradient of the MSE loss

Linear regression predicts `y_hat = X w`, where `X` is `n x d` and `w` has `d` entries. The loss is:

```
L(w) = (1/n) * sum_i (x_i . w - y_i)^2 = (1/n) * |X w - y|^2
```

Differentiating with respect to `w_j`: each squared term contributes `2 (x_i . w - y_i) * x_ij`. Collect all `j` at once and you get a matrix-vector product:

```
grad L(w) = (2/n) * X^T (X w - y)
```

Read it as: compute residuals `r = Xw - y` (one per sample), then each weight's gradient is the average of residuals weighted by that weight's feature column. This one line is the core of linear regression by gradient descent, and you will reuse it in Phase 05.

## Build It

### Step 1: Partial derivatives by central differences

```python
def bowl(point):
    x, y = point
    return x**2 + 3 * y**2 + x * y


def partial_derivative(f, point, index, h=1e-5):
    step = np.zeros_like(point, dtype=float)
    step[index] = h
    return (f(point + step) - f(point - step)) / (2 * h)


point = np.array([1.0, 2.0])
print(partial_derivative(bowl, point, 0))
print(partial_derivative(bowl, point, 1))
```

The output is `4.000000` and `13.000000`, matching `2x + y` and `6y + x`.

### Step 2: The full gradient

```python
def numerical_gradient(f, point, h=1e-5):
    point = np.asarray(point, dtype=float)
    return np.array([partial_derivative(f, point, i, h) for i in range(point.size)])


def bowl_gradient(point):
    x, y = point
    return np.array([2 * x + y, 6 * y + x])


grad = numerical_gradient(bowl, point)
print(grad, bowl_gradient(point), np.linalg.norm(grad))
```

The numerical gradient costs `2d` function evaluations. That is fine for checking but far too slow for training a model with millions of weights, which is why analytic gradients matter.

### Step 3: Directional derivatives

```python
def directional_derivative(gradient, direction):
    unit = direction / np.linalg.norm(direction)
    return gradient @ unit


for angle in np.deg2rad(np.arange(0, 360, 45)):
    direction = np.array([np.cos(angle), np.sin(angle)])
    print(np.rad2deg(angle), directional_derivative(grad, direction))
```

Sweeping eight compass directions, the largest rate is `+13.0` at 90 degrees and the smallest is `-13.0` at 270 degrees. The true maximum is at `72.9` degrees, the angle of `[4, 13]`, with rate `|grad| = 13.6015`. No direction does better than the gradient.

### Step 4: Perpendicular to level curves

```python
tangent_direction = np.array([-grad[1], grad[0]])
small_step = 1e-4 * tangent_direction / np.linalg.norm(tangent_direction)
print(bowl(point), bowl(point + small_step), grad @ tangent_direction)
```

Rotating the gradient by 90 degrees gives a direction along the level curve. Moving `1e-4` that way changes `f` from `15.00000000` to `15.00000001`: a second-order effect only.

### Step 5: Vectorized MSE gradient with a gradient check

```python
def mse_loss(weights, features, targets):
    residuals = features @ weights - targets
    return np.mean(residuals**2)


def mse_gradient(weights, features, targets):
    residuals = features @ weights - targets
    return 2 * features.T @ residuals / len(targets)


rng = np.random.default_rng(0)
features = rng.normal(size=(200, 3))
true_weights = np.array([2.0, -1.0, 0.5])
targets = features @ true_weights + rng.normal(scale=0.1, size=200)

weights = np.zeros(3)
analytic = mse_gradient(weights, features, targets)
numerical = numerical_gradient(lambda w: mse_loss(w, features, targets), weights)
relative_error = np.linalg.norm(analytic - numerical) / np.linalg.norm(analytic + numerical)
```

At `w = 0` the gradient is `[-4.05, 2.10, -0.67]`. Its signs are the opposite of the true weights `[2, -1, 0.5]`, which is exactly what you want: walking along the negative gradient moves each weight toward its true value. The relative error between analytic and numerical is `5.8e-12`, so the formula is correct. At the true weights, the gradient is nearly zero (`~0.02`), only nonzero because of noise.

The script also saves `gradient_field.png`: contour lines of the bowl with negative gradient arrows crossing them at right angles.

## Use It

SciPy ships the same tools for checking gradients:

```python
from scipy.optimize import approx_fprime, check_grad

print(approx_fprime(point, bowl, 1e-6))
print(check_grad(bowl, bowl_gradient, point))
print(check_grad(mse_loss, mse_gradient, weights, features, targets))
```

- `approx_fprime` returns a forward-difference gradient (`[4.000001, 13.000003]`; note the forward-difference bias in the last digits).
- `check_grad` returns the norm of the difference between your gradient function and a finite-difference estimate. Values around `1e-7` or smaller mean your gradient is right. For the MSE gradient it reports `4.9e-08`.

When you write a custom loss for scikit-learn's `SGDRegressor`, for `scipy.optimize.minimize` with `jac=`, or for any from-scratch model in this course, run `check_grad` once before trusting the training loop.

## Exercises

1. **Easy.** Compute the gradient of `f(x, y, z) = x*y*z + z^2` by hand and verify it with `numerical_gradient` at `(1, 2, 3)`.
2. **Medium.** Add an L2 penalty to the loss: `L(w) = MSE(w) + lam * |w|^2`. Derive its gradient, implement it, and verify it with `check_grad` for `lam = 0.1`.
3. **Hard.** Derive the gradient of the logistic loss `L(w) = -(1/n) sum [y log s(Xw) + (1-y) log(1 - s(Xw))]` where `s` is the sigmoid. Show that it simplifies to `(1/n) X^T (s(Xw) - y)`, implement it, and verify it numerically on random data with binary labels.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Partial derivative | "Derivative in one variable" | The rate of change of `f` along one input axis with all other inputs held fixed |
| Gradient | "The slope vector" | The vector of all partial derivatives; it points in the direction of steepest ascent |
| Directional derivative | "Slope in some direction" | The rate of change along a unit vector `u`, equal to `grad f . u` |
| Steepest descent | "Go downhill fastest" | Moving along `-grad f`, the direction that decreases `f` fastest to first order |
| Level curve | "Contour line" | The set of points where `f` takes a fixed value; the gradient is perpendicular to it |
| Residual | "The error" | `x_i . w - y_i`, the difference between prediction and target for one sample |
| Relative error | "How close two gradients are" | `|a - b| / |a + b|`; below about `1e-7` means a gradient implementation is correct |

## Further Reading

- [Khan Academy, Gradient and directional derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) -- Grant Sanderson's multivariable calculus series with visual explanations of partials and gradients
- [Mathematics for Machine Learning, chapter 5 (Vector Calculus)](https://mml-book.github.io/) -- gradients of vector-valued functions and matrix derivatives used in ML
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) -- reference sheet of matrix derivative identities like `d/dw |Xw - y|^2`
- [SciPy `check_grad` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html) -- the gradient-checking utility used in Use It
