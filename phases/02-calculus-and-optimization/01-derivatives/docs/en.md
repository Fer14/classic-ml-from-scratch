# Derivatives

> A derivative is the slope of a function at a single point. It tells you which way is downhill and how steep it is.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 00 (Python Foundations), Phase 01 (Linear Algebra)
**Time:** ~45 minutes

## Learning Objectives

- Define the derivative as the limit of secant slopes and compute it numerically with forward and central differences
- Apply the power, exponential, logarithm and sigmoid derivative rules and verify them against numerical estimates
- Explain why central differences are more accurate than forward differences and why a tiny step size eventually makes things worse
- Use the tangent line as a local linear approximation and locate critical points where the derivative is zero

## The Problem

Every model you will train in this course has a knob (a weight) and a number that says how wrong the model is (a loss). Training means turning the knob so the loss goes down. With one knob you could try values by hand. With ten thousand knobs you need a rule that tells you, for each knob, "turn it this way, and the loss will drop this fast".

That rule is the derivative. If you know that nudging a weight by a tiny amount raises the loss by 4.75 times that amount, you know to move the weight the other way. Gradient descent, backpropagation, and every optimizer in scikit-learn are built on this single idea.

Before you can trust a training loop, you need to be able to compute derivatives, check them numerically, and understand what they say about the shape of a function.

## The Concept

### Slope of a secant, slope of a tangent

Pick two points on the curve of `f`: one at `x`, one a step `h` to the right. The line through them is a secant, and its slope is rise over run:

```
secant slope = (f(x + h) - f(x)) / h
```

Shrink `h` toward zero. The second point slides toward the first, and the secant rotates into the tangent line. The derivative is the slope of that tangent:

```
f'(x) = lim (h -> 0)  (f(x + h) - f(x)) / h
```

```
   f(x)
    |                 * f(x+h)
    |               / 
    |             /   secant (big h)
    |          _/ 
    |       _-*  <- tangent touches here, slope = f'(x)
    |   _--   x
    +-------------------------- x
```

For `f(x) = x^3 - 2x + 1` at `x = 1.5`, the secant slopes for `h = 1, 0.1, 0.01, 0.001` are `10.25, 5.21, 4.795, 4.7545`. They converge to `4.75`, which is exactly `3(1.5)^2 - 2`.

### What the derivative tells you

- `f'(x) > 0`: the function is increasing. Moving right raises `f`.
- `f'(x) < 0`: the function is decreasing. Moving right lowers `f`.
- `f'(x) = 0`: the tangent is flat. You are at a minimum, a maximum, or a flat inflection point. These are critical points.
- `|f'(x)|` large: the function is steep. Small changes in `x` cause big changes in `f`.

In ML terms: if `f` is the loss and `x` is a weight, the sign of the derivative tells you which way to move the weight, and the magnitude tells you how sensitive the loss is to it.

### The rules you will actually use

You rarely compute limits by hand. A short table covers almost everything in classic ML:

| Function | Derivative |
|----------|------------|
| `c` (constant) | `0` |
| `x^n` | `n * x^(n-1)` |
| `e^x` | `e^x` |
| `ln(x)` | `1 / x` |
| `sin(x)` | `cos(x)` |
| `sigmoid(x) = 1 / (1 + e^-x)` | `sigmoid(x) * (1 - sigmoid(x))` |
| `a*f(x) + b*g(x)` | `a*f'(x) + b*g'(x)` (linearity) |
| `f(x) * g(x)` | `f'(x) g(x) + f(x) g'(x)` (product rule) |

Composition (`f(g(x))`) has its own rule, the chain rule, which gets a full lesson later in this phase.

### Numerical derivatives

A computer cannot take a limit, but it can use a small `h`. Two common formulas:

```
forward difference:  (f(x + h) - f(x)) / h          error ~ O(h)
central difference:  (f(x + h) - f(x - h)) / (2h)   error ~ O(h^2)
```

Why is central better? Expand with a Taylor series: `f(x + h) = f(x) + h f'(x) + (h^2/2) f''(x) + ...`. The forward formula keeps the `h f''(x)/2` term as error. In the central formula the `h^2` terms cancel, and the leading error is proportional to `h^2`. With `h = 1e-3`, forward error is around `1e-3` and central error is around `1e-6`.

### The step size trap

You might think: smaller `h`, better answer. Only up to a point. Floating-point numbers have about 16 significant digits. When `h` is tiny, `f(x + h)` and `f(x)` agree in almost all digits, and subtracting them throws away precision (catastrophic cancellation). The error curve is a V:

```
error
  |\                         /
  | \   truncation error    /  rounding error
  |  \                     /
  |   \___________________/
  +-------------------------------- h
  1e-1        1e-5       1e-11
```

A good default is `h ~ 1e-5` for central differences in double precision.

### The tangent line as an approximation

Near `x0`, a smooth function looks like its tangent:

```
f(x0 + dx) ~ f(x0) + f'(x0) * dx
```

This is a first-order Taylor approximation. It is excellent for small `dx` and poor for large `dx`. Gradient descent relies on exactly this: it trusts the linear approximation for a small step, takes the step, then recomputes. That is why step size matters, which you will see in the learning rate lesson.

## Build It

### Step 1: Secant slopes shrinking to the tangent

```python
def cubic(x):
    return x**3 - 2 * x + 1


def cubic_derivative(x):
    return 3 * x**2 - 2


def secant_slopes(f, x0, step_sizes):
    return [(h, (f(x0 + h) - f(x0)) / h) for h in step_sizes]


for h, slope in secant_slopes(cubic, 1.5, [1.0, 0.1, 0.01, 0.001, 0.0001]):
    print(f"h = {h:<8} secant slope = {slope:.6f}")
```

The printed slopes approach `4.75` from above. Each 10x reduction in `h` cuts the error by roughly 10x, which is the `O(h)` behavior of a one-sided difference.

### Step 2: Numerical vs analytic derivatives

```python
def forward_difference(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


functions = {
    "x^3 - 2x + 1": (cubic, cubic_derivative),
    "sin(x)": (np.sin, np.cos),
    "exp(x)": (np.exp, np.exp),
    "log(x)": (np.log, lambda x: 1 / x),
    "sigmoid(x)": (sigmoid, sigmoid_derivative),
}
for name, (f, df) in functions.items():
    print(name, df(0.8), forward_difference(f, 0.8), central_difference(f, 0.8))
```

Central differences match the analytic values to 8 decimal places. Forward differences are off in the 5th or 6th place. This comparison is called a gradient check, and it is the standard way to catch bugs in hand-written derivatives.

### Step 3: Error as a function of h

```python
exact = cubic_derivative(1.5)
for exponent in range(1, 13, 2):
    h = 10.0**-exponent
    fwd_err = abs(forward_difference(cubic, 1.5, h) - exact)
    ctr_err = abs(central_difference(cubic, 1.5, h) - exact)
    print(f"{h:8.0e} {fwd_err:15.3e} {ctr_err:15.3e}")
```

Output from the script:

```
       h   forward error   central error
   1e-01       4.600e-01       1.000e-02
   1e-03       4.501e-03       1.000e-06
   1e-05       4.500e-05       1.310e-10
   1e-07       4.507e-07       2.218e-09
   1e-09       3.930e-07       3.930e-07
   1e-11       2.181e-05       3.930e-07
```

The central error is best near `h = 1e-5` and gets worse below it. That is the V-shaped curve from the concept section, measured.

### Step 4: The tangent line as a local model

```python
def tangent_line(f, df, x0):
    slope = df(x0)
    intercept = f(x0) - slope * x0
    return slope, intercept


slope, intercept = tangent_line(cubic, cubic_derivative, 1.5)
for dx in [0.01, 0.1, 0.5, 1.0]:
    true_value = cubic(1.5 + dx)
    approx_value = slope * (1.5 + dx) + intercept
    print(dx, true_value, approx_value, abs(true_value - approx_value))
```

The gap grows roughly with `dx^2`: `0.0005` at `dx = 0.01`, `0.046` at `0.1`, `1.25` at `0.5`. The linear model is trustworthy only in a small neighborhood.

### Step 5: Critical points

Setting `f'(x) = 3x^2 - 2 = 0` gives `x = +-sqrt(2/3) ~ +-0.8165`. The script evaluates `f` there: `2.0887` at the negative root (a local maximum) and `-0.0887` at the positive root (a local minimum). Solving `f'(x) = 0` analytically only works for simple functions; for real models you will walk downhill numerically instead.

The script also saves `derivatives.png`, showing two secants rotating toward the tangent at `x = 1.5`.

## Use It

NumPy and SciPy already know how to differentiate:

```python
from scipy.differentiate import derivative

grid = np.linspace(-2, 2, 401)
numpy_slopes = np.gradient(cubic(grid), grid)

result = derivative(cubic, 1.5)
print(result.df)

poly = np.polynomial.Polynomial([1, -2, 0, 1])
print(poly.deriv().coef)
```

- `np.gradient` applies central differences to sampled data on a grid. Use it when you only have arrays of values, not a function. Its interior error here is `1e-4` because the grid spacing is `0.01`.
- `scipy.differentiate.derivative` picks step sizes adaptively and extrapolates, giving `4.7500000000` to full precision.
- `np.polynomial.Polynomial.deriv` differentiates polynomials symbolically: coefficients `[1, -2, 0, 1]` (constant first) become `[-2, 0, 3]`, which is `3x^2 - 2`.

In ML libraries you will almost never call these. scikit-learn's solvers have the derivatives of each loss written out by hand, and deep learning frameworks use automatic differentiation. Numerical derivatives remain your best debugging tool for checking that a hand-written gradient is correct.

## Exercises

1. **Easy.** Add `tanh(x)` with derivative `1 - tanh(x)^2` and `x * ln(x)` with derivative `ln(x) + 1` to the comparison table. Confirm central differences agree to at least 8 decimals.
2. **Medium.** Implement a second derivative with `(f(x + h) - 2f(x) + f(x - h)) / h^2`. Use it to classify the two critical points of the cubic as a minimum or maximum. What step size works best, and why is it larger than for the first derivative?
3. **Hard.** Implement Richardson extrapolation: combine central differences at `h` and `h/2` as `(4 D(h/2) - D(h)) / 3`. Measure its error over the same range of `h` and plot all three error curves on log-log axes. What slope does each curve have on the left side of the V?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Derivative | "The slope" | The limit of `(f(x+h) - f(x)) / h` as `h -> 0`: the instantaneous rate of change of `f` at `x` |
| Secant line | "A line through two points" | The line through `(x, f(x))` and `(x+h, f(x+h))`; its slope approximates the derivative |
| Tangent line | "The line that just touches" | The line through `(x0, f(x0))` with slope `f'(x0)`; the best linear approximation near `x0` |
| Central difference | "The better numerical derivative" | `(f(x+h) - f(x-h)) / 2h`, with error proportional to `h^2` instead of `h` |
| Gradient check | "Check your math with numbers" | Comparing an analytic derivative to a numerical one to catch implementation bugs |
| Critical point | "Where it is flat" | A point where `f'(x) = 0`: a minimum, a maximum, or a flat inflection |
| Catastrophic cancellation | "Floating-point noise" | Loss of significant digits when subtracting two nearly equal numbers, which ruins very small `h` |

## Further Reading

- [3Blue1Brown, Essence of Calculus (chapters 1-3)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) -- visual intuition for derivatives as slopes and as ratios of tiny nudges
- [Mathematics for Machine Learning, Deisenroth, Faisal, Ong (chapter 5)](https://mml-book.github.io/) -- free PDF covering differentiation, Taylor series and gradients with ML in mind
- [CS231n notes: Gradient checking](https://cs231n.github.io/neural-networks-3/#gradcheck) -- practical advice on centered differences, step sizes and relative error
- [SciPy `scipy.differentiate` documentation](https://docs.scipy.org/doc/scipy/reference/differentiate.html) -- the library functions used in Use It
