# The Learning Rate

> The learning rate is the one hyperparameter you will tune more than any other. Too small wastes time, too large destroys everything.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra: Eigenvalues and Eigenvectors), Phase 02 Lesson 04 (Gradient Descent)
**Time:** ~60 minutes

## Learning Objectives

- Identify the five regimes of a fixed learning rate (too slow, good, optimal, oscillating, diverging) from a loss trace
- Compute the stability limit `eta < 2 / L` from the largest eigenvalue of the Hessian and verify it experimentally
- Implement step, exponential and inverse-time learning rate schedules and explain when decay helps
- Implement backtracking line search with the Armijo condition so the optimizer picks its own step size

## The Problem

In the gradient descent lesson you used a learning rate of `0.1` and it worked. Why `0.1`? If you had picked `0.5` on a different dataset, the loss would have jumped to `1e25` within a few dozen steps. If you had picked `0.0001`, training would have taken 1000 times longer and you might have concluded the model was bad.

Every training run you do, in scikit-learn's `SGDClassifier`, in gradient boosting, in neural networks, has a learning rate. You need to know how to recognize a bad one from the loss curve, what sets the upper limit, and what tools exist so you do not have to guess.

## The Concept

### One dimension tells the whole story

Take `f(x) = (L/2) x^2`. Its gradient is `L x` and its curvature (second derivative) is `L`. One gradient step gives:

```
x_new = x - eta * L * x = (1 - eta L) * x
```

Everything depends on the factor `r = 1 - eta L`:

```
 r in (0, 1)    : x shrinks monotonically           (eta < 1/L)
 r = 0          : x hits 0 in one step               (eta = 1/L)
 r in (-1, 0)   : x flips sign each step but shrinks (1/L < eta < 2/L)
 r < -1         : x flips sign and grows: divergence (eta > 2/L)
```

With `L = 4` and starting at `x = 5`:

```
too small    eta = 0.02  factor = +0.92  x:  5.000  4.600  4.232  3.893 ...
good         eta = 0.15  factor = +0.40  x:  5.000  2.000  0.800  0.320 ...
optimal 1/L  eta = 0.25  factor = +0.00  x:  5.000  0.000  0.000  0.000 ...
oscillating  eta = 0.45  factor = -0.80  x:  5.000 -4.000  3.200 -2.560 ...
diverging    eta = 0.55  factor = -1.20  x:  5.000 -6.000  7.200 -8.640 ...
```

### Many dimensions: the largest curvature wins

For a quadratic loss in many weights, the curvature is a matrix, the Hessian `H`. For MSE on a design matrix `X`:

```
H = (2/n) X^T X
```

Along each eigenvector of `H`, gradient descent behaves like the 1D case with `L` replaced by that eigenvalue `lambda_i`. Every direction must have `|1 - eta lambda_i| < 1`, so the strictest constraint comes from the largest eigenvalue:

```
eta < 2 / lambda_max
```

Meanwhile, the slowest direction shrinks by `1 - eta lambda_min` per step. With the best fixed `eta`, the number of steps scales with the condition number `kappa = lambda_max / lambda_min`. This is the precise version of "elongated bowls are slow".

### Reading loss curves

```
 loss
  |\                                 \  too small: still falling steadily at the end
  | \_____                            
  |       \____________               good: drops fast, then flattens
  |  /\/\/\/\/\/\/\/\/\/              oscillating: jagged, slow to settle
  |                  ___/             diverging: grows, often to inf or nan
  +------------------------ step
```

If the loss is `nan` or huge, the learning rate is too large (or your features are badly scaled). If the curve is smooth but barely moving, it is too small.

### Schedules

A schedule changes `eta` over time:

```
step decay:     eta_t = eta_0 * drop^(floor(t / every))
exponential:    eta_t = eta_0 * decay^t
inverse time:   eta_t = eta_0 / (1 + decay * t)
```

On a deterministic, smooth problem like full-batch linear regression, decay does not help: the gradient already shrinks to zero near the minimum, so a constant `eta` converges, and decaying only slows it. Schedules matter when the gradient is **noisy**, as in stochastic gradient descent (lesson 07). There, a constant `eta` makes the weights bounce around the minimum forever, and decaying `eta` averages the noise away.

### Line search

Instead of fixing `eta`, choose it at every step. Backtracking line search starts with a large trial step and halves it until the loss drops "enough":

```
Armijo condition:  L(w - eta g) <= L(w) - c * eta * |g|^2     (c ~ 1e-4)
```

This guarantees progress without knowing `L`. The price is extra loss evaluations per step. SciPy's BFGS and L-BFGS use more refined line searches of this kind.

## Build It

### Step 1: The five regimes

```python
def quadratic_trajectory(curvature, learning_rate, start=5.0, steps=12):
    positions = [start]
    for _ in range(steps):
        positions.append(positions[-1] - learning_rate * curvature * positions[-1])
    return np.array(positions)
```

The table in the concept section is the script's output.

### Step 2: Measure the stability limit on real regression data

```python
features, design, targets = make_regression_data()
hessian = 2 * design.T @ design / len(targets)
eigenvalues = np.linalg.eigvalsh(hessian)
largest, smallest = eigenvalues[-1], eigenvalues[0]
```

`make_regression_data` draws 3 features with standard deviations `1, 3, 0.5`, so the bowl is stretched. The eigenvalues are `[0.47, 1.68, 2.277, 17.061]`, giving `2/L = 0.1172` and condition number `36.3`. Now run gradient descent with `eta` as a multiple of `1/L`:

```python
def run_gradient_descent(design, targets, schedule, steps=300):
    weights = np.zeros(design.shape[1])
    losses = []
    for step in range(steps):
        losses.append(mse(weights, design, targets))
        if not np.isfinite(losses[-1]) or losses[-1] > 1e12:
            break
        weights = weights - schedule(step) * mse_gradient(weights, design, targets)
    return weights, np.array(losses)
```

```
eta = 0.1/L: final loss = 0.4923      too slow
eta = 0.5/L: final loss = 0.09115
eta = 1.0/L: final loss = 0.09065
eta = 1.9/L: final loss = 0.09064     just under the limit, still fine
eta = 2.1/L: loss = 1.003e+12 after 143 steps
```

The theory predicts the cliff to within a few percent: `1.9/L` converges, `2.1/L` explodes.

### Step 3: Schedules

```python
def step_decay(rate, drop=0.5, every=50):
    return lambda step: rate * drop ** (step // every)


def exponential_decay(rate, decay=0.99):
    return lambda step: rate * decay**step


def inverse_time_decay(rate, decay=0.05):
    return lambda step: rate / (1 + decay * step)
```

Each schedule is a function from step number to learning rate, so `run_gradient_descent` accepts any of them. Starting from `eta_0 = 1.9 / L`:

```
constant      step 100 = 0.090688, step 299 = 0.090645
step decay    step 100 = 0.091306, step 299 = 0.090701
exponential   step 100 = 0.092999, step 299 = 0.090729
inverse time  step 100 = 0.134118, step 299 = 0.096193
```

Constant wins here, as the concept section predicted for a noise-free gradient. Inverse time decays too aggressively and is still far from converged. The plot `learning_rate_schedules.png` shows the gap to the best loss on a log scale.

### Step 4: Backtracking line search

```python
def backtracking_line_search(design, targets, steps=300, initial_rate=1.0, shrink=0.5, armijo=1e-4):
    weights = np.zeros(design.shape[1])
    losses, rates = [], []
    for _ in range(steps):
        loss = mse(weights, design, targets)
        gradient = mse_gradient(weights, design, targets)
        losses.append(loss)
        rate = initial_rate
        while mse(weights - rate * gradient, design, targets) > loss - armijo * rate * gradient @ gradient:
            rate *= shrink
        rates.append(rate)
        weights = weights - rate * gradient
    return weights, np.array(losses), np.array(rates)
```

It reaches the same `0.090645` with no tuning. The chosen rates `[0.125, 0.125, 0.125, 0.0625, 0.5, 0.0625, 0.5, ...]` are interesting: sometimes it takes a step far above `2/L`. That is safe for a single step when the gradient happens to point mostly along a flat direction, which is exactly where a big step pays off.

## Use It

scikit-learn's SGD estimators expose the learning rate directly:

```python
from sklearn.linear_model import SGDRegressor

for strategy, eta0 in [("constant", 0.001), ("constant", 0.5), ("invscaling", 0.01), ("adaptive", 0.01)]:
    model = SGDRegressor(learning_rate=strategy, eta0=eta0, max_iter=200, tol=None, random_state=0)
    model.fit(features, targets)
```

```
constant   eta0 = 0.001  loss = 0.09068
constant   eta0 = 0.5    diverged, loss = 3.03e+25
invscaling eta0 = 0.01   loss = 0.09067
adaptive   eta0 = 0.01   loss = 0.09335
```

- `constant` uses `eta0` throughout. `0.5` is far beyond what one-sample updates can tolerate and explodes.
- `invscaling` uses `eta0 / t^power_t`, an inverse-time schedule.
- `adaptive` keeps `eta0` while the loss improves and divides it by 5 when it stalls.
- The default for `SGDRegressor` is `invscaling`; for `SGDClassifier` it is `optimal`, a heuristic based on the regularization strength.

Gradient boosting (Phase 07) also has a `learning_rate`, which scales each tree's contribution rather than a gradient step, but the trade-off is identical: smaller is more stable and needs more iterations.

## Exercises

1. **Easy.** For `f(x) = 2x^2`, compute by hand the learning rate that converges in one step and the one beyond which gradient descent diverges. Check both with `quadratic_trajectory`.
2. **Medium.** Standardize the features in `make_regression_data` (zero mean, unit variance) before building the design matrix. Recompute the Hessian eigenvalues, the condition number, and the number of steps needed to reach a loss within `1e-6` of the optimum at `eta = 1/L`. Compare with the unscaled data.
3. **Hard.** Implement a learning rate range test: start at `eta = 1e-5` and multiply it by `1.1` every step for 150 steps, recording the loss. Plot loss vs `eta` on a log axis. Where does the loss start to rise, and how does that point compare to `2 / lambda_max`?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Learning rate | "Step size" | The scalar `eta` in `w <- w - eta g`; the most important optimizer hyperparameter |
| Divergence | "The loss blew up" | Iterates grow without bound because `|1 - eta lambda_max| > 1` |
| Hessian | "The curvature matrix" | The matrix of second derivatives; for MSE it is `(2/n) X^T X` |
| Stability limit | "The max learning rate" | `eta < 2 / lambda_max` for gradient descent on a quadratic |
| Condition number | "How stretched the problem is" | `lambda_max / lambda_min`; the number of steps needed grows with it |
| Learning rate schedule | "Decay" | A rule that changes `eta` over time, mainly to tame gradient noise in SGD |
| Backtracking line search | "Auto step size" | Shrinking a trial step until the Armijo sufficient-decrease condition holds |

## Further Reading

- [Goodfellow, Bengio, Courville, Deep Learning, section 4.3 and 8.3](https://www.deeplearningbook.org/) -- gradient-based optimization, curvature and learning rate schedules
- [Nocedal and Wright, Numerical Optimization, chapter 3](https://link.springer.com/book/10.1007/978-0-387-40065-5) -- line search methods and the Wolfe/Armijo conditions
- [Distill, Why Momentum Really Works](https://distill.pub/2017/momentum/) -- interactive analysis of step size, eigenvalues and convergence that also previews lesson 07
- [scikit-learn SGD documentation](https://scikit-learn.org/stable/modules/sgd.html) -- learning rate options for `SGDRegressor` and `SGDClassifier`
