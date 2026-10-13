# SGD and Momentum

> Do not read the whole dataset to take one step. Estimate the gradient from a handful of samples, and let momentum smooth the ride.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 02 Lessons 04-06 (Gradient Descent, Learning Rate, Convexity)
**Time:** ~75 minutes

## Learning Objectives

- Implement mini-batch stochastic gradient descent with shuffling and compare batch, mini-batch and single-sample updates on the same data budget
- Explain why a constant learning rate leaves SGD bouncing at a noise floor and show how decay removes it
- Implement heavy-ball momentum and Nesterov momentum and measure their speedup on an ill-conditioned problem
- Use scikit-learn's `SGDRegressor`, including `partial_fit` for streaming data

## The Problem

Full-batch gradient descent computes the gradient over every training sample before moving once. With 20,000 samples that is 20,000 predictions per step. With 20 million samples it is hopeless: a single step could take minutes, and you need thousands of steps.

There is also a second problem from the learning rate lesson: on ill-conditioned problems, gradient descent crawls along shallow valleys because the steep direction caps the step size.

Stochastic gradient descent attacks the first problem; momentum attacks the second. Together (and in descendants like Adam) they train almost every large model in existence. scikit-learn's `SGDClassifier`, `SGDRegressor`, and `MLPClassifier` all use them.

## The Concept

### The gradient is an average, so estimate it

The MSE gradient is an average over samples:

```
grad L(w) = (1/n) * sum_i grad l_i(w)
```

An average can be estimated from a random subset. Pick a mini-batch `B` of size `b`:

```
g_B(w) = (1/b) * sum_{i in B} grad l_i(w)
```

`g_B` is an unbiased estimate: `E[g_B] = grad L`. Its variance shrinks like `1/b`. So a step on a mini-batch of 32 points in the right direction on average, at 1/625 the cost of a full-batch step on 20,000 samples.

```
batch GD:       1 update per epoch,   exact direction
mini-batch SGD: n/b updates per epoch, noisy direction (variance ~ 1/b)
pure SGD (b=1): n updates per epoch,   very noisy direction
```

An **epoch** is one pass over the data. In practice you shuffle the data each epoch and walk through it in consecutive chunks, so every sample is used once per epoch.

### The noise floor

Near the minimum, the true gradient goes to zero but individual sample gradients do not: each sample still "wants" a slightly different `w`. With a constant learning rate, SGD keeps taking steps of size about `eta * |noise|` and bounces around the minimum forever. The size of that ball scales with `eta`.

Two fixes:

- **Decay the learning rate**, for example `eta_t = eta_0 / (1 + decay * t)`. Classic theory (Robbins-Monro) requires `sum eta_t = inf` and `sum eta_t^2 < inf`; `1/t` schedules satisfy both.
- **Grow the batch**, which shrinks the noise directly.

### Momentum: a heavy ball

Plain GD forgets its history. Momentum keeps a running velocity:

```
v_{t+1} = beta * v_t - eta * grad L(w_t)
w_{t+1} = w_t + v_{t+1}
```

`beta` (typically 0.9) is how much velocity survives each step. Two effects:

- In directions where the gradient keeps pointing the same way (the floor of a ravine), velocity accumulates, up to `1 / (1 - beta) = 10` times a single step.
- In directions where the gradient flips sign each step (the steep walls), consecutive contributions cancel and the zig-zag is damped.

```
 plain GD in a ravine                momentum in a ravine
 \/\/\/\/\/\/\/\/\/\/--> slow        \_____________________--> fast
```

### Nesterov momentum

Nesterov evaluates the gradient at the point momentum is about to carry you to, not where you are:

```
v_{t+1} = beta * v_t - eta * grad L(w_t + beta * v_t)
w_{t+1} = w_t + v_{t+1}
```

If the momentum is about to overshoot, the look-ahead gradient notices one step early and brakes. On convex problems this provably improves the convergence rate from depending on `kappa` to depending on `sqrt(kappa)`.

### Momentum with noisy gradients

With SGD, momentum's speedup is mostly an effective learning rate of `eta / (1 - beta)`. If you add `beta = 0.9` without shrinking `eta`, you have effectively multiplied the learning rate by 10, and the noise floor rises accordingly. When you switch momentum on, scale `eta` down.

### A word on Adam

Adam combines momentum with a per-weight learning rate that divides by a running average of squared gradients. It is the default for neural networks and is available in scikit-learn as `MLPClassifier(solver="adam")`. For the convex models in this course, plain SGD with a schedule or a second-order solver like L-BFGS is usually enough.

## Build It

### Step 1: Mini-batch SGD with shuffling

```python
def minibatch_sgd(design, targets, batch_size, learning_rate, epochs, decay=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = len(targets)
    weights = np.zeros(design.shape[1])
    history, samples_seen, step = [], 0, 0
    for _ in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch = order[start:start + batch_size]
            rate = learning_rate / (1 + decay * step)
            weights = weights - rate * batch_gradient(weights, design[batch], targets[batch])
            samples_seen += len(batch)
            step += 1
            if step % max(1, 2000 // batch_size) == 0:
                history.append((samples_seen, mse(weights, design, targets)))
    return weights, np.array(history)
```

`batch_size = n` gives batch GD, `batch_size = 1` gives pure SGD, everything in between is mini-batch. Two epochs on 20,000 samples with 5 features (best possible MSE `1.01172`):

```
batch (n=20000)  updates =      2  final MSE = 1.89464
mini-batch (32)  updates =   1250  final MSE = 1.01596
pure SGD (1)     updates =  40000  final MSE = 1.01647
```

Same amount of data read, same compute, but batch GD has taken only 2 steps. Mini-batch SGD is already within `0.004` of optimal. Pure SGD is close too, but in real code it is slow because it cannot use vectorized matrix operations; batch sizes of 32-256 are the usual compromise.

### Step 2: The noise floor

```python
for label, decay in [("constant eta", 0.0), ("decaying eta", 0.002)]:
    weights, history = minibatch_sgd(design, targets, 8, 0.05, 3, decay=decay)
    tail = history[-20:, 1]
```

```
constant eta  mean excess = 4.29e-02, std = 2.24e-02
decaying eta  mean excess = 4.18e-03, std = 2.46e-03
```

With small batches and a constant rate, the loss jitters about `0.04` above optimal. Decay cuts both the excess and the jitter by 10x.

### Step 3: Momentum and Nesterov

```python
def momentum_descent(gradient_fn, start, learning_rate, beta, steps, nesterov=False):
    position = np.array(start, dtype=float)
    velocity = np.zeros_like(position)
    path = [position.copy()]
    for _ in range(steps):
        lookahead = position + beta * velocity if nesterov else position
        velocity = beta * velocity - learning_rate * gradient_fn(lookahead)
        position = position + velocity
        path.append(position.copy())
    return np.array(path)
```

On `f = 0.01 x^2 + y^2` (curvatures `0.02` and `2`, condition number 100), starting at `(10, 1)`:

```
plain GD      eta = 0.5: steps to f < 1e-6 = 688
plain GD      eta = 0.9: steps to f < 1e-6 = 381
momentum 0.9  eta = 0.5: steps to f < 1e-6 = 126
Nesterov 0.9  eta = 0.5: steps to f < 1e-6 = 62
```

Plain GD cannot go much above `eta = 1` (the limit `2/L` for `L = 2`), and even near it needs 381 steps. Momentum is 3x faster and Nesterov 6x faster. The `x` coordinate shows why:

```
momentum x-coordinate: 10.00 9.90 9.71 9.44 9.11 8.72 8.28 7.80 7.29
plain GD x-coordinate: 10.00 9.90 9.80 9.70 9.61 9.51 9.41 9.32 9.23
```

Plain GD moves a constant `~0.1` per step along the flat direction. Momentum's steps grow: `0.10, 0.19, 0.27, 0.33, ...` as velocity accumulates.

### Step 4: Momentum with SGD

```python
velocity = beta * velocity - rate * batch_gradient(weights, design[batch], targets[batch])
weights = weights + velocity
```

```
beta = 0.0, eta = 0.01   effective eta = 0.010: mean excess MSE = 3.77e-03
beta = 0.9, eta = 0.01   effective eta = 0.100: mean excess MSE = 4.29e-02
beta = 0.9, eta = 0.001  effective eta = 0.010: mean excess MSE = 3.72e-03
```

Naively adding momentum made things 10x worse; scaling `eta` by `1 - beta` restores the same noise floor. On this well-conditioned problem, momentum buys nothing over plain SGD; its benefit appears on ill-conditioned losses like Step 3.

The script saves `sgd_and_momentum.png` with loss-vs-samples curves and the two ravine paths.

## Use It

```python
from sklearn.linear_model import LinearRegression, SGDRegressor

reference = LinearRegression().fit(features, targets)
model = SGDRegressor(learning_rate="invscaling", eta0=0.01, max_iter=5, tol=None, random_state=0).fit(features, targets)

streaming = SGDRegressor(learning_rate="invscaling", eta0=0.01, random_state=0)
for chunk in np.array_split(np.arange(len(targets)), 20):
    streaming.partial_fit(features[chunk], targets[chunk])
```

```
LinearRegression coef = [ 1.511 -2.012  0.503  2.995 -0.999], intercept = 3.998
SGDRegressor     coef = [ 1.517 -2.024  0.514  2.978 -0.998], intercept = 3.996
partial_fit on 20 chunks, MSE = 1.01454
```

- `SGDRegressor` does pure SGD (one sample per update) with shuffling every epoch and an inverse-scaling schedule. Five epochs land within `0.02` of the exact coefficients.
- `partial_fit` performs one epoch over whatever data you pass, keeping the weights between calls. This is how you train on data that does not fit in memory or arrives as a stream.
- `SGDRegressor` has no momentum option. `MLPRegressor(solver="sgd", momentum=0.9, nesterovs_momentum=True)` does, and `solver="adam"` gives you Adam.
- SGD is very sensitive to feature scale (Phase 04 covers this in depth). Always put a `StandardScaler` in front of it.

## Exercises

1. **Easy.** Rerun Step 1 with batch sizes 8, 64, 512 and 4096 for 2 epochs each. Plot final MSE against batch size. Where does the trade-off between noise and number of updates balance out?
2. **Medium.** Implement Polyak averaging: keep a running average of the weights over the last epoch of constant-rate SGD and evaluate the averaged weights. Compare its excess MSE with the last iterate and with the decaying schedule from Step 2.
3. **Hard.** Implement Adam from scratch (`m = b1 m + (1-b1) g`, `s = b2 s + (1-b2) g^2`, bias correction, `w -= eta m_hat / (sqrt(s_hat) + eps)`). Compare it to plain GD and Nesterov on the ravine from Step 3 and on a version of the regression data where one feature is multiplied by 100. Which optimizer is least affected by the bad scaling, and why?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Stochastic gradient descent | "SGD" | Updating weights with a gradient estimated from a random sample or mini-batch instead of the full dataset |
| Mini-batch | "Batch" | The subset of `b` samples used for one update; typical sizes are 32-256 |
| Epoch | "One pass" | Processing every training sample once; with mini-batches this is `n/b` updates |
| Noise floor | "SGD jitter" | The region around the minimum where constant-rate SGD keeps bouncing because sample gradients disagree |
| Momentum | "Heavy ball" | Accumulating a velocity `v = beta v - eta g` so consistent directions speed up and oscillations cancel |
| Nesterov momentum | "Look-ahead momentum" | Momentum that evaluates the gradient at `w + beta v`, braking before it overshoots |
| Effective learning rate | "What eta really is" | With momentum `beta`, steady-state steps are `eta / (1 - beta)` times the gradient |
| `partial_fit` | "Online learning" | scikit-learn API for updating a model incrementally on new chunks of data |

## Further Reading

- [Distill, Why Momentum Really Works (Gabriel Goh)](https://distill.pub/2017/momentum/) -- interactive explanation of momentum on quadratics, including the `sqrt(kappa)` speedup
- [Sebastian Ruder, An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) -- SGD, momentum, Nesterov, Adagrad, RMSprop and Adam side by side
- [Bottou, Curtis, Nocedal, Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) -- the theory of SGD noise, step sizes and mini-batching
- [scikit-learn Stochastic Gradient Descent guide](https://scikit-learn.org/stable/modules/sgd.html) -- practical tips for `SGDRegressor`, `SGDClassifier` and `partial_fit`
