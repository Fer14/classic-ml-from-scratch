# Loss Functions

> The loss is the only thing your model is trying to do. Choose it like you mean it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 02 (Calculus and Optimization), Phase 03 (Probability and Statistics), Phase 04 Lesson 01 (What Is Machine Learning)
**Time:** ~60 minutes

## Learning Objectives

- Implement MSE, MAE, Huber, binary cross-entropy, categorical cross-entropy, and hinge loss in NumPy
- Explain why the best constant prediction is the mean under MSE and the median under MAE
- Show how the choice of loss changes a fitted line when the data contains outliers
- Explain why cross-entropy punishes confident mistakes so heavily and connect it to maximum likelihood

## The Problem

An optimizer does exactly one thing: it makes a number go down. That number is the loss. If the loss does not reflect what you care about, the optimizer will happily give you a model that is "optimal" and useless.

Suppose you predict delivery times and a few deliveries were delayed by a storm. With squared error, those few points dominate the fit and drag every prediction upward. Suppose you classify tumors and your loss counts every mistake equally. A model that says "benign, 51% sure" and one that says "benign, 99.9% sure" get the same score when both are wrong, even though the second is far more dangerous.

The loss encodes your assumptions about noise and your preferences about errors. This lesson builds the standard ones and shows what each one actually optimizes.

## The Concept

### Loss, cost, objective

- A **loss** `L(y, y_hat)` scores a single prediction.
- The **cost** (empirical risk) averages it over the dataset: `J = (1/n) * sum L(y_i, y_hat_i)`.
- The **objective** is what the optimizer minimizes: the cost plus any regularization (Lesson 04).

People use the three words loosely. What matters is that training minimizes an average of per-example losses.

### Regression losses

```
residual r = y_hat - y

MSE    L = r^2
MAE    L = |r|
Huber  L = 0.5 * r^2                    if |r| <= delta
           delta * (|r| - 0.5 * delta)  otherwise
```

```
loss
 ^        MSE (parabola)
 |   \                 /
 |    \               /     Huber: parabola near 0,
 |     \     MAE     /      straight lines far away
 |      \   (V)     /
 |       \  / \    /
 |        \/   \  /
 +---------+----+----------> residual
           0
```

| Loss | Gradient w.r.t. prediction | Outlier influence | Optimal constant |
|---|---|---|---|
| MSE | `2r` (grows with error) | Large, quadratic | Mean |
| MAE | `sign(r)` (constant size) | Small, bounded | Median |
| Huber | `r` if small, `±delta` if large | Bounded | Between mean and median |

The "optimal constant" row is the cleanest intuition. If your model could only output one number `c`, MSE would pick `c = mean(y)` (set the derivative `2 * sum(c - y_i)` to zero) and MAE would pick the median (half the points pull left, half pull right). Means are dragged by outliers; medians are not.

### MSE is a likelihood

If you assume `y = f(x) + noise` with Gaussian noise, the negative log-likelihood (Phase 03, Maximum Likelihood) is, up to constants, the sum of squared residuals. Minimizing MSE is maximum likelihood under Gaussian noise. MAE corresponds to Laplace noise, which has heavier tails, which is why it tolerates outliers.

### Classification losses

For binary labels `y in {0, 1}` and predicted probability `p = P(y = 1)`:

```
Binary cross-entropy (log loss)
    L = -[ y * log(p) + (1 - y) * log(1 - p) ]
```

It is the negative log-likelihood of a Bernoulli variable. Look at how it grows as the probability assigned to the true class shrinks:

```
p(true class)   -log p
    0.99        0.0101
    0.90        0.1054
    0.50        0.6931
    0.10        2.3026
    0.01        4.6052
```

A single confident mistake costs more than dozens of slightly uncertain correct answers. That is what you want from a model that outputs probabilities.

For `K` classes with predicted probabilities `p_1..p_K`, **categorical cross-entropy** is `-log p_true`, the same idea.

**Hinge loss** is used by support vector machines (Phase 06). Labels are `y in {-1, +1}` and the model outputs a raw score `s`:

```
L = max(0, 1 - y * s)
```

It is zero when the example is on the correct side with margin at least 1, and grows linearly otherwise. It does not produce probabilities.

### Why not optimize accuracy directly?

Accuracy (0-1 loss) is a step function of the parameters: its gradient is zero almost everywhere and undefined at the steps. Gradient descent cannot use it. Cross-entropy and hinge are smooth (or piecewise linear) **surrogate losses** that upper-bound the 0-1 loss and can be optimized. You still *report* accuracy or another metric (Lesson 07); you just do not *train* on it.

## Build It

### Step 1: Regression losses

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    quadratic = 0.5 * residual ** 2
    linear = delta * (residual - 0.5 * delta)
    return np.mean(np.where(residual <= delta, quadratic, linear))
```

With residuals `[-0.5, 0, 0.5, 1, 6]` you get MSE 7.5, MAE 1.6, Huber 1.25. The single residual of 6 contributes 36 to the MSE sum and only 5.5 to the Huber sum.

### Step 2: Best constant under each loss

Eight salaries, one of them a CEO at 400:

```python
salaries = np.array([40, 42, 45, 47, 50, 52, 55, 400], dtype=float)
grid = np.linspace(0, 450, 45001)
best_mse = grid[np.argmin([mse_loss(salaries, c) for c in grid])]
best_mae = grid[np.argmin([mae_loss(salaries, c) for c in grid])]
```

MSE picks 91.37 (the mean); MAE picks a value between 47 and 50 (any median minimizes it). One outlier nearly doubled the MSE answer.

### Step 3: Fit a line with each loss

You train `y = slope * x + intercept` with gradient descent, swapping only the gradient of the loss with respect to the prediction.

```python
def gradient(loss_name, residual, delta=1.0):
    if loss_name == "mse":
        return 2 * residual
    if loss_name == "mae":
        return np.sign(residual)
    return np.clip(residual, -delta, delta)


def fit_line(x, y, loss_name, learning_rate=0.05, epochs=4000):
    slope, intercept = 0.0, 0.0
    for _ in range(epochs):
        residual = slope * x + intercept - y
        g = gradient(loss_name, residual)
        slope -= learning_rate * np.mean(g * x)
        intercept -= learning_rate * np.mean(g)
    return slope, intercept
```

The data is `y = 2x + 1 + noise` with 5 of 60 points shifted up by 15.

```
mse   fit: y = 1.572 x + 3.319
mae   fit: y = 2.029 x + 0.950
huber fit: y = 2.033 x + 1.005
```

MSE is pulled toward the outliers. MAE and Huber recover the true line.

### Step 4: Classification losses

```python
def binary_cross_entropy(y_true, probabilities, eps=1e-15):
    p = np.clip(probabilities, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def hinge(y_signed, scores):
    return np.mean(np.maximum(0, 1 - y_signed * scores))


def categorical_cross_entropy(y_true_index, probabilities, eps=1e-15):
    p = np.clip(probabilities, eps, 1)
    return -np.mean(np.log(p[np.arange(len(y_true_index)), y_true_index]))
```

The clip prevents `log(0) = -inf`. On five examples:

```
BCE confident and right: 0.0952
BCE unsure but right:    0.5456
BCE one confident miss:  0.8451
```

All three would have similar accuracy; the losses are very different.

## Use It

`sklearn.metrics` has every loss you built, and `HuberRegressor` fits a linear model with Huber loss.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, hinge_loss
from sklearn.linear_model import LinearRegression, HuberRegressor

mean_squared_error(y_true, y_pred)
log_loss(labels, confident_wrong)
hinge_loss(signed, scores)
ols = LinearRegression().fit(x[:, None], y)
huber = HuberRegressor(epsilon=1.35).fit(x[:, None], y)
```

The metric values match yours to four decimals. `LinearRegression` reproduces the outlier-dragged MSE line (1.572, 3.319) and `HuberRegressor` gives 2.048, 0.931. It differs slightly from your Huber fit because it also estimates a noise scale and adds a small L2 penalty by default.

## Exercises

1. **Easy.** Change `delta` in the Huber fit to 0.1, 1, 5, and 20. Which value behaves like MAE, and which like MSE? Explain from the formula.
2. **Medium.** Implement the quantile (pinball) loss `L = max(q * r, (q - 1) * r)` with `r = y - y_hat`, plug its gradient into `fit_line`, and fit the 10th, 50th and 90th percentile lines. Check that roughly 10%, 50% and 90% of points fall below each line.
3. **Hard.** Derive the gradient of binary cross-entropy with respect to the logit `z` where `p = sigmoid(z)`, and show it simplifies to `p - y`. Use it to train a logistic regression from scratch on the fruit data from Lesson 01 and compare its log loss with `sklearn.linear_model.LogisticRegression`.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Loss function | "How wrong the model is" | A function `L(y, y_hat)` scoring one prediction; training minimizes its average |
| MSE | "Squared error" | Mean of squared residuals; maximum likelihood under Gaussian noise; optimal constant is the mean |
| MAE | "Absolute error" | Mean of absolute residuals; robust to outliers; optimal constant is the median |
| Huber loss | "Smooth robust loss" | Quadratic for small residuals, linear beyond `delta`; combines MSE smoothness with MAE robustness |
| Cross-entropy | "Log loss" | Negative log of the probability assigned to the true class; Bernoulli/categorical negative log-likelihood |
| Hinge loss | "SVM loss" | `max(0, 1 - y * s)`; zero for correct predictions with enough margin |
| Surrogate loss | "Something differentiable" | A smooth loss optimized in place of a non-differentiable target like the 0-1 loss |

## Further Reading

- [The Elements of Statistical Learning (ESL)](https://hastie.su.domains/ElemStatLearn/) -- section 10.6 compares squared, absolute, Huber, exponential and hinge losses and their robustness
- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) -- reference for every regression and classification loss used here
- [StatQuest: Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) -- the link between likelihood and the losses you minimize
- [Kevin Murphy, Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) -- chapters 4-5 derive MSE and cross-entropy from probabilistic models
