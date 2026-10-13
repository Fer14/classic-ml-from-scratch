# The Bias-Variance Tradeoff

> Simple models are consistently wrong. Complex models are inconsistently right. Test error pays for both.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 03 (Probability and Statistics: expectation and variance), Phase 04 Lesson 04 (Overfitting and Regularization)
**Time:** ~50 minutes

## Learning Objectives

- State the bias-variance decomposition of expected squared error and explain each of its three terms
- Estimate bias and variance by simulation: train the same model on many independent training sets
- Show that more data reduces variance but not bias, and that regularization trades one for the other
- Use the decomposition to explain why bagging deep trees improves on a single deep tree

## The Problem

In the last lesson a degree-15 polynomial overfitted and a straight line underfitted. "Overfit" and "underfit" are labels, though, not measurements. When your model performs badly you need to know *why*, because the fixes are opposite:

- If the model is too rigid, more data will not help. You need a more flexible model or better features.
- If the model is too unstable, a more flexible model makes it worse. You need more data, regularization, or averaging.

The bias-variance decomposition splits test error into the part caused by rigidity, the part caused by instability, and the part nobody can fix.

## The Concept

### A thought experiment

Your training set is one random draw from the world. Draw a different 25 points and you get a different fitted model. Imagine repeating that 300 times and looking at the predictions at one test point `x0`:

```
                  true value f(x0)
                        |
 degree 1:   ooooo      |            predictions cluster tightly
             ^ mean     |            but in the wrong place
                        |            (high bias, low variance)

 degree 9:  o    o   o  | o  o    o  predictions centered on truth
                     mean           but scattered widely
                        |            (low bias, high variance)
```

- **Bias**: how far the *average* prediction (over training sets) is from the truth.
- **Variance**: how much individual predictions scatter around their own average.

### The decomposition

Assume `y = f(x) + noise` with `E[noise] = 0` and `Var[noise] = sigma^2`. Let `f_hat_D` be the model trained on dataset `D`. The expected squared error at `x`, averaged over datasets and noise, is:

```
E[(y - f_hat_D(x))^2] = (E_D[f_hat_D(x)] - f(x))^2     bias^2
                      + E_D[(f_hat_D(x) - E_D[f_hat_D(x)])^2]   variance
                      + sigma^2                         irreducible noise
```

The cross terms vanish because the noise on the test point is independent of the training set and has mean zero. Averaging over many `x` gives the decomposition of overall test MSE.

The noise term is a floor: no model can beat `sigma^2` on average, because the label itself is random.

### The tradeoff

```
error
  ^
  | \                                        /
  |  \  bias^2                     variance /
  |   \                                    /
  |    \____                         _____/
  |         \______           ______/
  |                \____ ____/       total = bias^2 + variance + noise
  |                     X      <- minimum
  |  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _   noise floor sigma^2
  +----------------------------------------------> model flexibility
```

Increasing flexibility (higher degree, deeper tree, smaller `lambda`, smaller `k` in k-NN) lowers bias and raises variance. The best model sits where their sum is smallest.

### What moves which term

| Change | Bias | Variance |
|---|---|---|
| More flexible model | Down | Up |
| Stronger regularization | Up | Down |
| More training data | Unchanged | Down |
| Better features | Down | Depends |
| Averaging many models (bagging) | Unchanged | Down |
| Less label noise | Unchanged | Down (and the noise floor drops) |

The "more data" row is why a large dataset lets you use a more flexible model: the variance penalty for flexibility shrinks as `n` grows.

### Diagnosing in practice

You only have one dataset, so you cannot run the simulation. You use proxies:

- High training error and similar validation error: high bias. Add capacity or features.
- Low training error and much higher validation error: high variance. Regularize, simplify, get more data.
- **Learning curves** (error versus training-set size) make this explicit: a high-bias model plateaus early at a high error; a high-variance model has a large train/validation gap that shrinks as `n` grows.

## Build It

### Step 1: Simulate many training sets

```python
NOISE_STD = 0.3


def true_function(x):
    return np.sin(2 * np.pi * x)


def sample_training_set(n_samples, rng):
    x = rng.uniform(0, 1, n_samples)
    return x, true_function(x) + rng.normal(0, NOISE_STD, n_samples)


def simulate(model_fn, n_datasets, n_samples, test_x, rng):
    predictions = np.empty((n_datasets, len(test_x)))
    for i in range(n_datasets):
        x, y = sample_training_set(n_samples, rng)
        predictions[i] = model_fn(x, y, test_x)
    return predictions
```

`predictions[i, j]` is the prediction of the model trained on dataset `i` at test point `j`. Column `j` is the scatter in the thought experiment.

At a single point `x0 = 0.25` where `f(x0) = 1`:

```
degree 1: mean prediction=0.496, std of predictions=0.148
degree 9: mean prediction=1.005, std of predictions=0.425
```

### Step 2: Decompose

```python
def decompose(predictions, test_x):
    mean_prediction = predictions.mean(axis=0)
    bias_squared = np.mean((mean_prediction - true_function(test_x)) ** 2)
    variance = np.mean(predictions.var(axis=0))
    return bias_squared, variance


def expected_test_mse(predictions, test_x, rng):
    noisy_targets = true_function(test_x) + rng.normal(0, NOISE_STD, predictions.shape)
    return np.mean((predictions - noisy_targets) ** 2)
```

`expected_test_mse` measures error directly against fresh noisy labels, so you can check that the three terms add up:

```
degree |  bias^2  | variance |  noise   |   sum    | measured MSE
   0   |   0.5427 |   0.0241 |   0.0900 |   0.6567 |   0.6611
   1   |   0.1557 |   0.0264 |   0.0900 |   0.2721 |   0.2770
   3   |   0.0031 |   0.0182 |   0.0900 |   0.1113 |   0.1125
   5   |   0.0006 |   0.0668 |   0.0900 |   0.1574 |   0.1559
   7   |   0.0013 |   0.5008 |   0.0900 |   0.5921 |   0.5945
   9   |   0.1192 |  43.8943 |   0.0900 |  44.1035 |  44.1283
```

The sum matches the measured MSE up to simulation noise. Bias collapses by degree 3; variance explodes after degree 5. Degree 3 wins. (The small bias at degree 9 is an artifact: a few wild fits drag the average.)

### Step 3: More data shrinks variance, not bias

```
n=  25 degree 1: bias^2=0.1537 variance=0.0247
n= 400 degree 1: bias^2=0.1565 variance=0.0016
n=  25 degree 9: bias^2=0.0142 variance=4.9403
n= 400 degree 9: bias^2=0.0000 variance=0.0019
```

The straight line stays wrong no matter how much data you give it. The degree-9 polynomial goes from useless to excellent.

## Use It

There is no `bias_variance` function in scikit-learn, but you can plug any sklearn estimator into the same simulation. Decision trees are a clean example because depth is a direct flexibility knob, and `BaggingRegressor` attacks variance.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


def tree_model(depth, bagged=False):
    def model_fn(x, y, test_x):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=0)
        model = BaggingRegressor(tree, n_estimators=30, random_state=0) if bagged else tree
        return model.fit(x[:, None], y).predict(test_x[:, None])
    return model_fn
```

```
tree depth    1: bias^2=0.0535 variance=0.0412 total=0.1847
tree depth    3: bias^2=0.0027 variance=0.0473 total=0.1399
tree depth None: bias^2=0.0008 variance=0.0951 total=0.1859
bagged deep trees: bias^2=0.0007 variance=0.0467 total=0.1374
```

Bagging keeps the near-zero bias of a fully grown tree and halves its variance. That single line is the motivation for random forests (Phase 07). If you need a ready-made function, the third-party `mlxtend.evaluate.bias_variance_decomp` does this simulation with bootstrap samples of one dataset.

## Exercises

1. **Easy.** Increase `NOISE_STD` to 0.6. Which columns of the degree table change, and by roughly how much? Which do not change at all?
2. **Medium.** Replace polynomial degree with ridge `lambda` (use the `RidgePolynomial` class from Lesson 04 at degree 12) and print bias^2 and variance for 8 log-spaced `lambda` values. Where is the minimum total error?
3. **Hard.** Implement a k-nearest-neighbors regressor from scratch and compute bias^2 and variance for `k = 1, 3, 10, 30, 100` with `n = 200`. Derive why the variance at `k` is about `sigma^2 / k` when the neighbors are close, and check it against your numbers.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Bias | "The model is too simple" | Squared distance between the average prediction over training sets and the true function |
| Variance | "The model is unstable" | Expected squared deviation of a single trained model's predictions from the average prediction |
| Irreducible error | "Noise" | `sigma^2`, the variance of the label around its true mean; a floor for any model |
| Bias-variance tradeoff | "Simple vs complex" | Flexibility lowers bias and raises variance; test error is minimized in between |
| Learning curve | "Error vs data size" | Training and validation error plotted against the number of training samples |
| Bagging | "Average many models" | Training models on bootstrap resamples and averaging them, which reduces variance |

## Further Reading

- [An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/) -- section 2.2.2 derives the decomposition and shows the U-shaped test error curve
- [The Elements of Statistical Learning (ESL)](https://hastie.su.domains/ElemStatLearn/) -- section 7.3 gives the decomposition for k-NN and linear models
- [Scott Fortmann-Roe, "Understanding the Bias-Variance Tradeoff"](https://scott.fortmann-roe.com/docs/BiasVariance.html) -- a short essay with the classic bullseye picture
- [StatQuest: Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) -- a 6-minute visual explanation
