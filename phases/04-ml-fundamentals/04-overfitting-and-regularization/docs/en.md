# Overfitting and Regularization

> A model that memorizes the training set has learned the noise. Regularization makes memorizing expensive.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra), Phase 02 (Calculus and Optimization), Phase 04 Lessons 02-03 (Train/Validation/Test, Loss Functions)
**Time:** ~60 minutes

## Learning Objectives

- Recognize overfitting and underfitting from the gap between training and test error
- Implement L2 (ridge) regularization in closed form and see how it shrinks weights and closes the gap
- Compare three practical regularizers: a weight penalty, more data, and early stopping
- Explain why features must be standardized before applying a weight penalty

## The Problem

You have 20 noisy points sampled from a sine wave. A degree-15 polynomial has 16 coefficients, enough to bend through nearly every point. Its training error is the lowest of any model you try. Its test error is about 100 times worse than a cubic.

```
degree  3 | train MSE 0.0593 | test MSE    0.0920 | max |w|         7.19
degree 15 | train MSE 0.0374 | test MSE    9.8478 | max |w|      6631.66
```

Look at the last column. To pass through noisy points, the flexible model uses enormous coefficients that cancel each other out between the training points and explode everywhere else. Overfitting is visible in the weights.

You could just pick a smaller degree. But often you do not know the right capacity in advance, or the model is not a polynomial at all. You want a knob that keeps a flexible model from using its flexibility on noise.

## The Concept

### Underfitting, good fit, overfitting

```
error
  ^
  |\                                     test error
  | \                                 __/
  |  \                             __/
  |   \___                    ____/
  |       \______      ______/
  |              \____/
  |  ----___
  |         ----____                     train error
  |                 -------_______________
  +--------------------------------------------> model capacity
     underfit      |  sweet spot |      overfit
   (high bias)                       (high variance)
```

- **Underfitting**: both errors high. The model is too simple to capture the pattern.
- **Overfitting**: training error low, test error high. The model captured the noise.
- The **generalization gap** is test error minus training error. A large gap is the signature of overfitting.

Lesson 05 (Bias-Variance) makes this picture quantitative.

### L2 regularization (ridge)

Add a penalty on the size of the weights to the loss:

```
J(w) = sum (y_i - x_i . w - b)^2  +  lambda * sum_j w_j^2
```

Setting the gradient to zero gives a closed form (compare with the normal equation from Phase 01):

```
w = (X^T X + lambda * I)^(-1) X^T (y - mean(y))
```

Three things happen as `lambda` grows:

1. Large weights become expensive, so the model prefers smooth functions.
2. `X^T X + lambda * I` is always invertible, even when features are collinear or there are more features than samples.
3. Training error goes up a little; test error usually goes down a lot, until `lambda` is so large that the model underfits.

Two details matter in practice:

- **Do not penalize the intercept.** Shifting all targets by a constant should not change the fit. Center `y` and fit the intercept separately.
- **Standardize features first.** The penalty treats all weights equally. If `x^15` ranges up to 1 and `x` ranges up to 1000, the same weight means different things. Standardizing puts every feature on the same scale so the penalty is fair (Lesson 08).

### L1 regularization (lasso), briefly

Replace `w_j^2` with `|w_j|`. The L1 penalty pushes some weights exactly to zero, which performs feature selection. It has no closed form; you will implement it with coordinate descent in Phase 05 (Lasso and Elastic Net).

### Other regularizers

| Technique | How it limits the model |
|---|---|
| Weight penalty (L1, L2) | Adds a cost to large parameters |
| More data | Noise averages out; the same model has less room to memorize |
| Early stopping | Stops gradient descent before the weights grow large enough to fit noise |
| Smaller model | Fewer parameters (lower degree, shallower tree) |
| Ensembling | Averaging many high-variance models (Phase 07) |

Early stopping and L2 are closely related: gradient descent starting from zero weights moves along directions of large curvature first, so stopping early leaves the low-curvature (noise-fitting) directions near zero, like a penalty would.

### Choosing lambda

`lambda` is a hyperparameter. Choose it on a validation set or with cross-validation (Lesson 06), never on the test set. In this lesson the code prints the test-optimal `lambda` only to show the shape of the curve.

## Build It

### Step 1: Data and polynomial features

```python
def true_function(x):
    return np.sin(2 * np.pi * x)


def make_data(n_samples, rng, noise=0.25):
    x = rng.uniform(0, 1, n_samples)
    return x, true_function(x) + rng.normal(0, noise, n_samples)


def polynomial_features(x, degree):
    return np.column_stack([x ** power for power in range(1, degree + 1)])
```

You draw 20 training points and 1000 test points.

### Step 2: Ridge with standardization and an unpenalized intercept

```python
class RidgePolynomial:
    def __init__(self, degree, penalty):
        self.degree = degree
        self.penalty = penalty

    def fit(self, x, y):
        features = polynomial_features(x, self.degree)
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0)
        scaled = (features - self.feature_mean) / self.feature_std
        self.intercept = y.mean()
        gram = scaled.T @ scaled + self.penalty * np.eye(self.degree)
        self.weights = np.linalg.solve(gram, scaled.T @ (y - self.intercept))
        return self

    def predict(self, x):
        scaled = (polynomial_features(x, self.degree) - self.feature_mean) / self.feature_std
        return scaled @ self.weights + self.intercept
```

With `penalty=1e-10` this is ordinary least squares (the tiny value only keeps the solve numerically stable). Because the features are centered, the intercept is exactly `mean(y)`.

### Step 3: Sweep lambda on the degree-15 model

```
lambda    1e-10 | train MSE 0.0374 | test MSE    9.8478 | ||w||     11856.98
lambda    1e-06 | train MSE 0.0456 | test MSE    0.3750 | ||w||        28.64
lambda    1e-02 | train MSE 0.0612 | test MSE    0.1016 | ||w||         4.38
lambda    1e+00 | train MSE 0.1024 | test MSE    0.1785 | ||w||         0.46
lambda    1e+02 | train MSE 0.2309 | test MSE    0.4426 | ||w||         0.08
```

Read it top to bottom: the weight norm collapses, training error rises monotonically, test error drops by two orders of magnitude and then rises again once the model is forced too flat. The degree-15 model with `lambda = 0.01` is about as good as the cubic.

### Step 4: More data as a regularizer

Same degree-15 model, no penalty, more samples:

```
n=   20 | train MSE 0.0535 | test MSE 12.6172
n=   50 | train MSE 0.0405 | test MSE 0.0830
n= 1000 | train MSE 0.0629 | test MSE 0.0663
```

With 1000 points the train and test errors converge to the noise level (`0.25^2 = 0.0625`). The model did not change; the data stopped letting it memorize.

### Step 5: Early stopping

Plain gradient descent on the standardized degree-15 features, recording test error at checkpoints:

```python
weights = np.zeros(15)
intercept = train_y.mean()
for epoch in range(1, 200001):
    residual = scaled_train @ weights + intercept - train_y
    weights -= 0.05 * scaled_train.T @ residual / len(train_y)
```

```
epoch    100 | train MSE 0.1048 | test MSE 0.1826
epoch  10000 | train MSE 0.0706 | test MSE 0.1227
epoch 100000 | train MSE 0.0499 | test MSE 0.1548
epoch 200000 | train MSE 0.0463 | test MSE 0.2982
```

Training error keeps falling. Test error bottoms out and then rises. Stopping at the validation minimum is a free regularizer.

## Use It

In scikit-learn you chain the same three steps in a pipeline so the scaler is fitted only on training data.

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

pipeline = make_pipeline(
    PolynomialFeatures(15, include_bias=False),
    StandardScaler(),
    Ridge(alpha=0.01),
)
pipeline.fit(train_x[:, None], train_y)
```

`Ridge` calls the penalty `alpha`, does not penalize the intercept, and uses the same objective as your class. The test MSE matches yours to four decimals for every `alpha`. For L1 use `Lasso`, for both use `ElasticNet`, and for early stopping see `SGDRegressor(early_stopping=True)`.

## Exercises

1. **Easy.** Remove the standardization from `RidgePolynomial` (use raw `x, x^2, ..., x^15`) and rerun the lambda sweep. Which `lambda` is now best, and why does the sweep look so different?
2. **Medium.** Add a validation set of 20 points, pick `lambda` on it from a log-spaced grid of 30 values, and report the test MSE of the chosen model. How close is it to the test-optimal `lambda`?
3. **Hard.** Use the SVD `X = U S V^T` to show that ridge multiplies each component of the least-squares solution by `s_j^2 / (s_j^2 + lambda)`. Implement ridge through the SVD, verify it matches `np.linalg.solve`, and print the shrinkage factor for each singular value of the degree-15 design matrix.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Overfitting | "Memorizing the training data" | Low training error and high test error because the model fitted noise |
| Underfitting | "Too simple" | High training and test error because the model cannot represent the pattern |
| Generalization gap | "Train-test gap" | Test error minus training error |
| Regularization | "Penalize complexity" | Any modification that limits how closely a model can fit the training data, to improve test error |
| L2 / Ridge | "Weight decay" | Adding `lambda * ||w||^2` to the loss, shrinking all weights toward zero |
| L1 / Lasso | "Sparse regularization" | Adding `lambda * ||w||_1`, which sets some weights exactly to zero |
| Early stopping | "Stop before it overfits" | Halting iterative training at the epoch with the best validation error |
| Capacity | "Model complexity" | How wide a range of functions the model family can represent |

## Further Reading

- [An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/) -- section 6.2 covers ridge and lasso with clear plots of coefficient shrinkage
- [Christopher Bishop, Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) -- section 1.1 uses exactly this sine-curve example to introduce overfitting and regularization
- [scikit-learn: Linear Models (Ridge, Lasso, ElasticNet)](https://scikit-learn.org/stable/modules/linear_model.html) -- API and objective functions for the regularized linear models
- [StatQuest: Regularization Part 1, Ridge Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) -- visual intuition for the penalty term
