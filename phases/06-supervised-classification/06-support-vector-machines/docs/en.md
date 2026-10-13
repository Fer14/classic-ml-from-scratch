# Support Vector Machines

> Of all the lines that separate your classes, pick the one that stays as far as possible from both.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Linear Algebra), Phase 02 (Calculus and Optimization), Phase 04 (ML Fundamentals), Phase 06 Lesson 05 (Perceptron)
**Time:** ~75 minutes

## Learning Objectives

- Explain the maximum-margin principle and derive why the margin width equals `2 / ||w||`
- Write the soft-margin SVM objective as regularization plus hinge loss, and implement it with subgradient descent in NumPy
- Identify support vectors and explain why only they determine the decision boundary
- Choose the regularization parameter `C` by reasoning about the margin width versus training errors trade-off

## The Problem

The perceptron stops at the first line that separates the training data. On separable data there are infinitely many such lines, and the perceptron picks one more or less at random. Some of them pass a hair away from a training point. A new point that lands slightly off from its neighbors will be misclassified.

A more robust choice is the line with the widest empty corridor around it. That corridor gives you a safety buffer against noise. This is the idea behind the support vector machine (SVM), introduced by Cortes and Vapnik in 1995, which was the state-of-the-art classifier for a decade and is still a strong baseline for small and medium tabular or text datasets.

## The Concept

### Margin geometry

A linear classifier predicts `sign(w . x + b)`. The distance from a point `x` to the hyperplane `w . x + b = 0` is:

```
distance = |w . x + b| / ||w||
```

You can rescale `w` and `b` by any positive constant without moving the hyperplane. SVMs fix that freedom by requiring the closest points on each side to satisfy `y_i (w . x_i + b) = 1`. Then the two dashed "gutter" lines are `w . x + b = +1` and `w . x + b = -1`, and the full width of the corridor is:

```
margin width = 2 / ||w||
```

```
            w.x+b = -1     w.x+b = 0     w.x+b = +1
                 :             |              :
     -   -       :             |              :      +    +
        -    -  (-)            |             (+)  +     +
     -      -    :             |              :     +
                 :<---- 2/||w|| ---->:
       support vector                     support vector
```

Maximizing the margin is the same as minimizing `||w||^2`.

### Hard margin

When the data is separable:

```
minimize    (1/2) ||w||^2
subject to  y_i (w . x_i + b) >= 1   for every i
```

This is a convex quadratic program with a unique solution.

### Soft margin and the hinge loss

Real data overlaps, so the hard-margin problem has no solution. You allow each point a slack `xi_i >= 0`, paid at rate `C`:

```
minimize    (1/2) ||w||^2 + C * sum_i xi_i
subject to  y_i (w . x_i + b) >= 1 - xi_i,   xi_i >= 0
```

At the optimum each slack is exactly `max(0, 1 - y_i f(x_i))`. Substituting gives an unconstrained objective:

```
J(w, b) = (1/2) ||w||^2 + C * sum_i max(0, 1 - y_i (w . x_i + b))
```

The second term is the **hinge loss**. Compare it to the other losses you have seen, as a function of the signed margin `m = y f(x)`:

| m = y f(x) | 0/1 loss | hinge `max(0, 1-m)` | logistic `log(1+e^-m)` |
|-----------|----------|-----------------|---------------------|
| 2.0 | 0 | 0 | 0.13 |
| 1.0 | 0 | 0 | 0.31 |
| 0.5 | 0 | 0.5 | 0.47 |
| 0.0 | 1 | 1 | 0.69 |
| -1.0 | 1 | 2 | 1.31 |

The hinge loss is exactly zero once a point is beyond the margin. That is where SVM sparsity comes from: points with `m > 1` contribute nothing to the gradient and nothing to the solution.

### Support vectors

The points with `y_i f(x_i) <= 1` (on the gutter or inside it, including misclassified ones) are the **support vectors**. Delete any other point and retrain: the boundary does not move. The solution can be written as `w = sum_i alpha_i y_i x_i` with `alpha_i > 0` only for support vectors. You will use this form in the next lesson to unlock kernels.

### The role of C

`C` sets the price of a margin violation.

- Small `C`: violations are cheap, the optimizer prefers a small `||w||`, so the margin is wide and many points are inside it. More bias, less variance.
- Large `C`: violations are expensive, the margin shrinks to classify more training points correctly. Less bias, more variance. As `C -> infinity` you recover the hard-margin SVM.

`C` plays the role of `1 / lambda` in ridge regression (Phase 05). Tune it with cross-validation, and always standardize features first, because the margin is measured in Euclidean distance.

### Optimizing it

The objective is convex but not differentiable at `m = 1`. A **subgradient** is enough. For the set `V` of points with `y_i f(x_i) < 1`:

```
dJ/dw = w - C * sum_{i in V} y_i x_i
dJ/db =   - C * sum_{i in V} y_i
```

Subgradient descent with a decaying step `eta_t = eta_0 / sqrt(t+1)` converges to the optimum. Libraries use faster solvers: LIBLINEAR uses coordinate descent on the dual, LIBSVM uses SMO (Lesson 07).

## Build It

### Step 1: Data and scaling

Two overlapping Gaussian blobs, standardized with statistics from the training split only.

```python
X, y = make_two_classes(n_per_class=150, separation=1.2, seed=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

### Step 2: The objective

```python
def hinge_objective(X, y, weights, bias, C):
    margins = y * (X @ weights + bias)
    return 0.5 * weights @ weights + C * np.sum(np.maximum(0.0, 1.0 - margins))
```

### Step 3: Subgradient descent

Each epoch finds the margin violators, takes the subgradient, and steps with a decaying learning rate. Dividing by `n_samples` keeps the step size independent of dataset size.

```python
class LinearSVM:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []
        for epoch in range(self.epochs):
            margins = y * self.decision_function(X)
            violators = margins < 1.0
            grad_weights = self.weights - self.C * (y[violators] @ X[violators])
            grad_bias = -self.C * np.sum(y[violators])
            step = self.learning_rate / np.sqrt(epoch + 1)
            self.weights -= step * grad_weights / n_samples
            self.bias -= step * grad_bias / n_samples
            self.history.append(hinge_objective(X, y, self.weights, self.bias, self.C))
        return self

    def support_vector_mask(self, X, y, tolerance=1e-2):
        return y * self.decision_function(X) <= 1.0 + tolerance
```

With `C=1`, `learning_rate=5` and 20,000 full-batch epochs (under a second on 210 points):

```
  epoch    0 | objective 27.971
  epoch  100 | objective 19.829
  epoch 1000 | objective 18.441
  epoch 19999 | objective 18.440
Margin width 2/||w||: 0.655
Points on or inside the margin: 24
Test accuracy:  0.922
```

### Step 4: Sweep C

```python
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model = LinearSVM(C=C, learning_rate=5.0, epochs=20000).fit(X_train, y_train)
    norm = np.linalg.norm(model.weights)
    inside = model.support_vector_mask(X_train, y_train).sum()
```

```
       C    ||w||   margin  in margin  test acc
    0.01    0.835    2.395        125     0.911
     0.1    1.557    1.285         49     0.922
     1.0    3.051    0.655         24     0.922
    10.0    4.392    0.455         15     0.911
   100.0    4.928    0.406         13     0.922
```

As `C` grows, `||w||` grows, the margin shrinks, and fewer points end up inside it. Test accuracy barely moves here because the problem is nearly linear and low-dimensional; on high-dimensional data the choice of `C` matters much more.

## Use It

scikit-learn has two linear SVM implementations. `LinearSVC` wraps LIBLINEAR and scales to millions of samples. `SVC(kernel="linear")` wraps LIBSVM, exposes the support vectors, and also supports kernels.

```python
from sklearn.svm import LinearSVC, SVC

sk_linear = LinearSVC(C=1.0, loss="hinge", max_iter=20000, random_state=0).fit(X_train, y_train)
sk_svc = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
print(sk_svc.coef_, sk_svc.intercept_, len(sk_svc.support_))
```

```
Scratch  w = [2.344 1.953], b = -0.094
LinearSVC w = [2.345 1.953], b = -0.094
SVC(linear) w = [2.344 1.953], b = -0.094
SVC support vectors: 24
Scratch objective: 18.440
SVC objective:     18.441
```

All three solve the same convex problem and land on the same hyperplane, with the same 24 support vectors. Notes for practice: `LinearSVC` defaults to `loss="squared_hinge"` and penalizes the intercept, so its coefficients differ slightly unless you set `loss="hinge"`. SVMs output scores, not probabilities; use `SVC(probability=True)` or `CalibratedClassifierCV` if you need them.

## Exercises

1. **Easy.** Remove every training point that is not a support vector (`support_vector_mask` is False), retrain, and confirm `w` and `b` barely change. Then remove one support vector and see the boundary move.
2. **Medium.** Implement **Pegasos** (Shalev-Shwartz et al., 2007): stochastic subgradient descent on `(lambda/2)||w||^2 + mean hinge` with one random sample per step and `eta_t = 1 / (lambda t)`. Compare its convergence (objective vs number of samples touched) with the full-batch version here.
3. **Hard.** Solve the hard-margin primal as a quadratic program with `scipy.optimize.minimize(method="SLSQP")` using the constraints `y_i (w . x_i + b) >= 1` on a separable dataset. Check that the minimum `y_i f(x_i)` is 1 and that the margin is larger than the one found by the perceptron from Lesson 05.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Support vector machine | "Max-margin classifier" | A linear classifier trained by minimizing `(1/2)||w||^2 + C * sum hinge loss` |
| Margin | "The street between classes" | The width `2 / ||w||` of the region between the hyperplanes `w . x + b = +1` and `-1` |
| Support vectors | "The points that matter" | Training points with `y_i f(x_i) <= 1`; the solution is a weighted sum of only these points |
| Hinge loss | "SVM loss" | `max(0, 1 - y f(x))`: zero for points beyond the margin, linear penalty otherwise |
| Soft margin | "Allow some mistakes" | Adding slack variables so the problem has a solution when classes overlap |
| C | "Regularization strength" | Cost per unit of margin violation; it is inverse regularization, so large C means a narrower margin and less regularization |
| Subgradient | "Gradient of a kink" | Any vector that lower-bounds a convex function at a non-differentiable point; enough to run descent |
| Hard margin | "No errors allowed" | The SVM with `C -> infinity`; only solvable on linearly separable data |

## Further Reading

- [Cortes and Vapnik, Support-Vector Networks (1995)](https://link.springer.com/article/10.1007/BF00994018) -- the paper that introduced the soft-margin SVM
- [Stanford CS229 Lecture Notes, Support Vector Machines](https://cs229.stanford.edu/main_notes.pdf) -- Andrew Ng's derivation of the margin, the primal, and the dual
- [StatQuest, Support Vector Machines Part 1](https://www.youtube.com/watch?v=efR1C6CvhmE) -- visual explanation of margins, soft margins and support vectors
- [scikit-learn User Guide, Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) -- practical notes on LinearSVC, SVC, scaling, and choosing C
