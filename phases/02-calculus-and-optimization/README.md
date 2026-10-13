# Phase 02: Calculus and Optimization

> Training a model means walking downhill on a loss surface. This phase teaches you to find the slope and take the step.

Every model in this course is fit by minimizing a loss. This phase builds the tools for that from the ground up: derivatives as slopes, gradients as the direction of steepest ascent, the chain rule as the engine behind backpropagation, and gradient descent as the algorithm that uses them. You then study what makes optimization fast or slow (the learning rate and the condition number), when it is guaranteed to find the best answer (convexity), and how to scale it to large datasets (stochastic gradient descent and momentum). Each lesson builds the method from scratch in NumPy and then checks it against SciPy or scikit-learn.

**Prerequisites:** Phase 00 (Python Foundations), Phase 01 (Linear Algebra)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Derivatives](01-derivatives/) | Forward and central difference derivatives, an error-vs-step-size study, and tangent-line approximations |
| 02 | [Partial Derivatives and Gradients](02-partial-derivatives-and-gradients/) | Numerical gradients, directional derivatives, and a gradient-checked vectorized MSE gradient |
| 03 | [Chain Rule](03-chain-rule/) | Manual backprop through logistic loss, a tiny reverse-mode autodiff `Value` class, and logistic regression trained with it |
| 04 | [Gradient Descent](04-gradient-descent/) | A general gradient descent optimizer that fits linear regression and matches the closed-form solution |
| 05 | [Learning Rate](05-learning-rate/) | The `2 / lambda_max` stability limit, learning rate schedules, and backtracking line search |
| 06 | [Convexity](06-convexity/) | A numerical chord test, Hessian eigenvalue certificates, and local vs global optimization on a non-convex function |
| 07 | [SGD and Momentum](07-sgd-and-momentum/) | Mini-batch SGD with shuffling and decay, heavy-ball and Nesterov momentum, compared with `SGDRegressor` |

Run the first lesson:

```bash
uv run python phases/02-calculus-and-optimization/01-derivatives/code/main.py
```
