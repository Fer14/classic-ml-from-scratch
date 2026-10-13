# Phase 05: Supervised Regression

> Predicting a number is where machine learning starts: one matrix equation, then everything you need to make it robust.

This phase builds the linear regression family from first principles in NumPy. You start with the closed-form least squares solution and its geometry, solve the same problem with gradient descent, bend straight lines into curves with polynomial features, and then tame overfitting and collinearity with ridge, lasso and elastic net (including a coordinate descent solver written by hand). The phase ends with the diagnostics that tell you whether a fitted model can actually be trusted. Every lesson finishes by matching your implementation against scikit-learn.

**Prerequisites:** Phase 01 (Linear Algebra), Phase 02 (Calculus and Optimization), Phase 03 (Probability and Statistics), Phase 04 (ML Fundamentals)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Linear Regression: The Closed Form](01-linear-regression-closed-form/) | OLS via the normal equation and `lstsq`, with R^2 and RMSE on held-out data |
| 02 | [Linear Regression with Gradient Descent](02-linear-regression-gradient-descent/) | Batch and mini-batch gradient descent, the effect of scaling, conversion back to original units |
| 03 | [Polynomial Regression](03-polynomial-regression/) | Vandermonde features, a degree sweep with validation error, plots of under- and overfitting |
| 04 | [Ridge Regression](04-ridge-regression/) | Closed-form and SVD ridge, shrinkage factors, alpha chosen by k-fold cross-validation |
| 05 | [Lasso and Elastic Net](05-lasso-and-elastic-net/) | Coordinate descent with soft-thresholding, the lasso path, the grouping effect |
| 06 | [Regression Diagnostics](06-regression-diagnostics/) | Adjusted R^2, standard errors and p-values, leverage, Cook's distance, VIF, diagnostic plots |

Run the first lesson:

```bash
uv run python phases/05-supervised-regression/01-linear-regression-closed-form/code/main.py
```
