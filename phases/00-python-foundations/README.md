# Phase 00: Python Foundations

> The tools you will hold for the rest of the course: a reproducible environment, plain Python, NumPy, pandas, matplotlib, and notebooks.

Every algorithm in this course is built from scratch, and that only works if the language and its numeric stack feel invisible. This phase sets up a locked `uv` environment, then builds small but real things with each tool: summary statistics with loops, a bag-of-words matrix with built-in containers, a scikit-learn-style estimator class, vectorized pairwise distances, a group-by and a join, a histogram, and a tiny notebook kernel. Each lesson implements something by hand first and then checks it against the library you will use from Phase 01 onward.

**Prerequisites:** None. Basic comfort with a terminal helps.

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Dev Environment](01-dev-environment/) | An environment checker for Python version, venv status, and pinned package versions |
| 02 | [Python Basics](02-python-basics/) | Mean, median, variance, and a Newton's-method square root with loops, checked against `statistics` |
| 03 | [Data Structures](03-data-structures/) | A bag-of-words count matrix from raw text, checked against `CountVectorizer` |
| 04 | [Functions and Classes](04-functions-and-classes/) | Baseline estimators with `fit`/`predict`/`score`, checked against `DummyRegressor` and `DummyClassifier` |
| 05 | [NumPy](05-numpy/) | Vectorized standardization and pairwise distances with broadcasting, checked against `StandardScaler` and `cdist` |
| 06 | [Pandas](06-pandas/) | Fill-missing, group-by, and left join in plain Python, reproduced with `fillna`, `groupby`, and `merge` |
| 07 | [Plotting with Matplotlib](07-plotting-with-matplotlib/) | A from-scratch histogram and a four-panel figure of loss curves, scatter, histogram, and bars |
| 08 | [Jupyter Notebooks](08-jupyter-notebooks/) | A notebook built as JSON and a mini kernel, checked against a real `ipykernel` via `nbclient` |

Run the first lesson:

```
uv run python phases/00-python-foundations/01-dev-environment/code/main.py
```
