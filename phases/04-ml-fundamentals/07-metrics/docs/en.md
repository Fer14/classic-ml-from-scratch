# Evaluation Metrics

> Accuracy answers a question nobody asked. Pick the metric that matches the cost of each mistake.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 (Probability and Statistics), Phase 04 Lessons 02 and 03 (Train/Validation/Test, Loss Functions)
**Time:** ~60 minutes

## Learning Objectives

- Build a confusion matrix and compute accuracy, precision, recall, and F1 from it
- Explain why accuracy is misleading on imbalanced data and choose a better metric for a given cost structure
- Implement the ROC curve, ROC AUC, and average precision from sorted scores, and explain what each measures
- Compute MAE, RMSE, R^2, and MAPE and describe how each reacts to outliers

## The Problem

You build a fraud detector. It reports 90% accuracy. Your manager is happy. Then you notice that 90% of transactions are legitimate, so a model that says "not fraud" to everything also gets 90% accuracy, and it catches zero fraud.

Different mistakes cost different amounts. Missing a cancer diagnosis is far worse than ordering an unnecessary biopsy. Flagging a legitimate email as spam annoys a user; letting phishing through can cost a company. A single "percent correct" hides all of that. You need metrics that separate the kinds of errors and let you pick the trade-off on purpose.

## The Concept

### The confusion matrix

For a binary classifier, every prediction lands in one of four cells:

```
                      predicted 0         predicted 1
                  +-------------------+-------------------+
      actual 0    |  TN true negative |  FP false positive|
                  +-------------------+-------------------+
      actual 1    |  FN false negative|  TP true positive |
                  +-------------------+-------------------+
```

Every classification metric is a ratio of these four counts.

### The core metrics

```
accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)        "of what I flagged, how much was real?"
recall    = TP / (TP + FN)        "of what was real, how much did I flag?"
F1        = 2 * precision * recall / (precision + recall)
specificity = TN / (TN + FP)      recall of the negative class
FPR       = FP / (FP + TN) = 1 - specificity
```

- **Precision** matters when false positives are expensive: spam filters, recommending a costly intervention.
- **Recall** (sensitivity, true positive rate) matters when false negatives are expensive: disease screening, fraud.
- **F1** is the harmonic mean, which is low if either is low. Use it when you need one number and both errors matter.

### Thresholds turn scores into decisions

Most classifiers output a score or probability. The prediction is `score >= threshold`. Lowering the threshold flags more items: recall goes up, precision usually goes down. The threshold is a business decision, not a property of the model, and you should choose it on validation data.

```
threshold:  high  ------------------------------------>  low
flagged:    few                                         many
precision:  high                                        low
recall:     low                                         high
```

### Threshold-free metrics

To compare *models* independent of the threshold, sweep all thresholds.

**ROC curve**: plot TPR (recall) against FPR for every threshold. A random scorer traces the diagonal; a perfect one hugs the top-left corner. **ROC AUC** is the area under it, and has a clean probabilistic meaning:

```
AUC = P(score of a random positive > score of a random negative)
```

So AUC = 0.5 is random ranking and 1.0 is perfect ranking.

**Precision-recall curve**: plot precision against recall. **Average precision (AP)** summarizes it: the mean of precision at each rank where a positive appears. The baseline for AP is the positive rate, not 0.5. With 1% positives, an AUC of 0.95 can coexist with a precision so low the model is useless; the PR curve shows it, the ROC curve hides it. Prefer PR/AP when positives are rare and you care about them.

### Regression metrics

```
MAE  = mean(|y - y_hat|)                     same units as y, robust to outliers
RMSE = sqrt(mean((y - y_hat)^2))             same units as y, punishes large errors
R^2  = 1 - SS_res / SS_tot                   fraction of variance explained; 0 = predict the mean
MAPE = mean(|y - y_hat| / |y|) * 100         percent error; explodes when y is near 0
```

RMSE is always at least MAE. A large ratio RMSE/MAE means a few large errors dominate.

### Matching metric to problem

| Situation | Reasonable metric |
|---|---|
| Balanced classes, equal error costs | Accuracy |
| Rare positives, you must catch them | Recall at a fixed precision, or PR AUC |
| False alarms are expensive | Precision at a fixed recall |
| Need calibrated probabilities | Log loss (Lesson 03), Brier score |
| Ranking quality only | ROC AUC |
| Regression, outliers are noise | MAE |
| Regression, large errors are disasters | RMSE |

## Build It

### Step 1: Imbalanced scores

You simulate 2000 examples with about 10% positives. Positive logits are drawn around +1.5 and negative logits around -1.0, then passed through a sigmoid.

```python
def make_scores(n_samples, positive_rate, rng):
    labels = (rng.uniform(size=n_samples) < positive_rate).astype(int)
    logits = np.where(labels == 1, rng.normal(1.5, 1.0, n_samples), rng.normal(-1.0, 1.0, n_samples))
    return labels, 1 / (1 + np.exp(-logits))
```

The "always predict 0" baseline scores accuracy 0.9025 with precision, recall and F1 all 0.

### Step 2: Confusion matrix and core metrics

```python
def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def classification_report(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
```

At threshold 0.5:

```
             pred 0   pred 1
actual 0      1516      289
actual 1        12      183
{'accuracy': 0.8495, 'precision': 0.3877, 'recall': 0.9385, 'f1': 0.5487}
```

This model has *lower* accuracy than the trivial baseline and is far more useful: it catches 94% of positives.

### Step 3: Sweep the threshold

```
threshold 0.20 | precision 0.141 | recall 1.000 | F1 0.247
threshold 0.50 | precision 0.388 | recall 0.938 | F1 0.549
threshold 0.65 | precision 0.608 | recall 0.795 | F1 0.689
threshold 0.80 | precision 0.934 | recall 0.508 | F1 0.658
```

Same model, very different behavior. Of these thresholds, F1 peaks at 0.65, not at the default 0.5.

### Step 4: ROC AUC and average precision

Sort by score, descending. Walking down the list, each positive moves the curve up and each negative moves it right.

```python
def roc_curve(y_true, scores):
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = y_true[order]
    true_positives = np.concatenate([[0], np.cumsum(sorted_labels)])
    false_positives = np.concatenate([[0], np.cumsum(1 - sorted_labels)])
    return false_positives / false_positives[-1], true_positives / true_positives[-1]


def area_under_curve(x, y):
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def average_precision(y_true, scores):
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = y_true[order]
    hits = np.cumsum(sorted_labels)
    precision_at_k = hits / np.arange(1, len(scores) + 1)
    return float(np.sum(precision_at_k * sorted_labels) / sorted_labels.sum())
```

Result: ROC AUC 0.9585, average precision 0.7937 against a baseline of 0.098. (This simple ROC treats every score as a separate threshold, which is exact when there are no tied scores.)

### Step 5: Regression metrics

```python
def regression_report(y_true, y_pred):
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    r2 = 1 - np.sum(errors ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    mape = np.mean(np.abs(errors / y_true)) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAPE%": float(mape)}
```

With 3 of 300 predictions off by 400: MAE 27.4, RMSE 49.6. Three bad predictions nearly doubled RMSE relative to MAE.

## Use It

```python
from sklearn import metrics

metrics.confusion_matrix(labels, predictions)
metrics.precision_score(labels, predictions)
metrics.recall_score(labels, predictions)
metrics.f1_score(labels, predictions)
metrics.roc_auc_score(labels, scores)
metrics.average_precision_score(labels, scores)
metrics.mean_absolute_error(actual, predicted)
metrics.r2_score(actual, predicted)
metrics.mean_absolute_percentage_error(actual, predicted)
```

Every value matches yours to four decimals. `metrics.classification_report` prints per-class precision, recall and F1 in one table, and `roc_curve` / `precision_recall_curve` return the full curves. The script plots both curves to `code/roc_pr.png`. Note that sklearn's confusion matrix puts actual classes on rows and predicted on columns, ordered `[[TN, FP], [FN, TP]]`, the same layout as above.

## Exercises

1. **Easy.** Find the threshold that maximizes F1 by sweeping 101 values between 0 and 1. Then find the lowest threshold that keeps precision at or above 0.8. Report recall at each.
2. **Medium.** Verify the probabilistic meaning of AUC: sample 100,000 random (positive, negative) pairs and compute the fraction where the positive has the higher score. Compare with `area_under_curve`.
3. **Hard.** Implement macro-, micro- and weighted-averaged precision, recall and F1 for a 3-class problem from a 3x3 confusion matrix. Construct an example where macro-F1 and micro-F1 disagree sharply and explain which one a product team should report.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Confusion matrix | "The error table" | Counts of TP, FP, FN, TN for a given threshold |
| Precision | "How many alarms are real" | TP / (TP + FP) |
| Recall | "Sensitivity" or "hit rate" | TP / (TP + FN), the true positive rate |
| F1 score | "Balance of precision and recall" | Harmonic mean of precision and recall |
| ROC AUC | "Area under the curve" | Probability that a random positive is scored above a random negative |
| Average precision | "PR AUC" | Mean precision at the rank of each positive; baseline equals the positive rate |
| Threshold | "Cutoff" | The score above which an example is predicted positive |
| RMSE | "Root mean squared error" | Square root of MSE, in the units of the target, sensitive to large errors |
| R^2 | "Variance explained" | 1 minus residual sum of squares over total sum of squares |

## Further Reading

- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) -- definitions and usage for every metric in this lesson
- [Tom Fawcett, "An introduction to ROC analysis"](https://www.sciencedirect.com/science/article/pii/S016786550500303X) -- the standard tutorial on ROC curves and AUC
- [Saito and Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- why PR curves matter for rare positives
- [StatQuest: ROC and AUC, Clearly Explained](https://www.youtube.com/watch?v=4jRBRDbJemM) -- visual walk-through of building the ROC curve
