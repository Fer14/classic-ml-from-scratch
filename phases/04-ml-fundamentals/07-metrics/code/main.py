from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def make_scores(n_samples, positive_rate, rng):
    labels = (rng.uniform(size=n_samples) < positive_rate).astype(int)
    logits = np.where(labels == 1, rng.normal(1.5, 1.0, n_samples), rng.normal(-1.0, 1.0, n_samples))
    return labels, 1 / (1 + np.exp(-logits))


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


def regression_report(y_true, y_pred):
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    r2 = 1 - np.sum(errors ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    mape = np.mean(np.abs(errors / y_true)) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAPE%": float(mape)}


def main():
    rng = np.random.default_rng(7)
    labels, scores = make_scores(2000, 0.1, rng)
    print(f"{len(labels)} samples, positive rate {labels.mean():.3f}")

    print("\n=== Step 1: accuracy lies on imbalanced data ===")
    always_negative = np.zeros_like(labels)
    print(f"always predict 0: {classification_report(labels, always_negative)}")

    print("\n=== Step 2: confusion matrix at threshold 0.5 ===")
    predictions = (scores >= 0.5).astype(int)
    tp, fp, fn, tn = confusion_counts(labels, predictions)
    print("             pred 0   pred 1")
    print(f"actual 0   {tn:7d}  {fp:7d}")
    print(f"actual 1   {fn:7d}  {tp:7d}")
    report = classification_report(labels, predictions)
    print({name: round(value, 4) for name, value in report.items()})

    print("\n=== Step 3: the threshold trades precision for recall ===")
    for threshold in (0.2, 0.35, 0.5, 0.65, 0.8):
        report = classification_report(labels, (scores >= threshold).astype(int))
        print(f"threshold {threshold:.2f} | precision {report['precision']:.3f} | recall {report['recall']:.3f} | F1 {report['f1']:.3f}")

    print("\n=== Step 4: threshold-free metrics ===")
    fpr, tpr = roc_curve(labels, scores)
    roc_auc = area_under_curve(fpr, tpr)
    ap = average_precision(labels, scores)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average precision: {ap:.4f}  (random baseline = positive rate = {labels.mean():.3f})")

    print("\n=== Step 5: regression metrics ===")
    actual = rng.uniform(50, 500, 300)
    predicted = actual + rng.normal(0, 30, 300)
    predicted[:3] += 400
    ours = regression_report(actual, predicted)
    print({name: round(value, 4) for name, value in ours.items()})

    print("\n=== Use It: scikit-learn ===")
    print(f"accuracy  {metrics.accuracy_score(labels, predictions):.4f}")
    print(f"precision {metrics.precision_score(labels, predictions):.4f}")
    print(f"recall    {metrics.recall_score(labels, predictions):.4f}")
    print(f"f1        {metrics.f1_score(labels, predictions):.4f}")
    print(f"confusion matrix:\n{metrics.confusion_matrix(labels, predictions)}")
    print(f"ROC AUC   {metrics.roc_auc_score(labels, scores):.4f}")
    print(f"AP        {metrics.average_precision_score(labels, scores):.4f}")
    print(f"MAE {metrics.mean_absolute_error(actual, predicted):.4f}  RMSE {np.sqrt(metrics.mean_squared_error(actual, predicted)):.4f}  R2 {metrics.r2_score(actual, predicted):.4f}  MAPE% {100 * metrics.mean_absolute_percentage_error(actual, predicted):.4f}")

    precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels, scores)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend()
    axes[1].plot(recall_curve, precision_curve, label=f"AP={ap:.3f}")
    axes[1].axhline(labels.mean(), linestyle="--", color="gray")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend()
    figure.tight_layout()
    output_path = Path(__file__).parent / "roc_pr.png"
    figure.savefig(output_path, dpi=100)
    print(f"\nSaved {output_path.name}")


if __name__ == "__main__":
    main()
