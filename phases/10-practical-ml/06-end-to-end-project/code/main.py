import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def load_data():
    data = load_breast_cancer()
    y_malignant = (data.target == 0).astype(int)
    return data.data, y_malignant, list(data.feature_names)


class ScratchScaledLogistic:
    def __init__(self, l2=0.01, learning_rate=0.5, epochs=1500):
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-12
        Z = (X - self.mean) / self.std
        self.w = np.zeros(Z.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
            error = p - y
            self.w -= self.learning_rate * (Z.T @ error / len(y) + self.l2 * self.w)
            self.b -= self.learning_rate * error.mean()
        return self

    def predict_proba(self, X):
        Z = (X - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def stratified_folds(y, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(n_folds)]


def recall_at(y_true, y_pred):
    return np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)


def scratch_cv(l2, X, y, folds):
    accuracies, recalls = [], []
    for train_idx, test_idx in folds:
        model = ScratchScaledLogistic(l2=l2).fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        accuracies.append(np.mean(y_pred == y[test_idx]))
        recalls.append(recall_at(y[test_idx], y_pred))
    return float(np.mean(accuracies)), float(np.mean(recalls))


def out_of_fold_probabilities(make_model, X, y, folds):
    probabilities = np.zeros(len(y))
    for train_idx, test_idx in folds:
        probabilities[test_idx] = make_model().fit(X[train_idx], y[train_idx]).predict_proba(X[test_idx])[:, 1]
    return probabilities


def lowest_threshold_for_recall(y_true, probabilities, target_recall):
    for threshold in np.round(np.arange(0.95, 0.04, -0.01), 2):
        if recall_at(y_true, (probabilities >= threshold).astype(int)) >= target_recall:
            return float(threshold)
    return 0.05


def main():
    X, y, feature_names = load_data()
    print("=== 1. Frame the problem ===")
    print(f"Wisconsin breast cancer: {X.shape[0]} tumors, {X.shape[1]} numeric features")
    print(f"positive class = malignant: {y.sum()} ({y.mean():.1%})")
    print("cost asymmetry: a missed malignant tumor (FN) is far worse than a false alarm (FP)")

    print("\n=== 2. Lock away a test set ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    print(f"train {len(y_train)} / test {len(y_test)}; test is not touched until step 7")

    print("\n=== 3. Quick look at the data ===")
    spreads = X_train.std(axis=0)
    print(f"feature std ranges from {spreads.min():.4f} ({feature_names[spreads.argmin()]}) to {spreads.max():.1f} ({feature_names[spreads.argmax()]})")
    print("-> scaling is required for linear models and gradient descent")
    correlations = np.corrcoef(X_train, rowvar=False)
    upper = np.triu(np.abs(correlations), k=1)
    print(f"feature pairs with |corr| > 0.95: {int(np.sum(upper > 0.95))} -> strong redundancy")

    print("\n=== 4. Baselines ===")
    folds = stratified_folds(y_train)
    majority_accuracy = 1 - y_train.mean()
    print(f"always predict benign: accuracy={majority_accuracy:.3f} recall=0.000")

    print("\n=== 5. From-scratch model, tuned with CV ===")
    best_l2, best_accuracy = None, -1.0
    for l2 in [0.0, 0.001, 0.01, 0.1, 1.0]:
        accuracy, recall = scratch_cv(l2, X_train, y_train, folds)
        print(f"  scratch logistic l2={l2:<6} cv accuracy={accuracy:.3f} cv recall={recall:.3f}")
        if accuracy > best_accuracy:
            best_l2, best_accuracy = l2, accuracy
    print(f"  chosen l2={best_l2}")

    print("\n=== 6. Compare against library models (same folds) ===")
    cv = [(train, test) for train, test in folds]
    candidates = {
        "logistic (sklearn)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1),
        "hist gradient boosting": HistGradientBoostingClassifier(random_state=0),
    }
    for name, candidate in candidates.items():
        scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"  {name:<24} cv ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        {"logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=cv,
        scoring="roc_auc",
    ).fit(X_train, y_train)
    print(f"  tuned logistic: C={search.best_params_['logisticregression__C']} cv ROC-AUC = {search.best_score_:.4f}")
    chosen_C = search.best_params_["logisticregression__C"]

    def make_final_model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=chosen_C, max_iter=1000))

    oof_probabilities = out_of_fold_probabilities(make_final_model, X_train, y_train, folds)
    threshold = lowest_threshold_for_recall(y_train, oof_probabilities, target_recall=0.98)
    print(f"  threshold for >= 98% out-of-fold recall: {threshold:.2f}")

    print("\n=== 7. Final evaluation on the untouched test set ===")
    final_model = make_final_model().fit(X_train, y_train)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    scratch_final = ScratchScaledLogistic(l2=best_l2).fit(X_train, y_train)
    for name, y_pred in [
        ("scratch logistic @0.50", scratch_final.predict(X_test)),
        ("sklearn logistic @0.50", (test_probabilities >= 0.5).astype(int)),
        (f"sklearn logistic @{threshold:.2f}", (test_probabilities >= threshold).astype(int)),
    ]:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(
            f"  {name:<24} acc={np.mean(y_pred == y_test):.3f} precision={precision_score(y_test, y_pred):.3f} "
            f"recall={recall_score(y_test, y_pred):.3f} | TP={tp} FP={fp} FN={fn} TN={tn}"
        )
    print(f"  test ROC-AUC = {roc_auc_score(y_test, test_probabilities):.4f}")

    print("\n=== 8. What does the model use? ===")
    importance = permutation_importance(final_model, X_test, y_test, n_repeats=20, random_state=0, scoring="roc_auc")
    for j in np.argsort(-importance.importances_mean)[:5]:
        print(f"  {feature_names[j]:<24} drop in ROC-AUC = {importance.importances_mean[j]:.4f}")
    print("  (correlated features share credit, so individual importances look small)")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
