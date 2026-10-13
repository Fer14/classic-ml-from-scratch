import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def k_fold_indices(n_samples, n_folds, rng=None):
    indices = np.arange(n_samples) if rng is None else rng.permutation(n_samples)
    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[: n_samples % n_folds] += 1
    folds, start = [], 0
    for size in fold_sizes:
        validation = indices[start:start + size]
        train = np.concatenate([indices[:start], indices[start + size:]])
        folds.append((train, validation))
        start += size
    return folds


def stratified_k_fold_indices(labels, n_folds, rng):
    fold_members = [[] for _ in range(n_folds)]
    for label in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == label))
        for position, index in enumerate(members):
            fold_members[position % n_folds].append(index)
    all_indices = np.arange(len(labels))
    folds = []
    for members in fold_members:
        validation = np.array(sorted(members))
        folds.append((np.setdiff1d(all_indices, validation), validation))
    return folds


class StandardizedRidge:
    def __init__(self, penalty):
        self.penalty = penalty

    def fit(self, features, targets):
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        scaled = (features - self.mean) / self.std
        self.intercept = targets.mean()
        gram = scaled.T @ scaled + self.penalty * np.eye(features.shape[1])
        self.weights = np.linalg.solve(gram, scaled.T @ (targets - self.intercept))
        return self

    def predict(self, features):
        return ((features - self.mean) / self.std) @ self.weights + self.intercept


def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def cross_validate(features, targets, penalty, folds):
    scores = []
    for train, validation in folds:
        model = StandardizedRidge(penalty).fit(features[train], targets[train])
        scores.append(r2_score(targets[validation], model.predict(features[validation])))
    return np.array(scores)


def main():
    features, targets = load_diabetes(return_X_y=True)
    print(f"Diabetes dataset: {features.shape[0]} samples, {features.shape[1]} features")

    print("\n=== Step 1: k-fold indices ===")
    folds = k_fold_indices(len(targets), 5, np.random.default_rng(0))
    for number, (train, validation) in enumerate(folds):
        print(f"fold {number}: train={len(train)} validation={len(validation)} first validation ids={validation[:4]}")
    every_index = np.sort(np.concatenate([validation for _, validation in folds]))
    print(f"each sample validated exactly once: {np.array_equal(every_index, np.arange(len(targets)))}")

    print("\n=== Step 2: a single split is noisy ===")
    single_split_scores = []
    for seed in range(20):
        order = np.random.default_rng(seed).permutation(len(targets))
        train, validation = order[:353], order[353:]
        model = StandardizedRidge(1.0).fit(features[train], targets[train])
        single_split_scores.append(r2_score(targets[validation], model.predict(features[validation])))
    cv_means = [cross_validate(features, targets, 1.0, k_fold_indices(len(targets), 5, np.random.default_rng(seed))).mean() for seed in range(20)]
    print(f"20 random 80/20 splits: R^2 range [{min(single_split_scores):.3f}, {max(single_split_scores):.3f}], std {np.std(single_split_scores):.3f}")
    print(f"20 shuffled 5-fold CVs: R^2 range [{min(cv_means):.3f}, {max(cv_means):.3f}], std {np.std(cv_means):.3f}")

    print("\n=== Step 3: choose lambda with 5-fold CV ===")
    penalties = [0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    cv_results = {}
    for penalty in penalties:
        scores = cross_validate(features, targets, penalty, folds)
        cv_results[penalty] = scores.mean()
        print(f"lambda {penalty:7.2f} | mean R^2 {scores.mean():.4f} | std {scores.std():.4f}")
    best_penalty = max(cv_results, key=cv_results.get)
    print(f"best lambda: {best_penalty}")

    print("\n=== Step 4: leave-one-out ===")
    loo_folds = k_fold_indices(len(targets), len(targets))
    squared_errors = []
    for train, validation in loo_folds:
        model = StandardizedRidge(best_penalty).fit(features[train], targets[train])
        squared_errors.append((model.predict(features[validation])[0] - targets[validation][0]) ** 2)
    print(f"LOO fits: {len(loo_folds)}, LOO MSE: {np.mean(squared_errors):.2f}")

    print("\n=== Step 5: stratified folds keep class balance ===")
    labels = (targets > np.quantile(targets, 0.85)).astype(int)
    plain_rates = [labels[validation].mean() for _, validation in k_fold_indices(len(labels), 5)]
    stratified_rates = [labels[validation].mean() for _, validation in stratified_k_fold_indices(labels, 5, np.random.default_rng(0))]
    print(f"overall positive rate: {labels.mean():.3f}")
    print(f"unshuffled k-fold rates: {np.round(plain_rates, 3)}")
    print(f"stratified k-fold rates: {np.round(stratified_rates, 3)}")

    print("\n=== Use It: scikit-learn ===")
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    sk_scores = cross_val_score(pipeline, features, targets, cv=KFold(5, shuffle=False), scoring="r2")
    ours = cross_validate(features, targets, 1.0, k_fold_indices(len(targets), 5))
    print(f"sklearn cross_val_score R^2: {np.round(sk_scores, 4)}")
    print(f"ours (same unshuffled folds): {np.round(ours, 4)}")
    search = GridSearchCV(pipeline, {"ridge__alpha": penalties}, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    search.fit(features, targets)
    print(f"GridSearchCV best alpha: {search.best_params_['ridge__alpha']}, best mean R^2: {search.best_score_:.4f}")
    strat_rates = [labels[validation].mean() for _, validation in StratifiedKFold(5, shuffle=True, random_state=0).split(features, labels)]
    print(f"sklearn StratifiedKFold rates: {np.round(strat_rates, 3)}")


if __name__ == "__main__":
    main()
