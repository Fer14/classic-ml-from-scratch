import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


class MeanImputer:
    def fit(self, X, y=None):
        self.means = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        return np.where(np.isnan(X), self.means, X)


class Standardizer:
    def fit(self, X, y=None):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


class OneHot:
    def fit(self, X, y=None):
        self.categories = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        blocks = [(X[:, [j]] == cats[None, :]).astype(float) for j, cats in enumerate(self.categories)]
        return np.hstack(blocks)


class TopKCorrelation:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        scores = np.abs(X_centered.T @ y_centered) / (np.linalg.norm(X_centered, axis=0) * np.linalg.norm(y_centered) + 1e-12)
        self.selected = np.argsort(-scores)[: self.k]
        return self

    def transform(self, X):
        return X[:, self.selected]


class LogisticModel:
    def __init__(self, learning_rate=0.5, epochs=500, l2=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            error = self.predict_proba(X) - y
            self.w -= self.learning_rate * (X.T @ error / len(y) + self.l2 * self.w)
            self.b -= self.learning_rate * error.mean()
        return self

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        for _, transformer in self.steps[:-1]:
            X = transformer.fit(X, y).transform(X)
        self.steps[-1][1].fit(X, y)
        return self

    def transform(self, X):
        for _, transformer in self.steps:
            X = transformer.transform(X)
        return X

    def predict(self, X):
        for _, transformer in self.steps[:-1]:
            X = transformer.transform(X)
        return self.steps[-1][1].predict(X)


class ColumnSplitter:
    def __init__(self, branches):
        self.branches = branches

    def fit(self, df, y=None):
        for _, transformer, columns in self.branches:
            transformer.fit(df[columns].to_numpy(), y)
        return self

    def transform(self, df):
        return np.hstack([transformer.transform(df[columns].to_numpy()) for _, transformer, columns in self.branches])


def make_folds(n_samples, n_folds=5, seed=0):
    order = np.random.default_rng(seed).permutation(n_samples)
    return [(np.setdiff1d(order, test), test) for test in np.array_split(order, n_folds)]


def cv_accuracy(make_model, X, y, folds, take=lambda data, idx: data[idx]):
    scores = []
    for train_idx, test_idx in folds:
        model = make_model().fit(take(X, train_idx), y[train_idx])
        scores.append(np.mean(model.predict(take(X, test_idx)) == y[test_idx]))
    return float(np.mean(scores))


def make_customer_data(n_samples=400, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 70, n_samples)
    income = rng.normal(50_000, 15_000, n_samples)
    city = rng.choice(["madrid", "lisbon", "paris"], n_samples)
    city_effect = np.select([city == "madrid", city == "lisbon"], [1.0, -1.0], 0.0)
    logit = 0.06 * (age - 44) + 0.00006 * (income - 50_000) + city_effect
    y = (rng.random(n_samples) < 1 / (1 + np.exp(-logit))).astype(int)
    income[rng.random(n_samples) < 0.15] = np.nan
    return pd.DataFrame({"age": age, "income": income, "city": city}), y


def main():
    print("=== Part 1: leakage from preprocessing outside the CV loop ===")
    rng = np.random.default_rng(0)
    X_noise = rng.normal(size=(100, 2000))
    y_noise = rng.integers(0, 2, size=100)
    folds = make_folds(len(y_noise))
    print("data: 100 samples, 2000 pure-noise features, random labels (true accuracy is 50%)")

    leaky_selector = TopKCorrelation(k=20).fit(X_noise, y_noise)
    X_leaky = leaky_selector.transform(X_noise)
    leaky_score = cv_accuracy(lambda: Pipeline([("scale", Standardizer()), ("model", LogisticModel())]), X_leaky, y_noise, folds)
    print(f"  select on ALL data, then CV:     accuracy = {leaky_score:.3f}  <- leakage")

    honest_score = cv_accuracy(
        lambda: Pipeline([("select", TopKCorrelation(k=20)), ("scale", Standardizer()), ("model", LogisticModel())]),
        X_noise,
        y_noise,
        folds,
    )
    print(f"  select INSIDE pipeline, CV:      accuracy = {honest_score:.3f}  <- honest")

    sklearn_pipe = SklearnPipeline(
        [("select", SelectKBest(f_classif, k=20)), ("scale", StandardScaler()), ("model", SklearnLogisticRegression())]
    )
    sklearn_score = cross_val_score(sklearn_pipe, X_noise, y_noise, cv=KFold(5, shuffle=True, random_state=0)).mean()
    print(f"  sklearn Pipeline + cross_val_score: accuracy = {sklearn_score:.3f}")

    print("\n=== Part 2: mixed column types with missing values ===")
    df, y = make_customer_data()
    print(df.head(4).to_string())
    print(f"missing income values: {df['income'].isna().sum()}")

    def make_scratch_pipeline():
        preprocess = ColumnSplitter(
            [
                ("numeric", Pipeline([("impute", MeanImputer()), ("scale", Standardizer())]), ["age", "income"]),
                ("categorical", OneHot(), ["city"]),
            ]
        )
        return Pipeline([("preprocess", preprocess), ("model", LogisticModel())])

    folds = make_folds(len(y))
    scratch_score = cv_accuracy(make_scratch_pipeline, df, y, folds, take=lambda data, idx: data.iloc[idx])
    print(f"  scratch pipeline 5-fold accuracy: {scratch_score:.3f}")

    new_customers = pd.DataFrame({"age": [25.0, 65.0], "income": [np.nan, 80_000.0], "city": ["lisbon", "madrid"]})
    fitted = make_scratch_pipeline().fit(df, y)
    print(f"  predictions for two new customers: {fitted.predict(new_customers)}")

    print("\n=== Use It: scikit-learn ===")
    preprocess = ColumnTransformer(
        [
            ("numeric", SklearnPipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), ["age", "income"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        ]
    )
    sklearn_model = SklearnPipeline([("preprocess", preprocess), ("model", SklearnLogisticRegression())])
    sklearn_cv = cross_val_score(sklearn_model, df, y, cv=[(train, test) for train, test in folds]).mean()
    print(f"  sklearn pipeline 5-fold accuracy: {sklearn_cv:.3f}")
    sklearn_model.fit(df, y)
    print(f"  predictions for two new customers: {sklearn_model.predict(new_customers)}")
    print(f"  step names: {list(sklearn_model.named_steps)}")
    print(f"  learned imputation mean for income: {sklearn_model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute'].statistics_[1]:.1f}")
    print(f"  scratch imputation mean for income: {fitted.steps[0][1].branches[0][1].steps[0][1].means[1]:.1f}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
