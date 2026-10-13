import math
import random

from sklearn.dummy import DummyClassifier, DummyRegressor


def mean_squared_error(actual, predicted):
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)


def mean_absolute_error(actual, predicted):
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def accuracy(actual, predicted):
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def evaluate(actual, predicted, *metrics, decimals=3, **named_metrics):
    results = {metric.__name__: metric(actual, predicted) for metric in metrics}
    results.update({name: metric(actual, predicted) for name, metric in named_metrics.items()})
    return {name: round(value, decimals) for name, value in results.items()}


def make_power_loss(power):
    def power_loss(actual, predicted):
        return sum(abs(a - p) ** power for a, p in zip(actual, predicted)) / len(actual)
    power_loss.__name__ = f"power_loss_{power}"
    return power_loss


class NotFittedError(Exception):
    pass


class Estimator:
    def fit(self, features, targets):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError

    def check_is_fitted(self):
        fitted = [name for name in vars(self) if name.endswith("_") and not name.startswith("_")]
        if not fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted yet. Call fit first.")

    def __repr__(self):
        params = ", ".join(f"{name}={value!r}" for name, value in vars(self).items() if not name.endswith("_"))
        return f"{type(self).__name__}({params})"


class MeanRegressor(Estimator):
    def fit(self, features, targets):
        self.mean_ = sum(targets) / len(targets)
        return self

    def predict(self, features):
        self.check_is_fitted()
        return [self.mean_ for _ in features]

    def score(self, features, targets):
        predictions = self.predict(features)
        target_mean = sum(targets) / len(targets)
        residual = sum((t - p) ** 2 for t, p in zip(targets, predictions))
        total = sum((t - target_mean) ** 2 for t in targets)
        return 1 - residual / total


class QuantileRegressor(MeanRegressor):
    def __init__(self, quantile=0.5):
        self.quantile = quantile

    def fit(self, features, targets):
        ordered = sorted(targets)
        position = self.quantile * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        self.mean_ = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
        return self


class MajorityClassifier(Estimator):
    def fit(self, features, labels):
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        self.classes_ = sorted(counts)
        self.majority_ = max(self.classes_, key=lambda label: counts[label])
        return self

    def predict(self, features):
        self.check_is_fitted()
        return [self.majority_ for _ in features]

    def score(self, features, labels):
        return accuracy(labels, self.predict(features))


random.seed(3)
house_sizes = [[random.uniform(50, 200)] for _ in range(40)]
house_prices = [round(2.0 * size[0] + random.gauss(0, 30) + 100, 1) for size in house_sizes]
spam_labels = random.choices(["ham", "spam"], weights=[0.7, 0.3], k=40)

print("Step 1: functions, default and keyword arguments")
constant_guess = [sum(house_prices) / len(house_prices)] * len(house_prices)
print(f"  mse={mean_squared_error(house_prices, constant_guess):.2f}")
print(f"  evaluate(...) -> {evaluate(house_prices, constant_guess, mean_squared_error, mean_absolute_error)}")
print(f"  evaluate(..., decimals=1, max_err=...) -> "
      f"{evaluate(house_prices, constant_guess, mean_absolute_error, decimals=1, max_err=lambda a, p: max(abs(x - y) for x, y in zip(a, p)))}")

print()
print("Step 2: functions that return functions (closures)")
cubic_loss = make_power_loss(3)
print(f"  {cubic_loss.__name__}: {cubic_loss(house_prices, constant_guess):.1f}")
print(f"  make_power_loss(2) equals MSE: {math.isclose(make_power_loss(2)(house_prices, constant_guess), mean_squared_error(house_prices, constant_guess))}")

print()
print("Step 3: classes with fit / predict")
regressor = MeanRegressor()
print(f"  repr before fit: {regressor!r}")
try:
    regressor.predict(house_sizes)
except NotFittedError as error:
    print(f"  predict before fit raises: {error}")
regressor.fit(house_sizes, house_prices)
print(f"  learned mean_ = {regressor.mean_:.3f}")
print(f"  first 3 predictions: {[round(p, 2) for p in regressor.predict(house_sizes[:3])]}")
print(f"  R^2 on training data = {regressor.score(house_sizes, house_prices):.3f}")

print()
print("Step 4: inheritance")
median_model = QuantileRegressor(quantile=0.5).fit(house_sizes, house_prices)
print(f"  {median_model!r} learned {median_model.mean_:.3f}")
print(f"  isinstance(median_model, MeanRegressor) = {isinstance(median_model, MeanRegressor)}")
classifier = MajorityClassifier().fit(house_sizes, spam_labels)
print(f"  {classifier!r} classes_={classifier.classes_} majority_={classifier.majority_!r}")
print(f"  accuracy = {classifier.score(house_sizes, spam_labels):.3f}")

print()
print("Use It: scikit-learn dummy estimators share the same API")
sklearn_mean = DummyRegressor(strategy="mean").fit(house_sizes, house_prices)
sklearn_median = DummyRegressor(strategy="median").fit(house_sizes, house_prices)
sklearn_majority = DummyClassifier(strategy="most_frequent").fit(house_sizes, spam_labels)
print(f"  mean:     ours={regressor.mean_:.3f}  sklearn={sklearn_mean.constant_[0][0]:.3f}")
print(f"  median:   ours={median_model.mean_:.3f}  sklearn={sklearn_median.constant_[0][0]:.3f}")
print(f"  majority: ours={classifier.majority_!r}  sklearn={str(sklearn_majority.predict(house_sizes[:1])[0])!r}")
print(f"  R^2:      ours={regressor.score(house_sizes, house_prices):.3f}  sklearn={sklearn_mean.score(house_sizes, house_prices):.3f}")
print(f"  accuracy: ours={classifier.score(house_sizes, spam_labels):.3f}  sklearn={sklearn_majority.score(house_sizes, spam_labels):.3f}")
