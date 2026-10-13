import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler


class StandardScalerScratch:
    def fit(self, features):
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, features):
        return (features - self.mean) / self.scale


class MinMaxScalerScratch:
    def fit(self, features):
        self.minimum = features.min(axis=0)
        self.range = features.max(axis=0) - self.minimum
        self.range[self.range == 0] = 1.0
        return self

    def transform(self, features):
        return (features - self.minimum) / self.range


class RobustScalerScratch:
    def fit(self, features):
        self.median = np.median(features, axis=0)
        self.iqr = np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0)
        self.iqr[self.iqr == 0] = 1.0
        return self

    def transform(self, features):
        return (features - self.median) / self.iqr


class OneHotEncoderScratch:
    def fit(self, column):
        self.categories = np.unique(column)
        return self

    def transform(self, column):
        return (column[:, None] == self.categories[None, :]).astype(float)


class OrdinalEncoderScratch:
    def __init__(self, order):
        self.mapping = {category: index for index, category in enumerate(order)}

    def transform(self, column):
        return np.array([self.mapping[value] for value in column], dtype=float)


def knn_predict(train_x, train_y, query_x, k=5):
    distances = np.linalg.norm(query_x[:, None, :] - train_x[None, :, :], axis=2)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    return (train_y[neighbors].mean(axis=1) > 0.5).astype(int)


def gradient_descent_mse(features, targets, learning_rate, steps=1000):
    weights = np.zeros(features.shape[1])
    intercept = 0.0
    for _ in range(steps):
        residual = features @ weights + intercept - targets
        weights -= learning_rate * 2 * features.T @ residual / len(targets)
        intercept -= learning_rate * 2 * residual.mean()
    return np.mean((features @ weights + intercept - targets) ** 2)


def make_customers(n_samples, rng):
    income = rng.normal(60000, 15000, n_samples)
    age = rng.uniform(18, 70, n_samples)
    signal = (age - 44) / 15 + rng.normal(0, 0.5, n_samples)
    labels = (signal > 0).astype(int)
    return np.column_stack([income, age]), labels


def main():
    rng = np.random.default_rng(5)

    print("=== Step 1: why scale? distances are dominated by large features ===")
    features, labels = make_customers(600, rng)
    train_x, test_x = features[:400], features[400:]
    train_y, test_y = labels[:400], labels[400:]
    raw_accuracy = np.mean(knn_predict(train_x, train_y, test_x) == test_y)
    scaler = StandardScalerScratch().fit(train_x)
    scaled_accuracy = np.mean(knn_predict(scaler.transform(train_x), train_y, scaler.transform(test_x)) == test_y)
    print(f"income std {train_x[:, 0].std():.0f}, age std {train_x[:, 1].std():.1f}")
    print(f"kNN accuracy raw features:    {raw_accuracy:.3f}")
    print(f"kNN accuracy scaled features: {scaled_accuracy:.3f}")

    print("\n=== Step 2: scaling speeds up gradient descent ===")
    targets = 0.0001 * features[:, 0] + 0.5 * features[:, 1] + rng.normal(0, 1, len(features))
    scaled_features = StandardScalerScratch().fit(features).transform(features)
    print(f"condition number raw:          {np.linalg.cond(np.column_stack([np.ones(len(features)), features])):.0f}")
    print(f"condition number standardized: {np.linalg.cond(np.column_stack([np.ones(len(features)), scaled_features])):.2f}")
    print(f"raw features, lr=1e-10 (lr=1e-9 diverges), 1000 steps: MSE {gradient_descent_mse(features, targets, 1e-10):.3f}")
    print(f"standardized features, lr=0.1, 1000 steps:            MSE {gradient_descent_mse(scaled_features, targets, 0.1):.3f}")

    print("\n=== Step 3: three scalers, one outlier ===")
    column = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]])
    for name, scaler_class in (("standard", StandardScalerScratch), ("min-max", MinMaxScalerScratch), ("robust", RobustScalerScratch)):
        transformed = scaler_class().fit(column).transform(column).ravel()
        print(f"{name:8s}: {np.round(transformed, 3)}")

    print("\n=== Step 4: fit on train, transform test ===")
    leaky = StandardScalerScratch().fit(features)
    clean = StandardScalerScratch().fit(train_x)
    print(f"mean learned from all data:   income {leaky.mean[0]:.1f}, age {leaky.mean[1]:.2f}")
    print(f"mean learned from train only: income {clean.mean[0]:.1f}, age {clean.mean[1]:.2f}")
    print(f"test set after train-fitted scaling, mean {np.round(clean.transform(test_x).mean(axis=0), 3)} (not exactly 0, and that is correct)")

    print("\n=== Step 5: encoding categorical features ===")
    cities = np.array(["madrid", "paris", "tokyo", "paris", "madrid", "lima"])
    sizes = np.array(["small", "large", "medium", "small", "large", "medium"])
    one_hot = OneHotEncoderScratch().fit(cities)
    print(f"one-hot categories: {one_hot.categories}")
    print(one_hot.transform(cities).astype(int))
    ordinal = OrdinalEncoderScratch(["small", "medium", "large"])
    print(f"ordinal sizes: {ordinal.transform(sizes)}")
    unseen = np.array(["berlin", "paris"])
    print(f"unseen category 'berlin' one-hot: {one_hot.transform(unseen).astype(int).tolist()}")

    print("\n=== Use It: scikit-learn ===")
    sk_standard = StandardScaler().fit(column)
    sk_minmax = MinMaxScaler().fit(column)
    sk_robust = RobustScaler().fit(column)
    print(f"StandardScaler: {np.round(sk_standard.transform(column).ravel(), 3)}")
    print(f"MinMaxScaler:   {np.round(sk_minmax.transform(column).ravel(), 3)}")
    print(f"RobustScaler:   {np.round(sk_robust.transform(column).ravel(), 3)}")
    encoder = OneHotEncoder(handle_unknown="ignore").fit(cities[:, None])
    print(f"OneHotEncoder categories: {encoder.categories_[0]}")
    print(f"unseen 'berlin': {encoder.transform(unseen[:, None]).toarray().astype(int).tolist()}")
    ordinal_sk = OrdinalEncoder(categories=[["small", "medium", "large"]]).fit(sizes[:, None])
    print(f"OrdinalEncoder: {ordinal_sk.transform(sizes[:, None]).ravel()}")
    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(train_x, train_y)
    print(f"Pipeline(StandardScaler, kNN) accuracy: {pipeline.score(test_x, test_y):.3f}")
    table = np.column_stack([cities, sizes, np.array([30, 45, 28, 52, 39, 61])]).astype(object)
    transformer = ColumnTransformer([
        ("city", OneHotEncoder(), [0]),
        ("size", OrdinalEncoder(categories=[["small", "medium", "large"]]), [1]),
        ("age", StandardScaler(), [2]),
    ])
    print(f"ColumnTransformer output shape: {transformer.fit_transform(table).shape}")


if __name__ == "__main__":
    main()
