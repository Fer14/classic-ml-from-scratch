from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


def class_proportions(labels):
    _, counts = np.unique(labels, return_counts=True)
    return counts / counts.sum()


def entropy(labels):
    if len(labels) == 0:
        return 0.0
    p = class_proportions(labels)
    return float(-np.sum(p * np.log2(p))) + 0.0


def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = class_proportions(labels)
    return float(1.0 - np.sum(p**2))


def misclassification(labels):
    if len(labels) == 0:
        return 0.0
    p = class_proportions(labels)
    return float(1.0 - np.max(p))


def weighted_child_impurity(children, impurity):
    total = sum(len(child) for child in children)
    return sum(len(child) / total * impurity(child) for child in children)


def impurity_decrease(parent, children, impurity):
    return impurity(parent) - weighted_child_impurity(children, impurity)


def split_information(children):
    total = sum(len(child) for child in children)
    fractions = np.array([len(child) / total for child in children if len(child) > 0])
    return float(-np.sum(fractions * np.log2(fractions)))


def gain_ratio(parent, children):
    split_info = split_information(children)
    if split_info == 0:
        return 0.0
    return impurity_decrease(parent, children, entropy) / split_info


def binary_curves(n_points=101):
    p = np.linspace(0, 1, n_points)
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    entropy_curve = -(safe * np.log2(safe) + (1 - safe) * np.log2(1 - safe))
    gini_curve = 1 - p**2 - (1 - p) ** 2
    misclassification_curve = 1 - np.maximum(p, 1 - p)
    return p, entropy_curve, gini_curve, misclassification_curve


def describe_split(name, parent, children):
    print(f"  {name}")
    print(f"    children sizes={[len(c) for c in children]}")
    for impurity in (entropy, gini, misclassification):
        print(f"    {impurity.__name__:18s} decrease={impurity_decrease(parent, children, impurity):.4f}")
    print(f"    {'gain_ratio':18s} value={gain_ratio(parent, children):.4f}")


def main():
    print("=== Step 1: Impurity of a few nodes ===")
    for labels in [[0] * 10, [0] * 5 + [1] * 5, [0] * 8 + [1] * 2, [0, 0, 1, 1, 2, 2]]:
        arr = np.array(labels)
        counts = str(np.bincount(arr).tolist())
        print(f"  counts={counts:12s}  entropy={entropy(arr):.3f}  gini={gini(arr):.3f}  misclass={misclassification(arr):.3f}")

    print("\n=== Step 2: Binary impurity curves ===")
    p, entropy_curve, gini_curve, misclassification_curve = binary_curves()
    for value in [0.0, 0.1, 0.25, 0.5]:
        index = int(round(value * 100))
        print(f"  p={value:.2f}  entropy={entropy_curve[index]:.3f}  gini={gini_curve[index]:.3f}  misclass={misclassification_curve[index]:.3f}")
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(p, entropy_curve / 2, label="entropy / 2")
    axis.plot(p, gini_curve, label="gini")
    axis.plot(p, misclassification_curve, label="misclassification")
    axis.set_xlabel("proportion of class 1")
    axis.set_ylabel("impurity")
    axis.legend()
    figure.tight_layout()
    output_path = Path(__file__).parent / "impurity_curves.png"
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    print(f"  saved {output_path.name}")

    print("\n=== Step 3: Why misclassification error is a weak split criterion ===")
    parent = np.array([0] * 400 + [1] * 400)
    split_a = [np.array([0] * 300 + [1] * 100), np.array([0] * 100 + [1] * 300)]
    split_b = [np.array([0] * 200 + [1] * 400), np.array([0] * 200)]
    describe_split("Split A: (300,100) | (100,300)", parent, split_a)
    describe_split("Split B: (200,400) | (200,0)", parent, split_b)
    print("  Misclassification rates both splits the same; entropy and gini prefer B, which creates a pure node.")

    print("\n=== Step 4: Information gain is biased toward many-valued features ===")
    rng = np.random.default_rng(7)
    n_samples = 200
    labels = rng.integers(0, 2, n_samples)
    useful_feature = np.where(rng.random(n_samples) < 0.8, labels, 1 - labels)
    id_like_feature = np.arange(n_samples)
    useful_children = [labels[useful_feature == value] for value in np.unique(useful_feature)]
    id_children = [labels[id_like_feature == value] for value in np.unique(id_like_feature)]
    print(f"  useful binary feature: info_gain={impurity_decrease(labels, useful_children, entropy):.4f}  gain_ratio={gain_ratio(labels, useful_children):.4f}")
    print(f"  ID-like feature (200 values): info_gain={impurity_decrease(labels, id_children, entropy):.4f}  gain_ratio={gain_ratio(labels, id_children):.4f}")
    print("  Information gain picks the useless ID column; gain ratio penalizes it.")

    print("\n=== Use It: criterion in scikit-learn ===")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    for criterion in ["gini", "entropy", "log_loss"]:
        model = DecisionTreeClassifier(criterion=criterion, max_depth=4, random_state=0)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        root_feature = data.feature_names[model.tree_.feature[0]]
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  criterion={criterion:8s}  cv_acc={cv_scores.mean():.3f}  test_acc={test_acc:.3f}  root split='{root_feature}'")

    root_samples = y_train
    model = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=0).fit(X_train, y_train)
    left_mask = X_train[:, model.tree_.feature[0]] <= model.tree_.threshold[0]
    our_gain = impurity_decrease(root_samples, [y_train[left_mask], y_train[~left_mask]], gini)
    sklearn_gain = model.tree_.impurity[0] - (
        model.tree_.n_node_samples[1] * model.tree_.impurity[1] + model.tree_.n_node_samples[2] * model.tree_.impurity[2]
    ) / model.tree_.n_node_samples[0]
    print(f"  root gini decrease: ours={our_gain:.4f}  sklearn tree_={sklearn_gain:.4f}")


if __name__ == "__main__":
    main()
