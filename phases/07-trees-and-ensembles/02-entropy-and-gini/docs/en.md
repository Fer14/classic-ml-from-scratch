# Entropy and Gini

> A split is good when it makes the children more certain than the parent. Entropy and Gini are two ways to measure certainty.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 03 (Probability and Statistics), Phase 07 Lesson 01 (Decision Trees)
**Time:** ~60 minutes

## Learning Objectives

- Compute entropy, Gini impurity, and misclassification error for any node, and explain what each one measures
- Calculate information gain and gain ratio for a candidate split
- Explain why misclassification error is a poor split criterion even though it is the metric you care about
- Show why plain information gain favors high-cardinality features and how gain ratio corrects it

## The Problem

In the previous lesson you used Gini impurity without questioning it. But open any tree library and you find a `criterion` argument: `"gini"`, `"entropy"`, `"log_loss"`. Older algorithms (ID3, C4.5) use information gain and gain ratio. Why so many choices? Why not just pick the split that minimizes classification errors, since that is what you will be scored on?

The answer matters because the impurity function is the only thing that tells a tree which split is "best". Choose a bad one and the tree makes splits that look fine locally but leave the children no easier to split. Choose an unbalanced one and the tree latches onto ID columns and zip codes.

## The Concept

### Three ways to measure a mixed node

Let `p_k` be the fraction of class k in a node.

```
Entropy:            H = - sum_k p_k * log2(p_k)
Gini impurity:      G = 1 - sum_k p_k^2
Misclassification:  E = 1 - max_k p_k
```

All three are 0 for a pure node and maximal when classes are equally mixed. They differ in shape and in meaning.

- **Entropy** comes from information theory (Phase 03): the expected number of bits needed to encode the label of a random sample from the node. A 50/50 binary node needs 1 bit. A pure node needs 0.
- **Gini** is the probability that two samples drawn at random (with replacement) from the node have different labels. It is also the expected error rate if you predicted by sampling a label from the node distribution.
- **Misclassification error** is the error rate if the node predicts its majority class.

For two classes with `p` the proportion of class 1:

```
p (class 1)   entropy   entropy/2   gini    misclassification
0.00          0.000     0.000       0.000   0.000
0.10          0.469     0.234       0.180   0.100
0.25          0.811     0.406       0.375   0.250
0.50          1.000     0.500       0.500   0.500
```

The script saves the real curves to `impurity_curves.png`. Entropy (scaled by 1/2 to share the axis) and Gini are smooth and strictly concave. Misclassification is a tent made of two straight lines.

### Information gain

The impurity decrease from Lesson 01, computed with entropy, has a name: information gain.

```
IG(S, split) = H(S) - sum_j |S_j|/|S| * H(S_j)
```

It is the mutual information between the split outcome and the label: how many bits of uncertainty about the class you remove by learning which side of the threshold the sample falls on.

### Why concavity matters

Here is the classic example. The parent has 400 samples of each class. Compare:

```
Split A:  (300, 100) | (100, 300)
Split B:  (200, 400) | (200,   0)
```

Both splits misclassify 200 samples, so misclassification error says they are equally good (decrease 0.25 each). But Split B produces a pure node that needs no further work, while Split A leaves two 75/25 nodes. Entropy and Gini both prefer B:

```
                 Split A   Split B
entropy gain      0.1887    0.3113
gini decrease     0.1250    0.1667
misclass decrease 0.2500    0.2500
```

The reason is geometric. The weighted child impurity is a weighted average of points on the impurity curve. For a strictly concave curve, moving the children toward the extremes (purer) always lowers that average, even if the majority-class predictions do not change. For the piecewise-linear tent, children on the same side of 0.5 average out to exactly the parent, so many useful splits score zero gain. That makes trees grown with misclassification error stall early.

Misclassification error is still useful for one thing: pruning, where you evaluate whole subtrees by the metric you actually care about.

### Gini vs entropy in practice

Gini and entropy produce nearly identical trees in practice; studies find they disagree on only a small fraction of splits. Gini is slightly cheaper (no logarithm), which is why it is the default in CART and scikit-learn. Entropy punishes very impure nodes a bit more. In sklearn, `"entropy"` and `"log_loss"` are the same criterion. Treat the choice as a hyperparameter, not a philosophy.

### The many-values bias and gain ratio

ID3 split categorical features into one branch per value. A feature with 200 unique values (a customer ID) produces 200 children of size 1, each perfectly pure. Information gain is then the full parent entropy, the maximum possible, even though the feature is worthless for new data.

C4.5 fixes this with gain ratio, dividing by the entropy of the split itself:

```
SplitInfo(S, split) = - sum_j |S_j|/|S| * log2(|S_j|/|S|)
GainRatio = IG / SplitInfo
```

A split into 200 equal pieces has SplitInfo = log2(200), about 7.6 bits, so its gain ratio collapses. A balanced binary split has SplitInfo = 1, so its gain ratio equals its information gain.

CART sidesteps the issue by only using binary splits, but continuous features with many unique thresholds still give it more chances to find a lucky split. This is one reason impurity-based feature importances are biased toward high-cardinality features (Phase 10 covers permutation importance as the fix).

## Build It

### Step 1: The three impurity functions

```python
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
```

The `+ 0.0` turns a pure node's `-0.0` into `0.0` for clean printing. `np.unique` only returns classes that are present, so `log2(0)` never happens.

```
counts=[8, 2]        entropy=0.722  gini=0.320  misclass=0.200
counts=[2, 2, 2]     entropy=1.585  gini=0.667  misclass=0.667
```

With three equal classes the maximum entropy is log2(3) = 1.585 and the maximum Gini is 1 - 1/3 = 0.667.

### Step 2: Impurity decrease and gain ratio

```python
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
```

These accept any number of children, so they work for multiway categorical splits as well as binary ones.

### Step 3: Compare Split A and Split B

```python
parent = np.array([0] * 400 + [1] * 400)
split_a = [np.array([0] * 300 + [1] * 100), np.array([0] * 100 + [1] * 300)]
split_b = [np.array([0] * 200 + [1] * 400), np.array([0] * 200)]
```

The script prints the table from The Concept section. Misclassification gives a tie; the concave measures pick B.

### Step 4: Demonstrate the many-values bias

```python
labels = rng.integers(0, 2, n_samples)
useful_feature = np.where(rng.random(n_samples) < 0.8, labels, 1 - labels)
id_like_feature = np.arange(n_samples)
```

`useful_feature` agrees with the label 80% of the time. `id_like_feature` is a unique row ID.

```
useful binary feature: info_gain=0.2401  gain_ratio=0.2401
ID-like feature (200 values): info_gain=0.9982  gain_ratio=0.1306
```

Information gain ranks the ID column four times higher. Gain ratio reverses the ranking.

## Use It

In scikit-learn the impurity is the `criterion` argument:

```python
from sklearn.tree import DecisionTreeClassifier

for criterion in ["gini", "entropy", "log_loss"]:
    model = DecisionTreeClassifier(criterion=criterion, max_depth=4, random_state=0)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
```

On the breast cancer dataset:

```
criterion=gini      cv_acc=0.927  test_acc=0.906  root split='worst perimeter'
criterion=entropy   cv_acc=0.925  test_acc=0.930  root split='worst perimeter'
criterion=log_loss  cv_acc=0.925  test_acc=0.930  root split='worst perimeter'
```

Same root split, cross-validated accuracies within 0.002. The test accuracy gap comes from a 171-sample test set, where one or two predictions move the number by a percent. That is why you compare criteria with cross-validation, not a single split.

The fitted tree exposes its internals in `model.tree_`: `impurity`, `n_node_samples`, `feature`, `threshold`. The script recomputes the root Gini decrease from these arrays and gets the same value as your `impurity_decrease` (0.3435). For regression, the criterion options are `"squared_error"`, `"friedman_mse"`, `"absolute_error"`, and `"poisson"`.

## Exercises

1. (Easy) Prove numerically that for two classes Gini is always at most half of entropy measured in bits, by evaluating both on a grid of p values.
2. (Medium) Write a function that, for a given dataset, counts how often Gini and entropy choose a different best split at the root across 100 bootstrap resamples. Use the breast cancer dataset. How often do they disagree?
3. (Hard) Implement C4.5-style multiway splits for categorical features in your Lesson 01 tree, using gain ratio. Add a random ID column to a dataset and confirm your tree ignores it, while an information-gain version splits on it first.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Entropy | "Disorder" | -sum p_k log2 p_k, the expected bits needed to encode a label drawn from the node |
| Gini impurity | "Gini index" | 1 - sum p_k^2, the probability that two random draws from the node disagree |
| Misclassification error | "Error rate" | 1 - max p_k, the error from predicting the node's majority class |
| Information gain | "How much a split helps" | Parent entropy minus weighted child entropy; the mutual information between split and label |
| Gain ratio | "Normalized information gain" | Information gain divided by the entropy of the split proportions, used by C4.5 to penalize many-way splits |
| Split information | "Intrinsic value" | The entropy of the child sizes, large when a split creates many small children |
| Concave impurity | "Curved impurity" | An impurity whose weighted child average drops whenever children become purer, which keeps split search making progress |
| criterion | "The sklearn knob" | The DecisionTree argument selecting the impurity: gini, entropy, log_loss, or regression losses |

## Further Reading

- [The Elements of Statistical Learning, Section 9.2](https://hastie.su.domains/ElemStatLearn/) -- compares Gini, cross-entropy, and misclassification as node impurities
- [StatQuest: Entropy (for data science) Clearly Explained](https://www.youtube.com/watch?v=YtebGVx-Fxw) -- builds intuition for entropy as surprise
- J. R. Quinlan, *C4.5: Programs for Machine Learning* (1993) -- the origin of gain ratio
- [scikit-learn User Guide: Tree mathematical formulation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation) -- exact definitions of every criterion sklearn implements
