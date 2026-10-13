# Hierarchical Clustering

> Instead of committing to k up front, build the whole family tree of merges and decide where to cut it afterwards.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 08 Lesson 01 (K-Means), Phase 01 (Linear Algebra: Norms & Distances)
**Time:** ~75 minutes

## Learning Objectives

- Implement agglomerative clustering from scratch with a distance matrix and the Lance-Williams update
- Compare single, complete, average and Ward linkage and predict which one suits which cluster shape
- Read a dendrogram and choose the number of clusters from the jumps in merge height
- Verify your merge heights against `scipy.cluster.hierarchy.linkage` and labels against scikit-learn

## The Problem

K-means makes you pick `k` before you see anything. You run it for `k = 2..10`, get ten unrelated partitions, and have to compare them. And because it only knows "distance to a center", it cannot follow a curved cluster.

Hierarchical clustering produces a single object that contains every partition from `n` clusters down to 1: a tree of merges called a dendrogram. You inspect the tree, see where the big jumps in distance are, and cut it at the level that makes sense. Biologists use it for gene expression and phylogenies; analysts use it for customer and document taxonomies because the nested structure ("premium customers split into two kinds") is itself useful.

## The Concept

### Agglomerative (Bottom-Up) Clustering

```
1. Start with every point as its own cluster (n clusters).
2. Find the two closest clusters and merge them. Record the distance (the merge height).
3. Update the distances between the new cluster and every other cluster.
4. Repeat until one cluster remains (n - 1 merges).
```

The result is a list of merges: which two clusters, at what height, giving what size. Drawn as a tree, it is a dendrogram:

```
height
  28 |            ___________|___________
     |           |                       |
  25 |      _____|_____                  |
     |     |           |                 |
   5 |   __|__       __|__             __|__
     |  |     |     |     |           |     |
     |  a  b  c     d  e  f           g  h  i
               cut here (height 10) -> 3 clusters
```

A horizontal cut at height `h` gives the clusters that exist after all merges below `h`. A large vertical gap means the clusters below it were far apart before they were forced together: a natural place to cut.

### Linkage: What "Distance Between Clusters" Means

The algorithm needs a distance between two groups of points. That choice, the linkage, completely changes the result:

| Linkage | Distance between clusters A and B | Behavior |
|---------|-----------------------------------|----------|
| Single | min distance between any a in A, b in B | Follows chains; finds curved shapes; sensitive to noise bridges |
| Complete | max distance between any a, b | Compact, similar-diameter clusters; sensitive to outliers |
| Average | mean of all pairwise distances | Compromise between single and complete |
| Ward | increase in total within-cluster variance caused by merging | Like k-means: compact, spherical, similar-size clusters |

### The Lance-Williams Update

Recomputing every cluster-to-cluster distance from raw points after each merge would be slow. Lance and Williams showed that all four linkages can be updated from the old distances alone. When clusters `i` and `j` merge, the distance to any other cluster `k` is:

```
single:    d(ij, k) = min(d(i,k), d(j,k))
complete:  d(ij, k) = max(d(i,k), d(j,k))
average:   d(ij, k) = (n_i d(i,k) + n_j d(j,k)) / (n_i + n_j)
ward:      d(ij, k)^2 = [(n_i+n_k) d(i,k)^2 + (n_j+n_k) d(j,k)^2 - n_k d(i,j)^2] / (n_i+n_j+n_k)
```

With this update the naive algorithm costs `O(n^2)` memory for the distance matrix and `O(n^3)` time (n merges, each scanning an n x n matrix). Optimized implementations reach `O(n^2)` time, but the memory cost means hierarchical clustering is practical up to tens of thousands of points, not millions.

## Build It

### Step 1: Distance matrix and the Lance-Williams update

```python
def pairwise_distances(X):
    return np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))


def lance_williams(method, d_ik, d_jk, d_ij, size_i, size_j, size_k):
    if method == "single":
        return np.minimum(d_ik, d_jk)
    if method == "complete":
        return np.maximum(d_ik, d_jk)
    if method == "average":
        return (size_i * d_ik + size_j * d_jk) / (size_i + size_j)
    if method == "ward":
        total = size_i + size_j + size_k
        squared = ((size_i + size_k) * d_ik**2 + (size_j + size_k) * d_jk**2 - size_k * d_ij**2) / total
        return np.sqrt(squared)
    raise ValueError(method)
```

The function is vectorized over `k`: `d_ik`, `d_jk` and `size_k` are arrays covering every other active cluster.

### Step 2: The merge loop

The output uses the same format as SciPy: each row is `[left_id, right_id, height, size]`, and the cluster created at step `s` gets id `n + s`.

```python
def agglomerative(X, method):
    n = len(X)
    distances = pairwise_distances(X)
    np.fill_diagonal(distances, np.inf)
    active = np.ones(n, dtype=bool)
    sizes = np.ones(n)
    cluster_ids = np.arange(n)
    merges = []
    for step in range(n - 1):
        masked = np.where(active[:, None] & active[None, :], distances, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        i, j = min(i, j), max(i, j)
        height = distances[i, j]
        others = active.copy()
        others[[i, j]] = False
        updated = lance_williams(method, distances[i, others], distances[j, others], height,
                                 sizes[i], sizes[j], sizes[others])
        distances[i, others] = updated
        distances[others, i] = updated
        merges.append([cluster_ids[i], cluster_ids[j], height, sizes[i] + sizes[j]])
        sizes[i] += sizes[j]
        cluster_ids[i] = n + step
        active[j] = False
    return np.array(merges)
```

Row/column `i` is reused for the merged cluster and `j` is deactivated, so the matrix never needs resizing.

### Step 3: Cut the tree

To get `k` clusters, replay the first `n - k` merges and see which root each point ends up under.

```python
def cut_tree(merges, n, n_clusters):
    parent = list(range(2 * n - 1))

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node

    for step in range(n - n_clusters):
        left, right = int(merges[step, 0]), int(merges[step, 1])
        parent[left] = n + step
        parent[right] = n + step
    roots = [find(point) for point in range(n)]
    _, labels = np.unique(roots, return_inverse=True)
    return labels
```

### Step 4: Read the dendrogram

On three blobs of 30 points with Ward linkage, the last merges are:

```
    merge  171 +  168 -> size  30 at height   4.634
    merge  166 +  173 -> size  30 at height   4.890
    merge  170 +  172 -> size  30 at height   5.350
    merge  174 +  175 -> size  60 at height  24.909
    merge  177 +  176 -> size  90 at height  27.956
  Height jumps between the last merges: [0.83, 0.256, 0.461, 19.559, 3.046]
```

The three blobs each complete at height ~5. The next merge jumps to ~25. Any cut between 5.4 and 24.9 gives exactly 3 clusters: that is the answer the tree is giving you.

### Step 5: Linkage changes everything

```
  method    |  blobs |  moons
  single    |  1.000 |  1.000
  complete  |  0.966 |  0.355
  average   |  1.000 |  0.534
  ward      |  1.000 |  0.510
```

On round blobs every linkage works (complete linkage misassigns a couple of edge points). On two interleaving half-moons only single linkage succeeds: it merges each point with its nearest neighbor, so it walks along each crescent. Ward and complete prefer compact clusters and cut the moons in the middle, just like k-means would. The price of single linkage: one stray noise point between two clusters can act as a bridge and chain them together.

## Use It

```python
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering

theirs = linkage(X_blobs, method="ward")
scipy_labels = fcluster(theirs, t=3, criterion="maxclust")
sklearn_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_blobs)
```

```
  single    | heights match scipy: True | ARI vs scipy 1.000 | ARI vs sklearn 1.000
  complete  | heights match scipy: True | ARI vs scipy 1.000 | ARI vs sklearn 1.000
  average   | heights match scipy: True | ARI vs scipy 1.000 | ARI vs sklearn 1.000
  ward      | heights match scipy: True | ARI vs scipy 1.000 | ARI vs sklearn 1.000
```

Your implementation reproduces SciPy's merge heights exactly. In practice:

- Use SciPy's `linkage` when you want the full tree: `dendrogram(Z)` draws it with matplotlib, and `fcluster(Z, t=h, criterion="distance")` cuts at a height.
- Use scikit-learn's `AgglomerativeClustering` inside pipelines. `distance_threshold=h` with `n_clusters=None` cuts by height, and `connectivity=` restricts merges to a nearest-neighbor graph, which makes it scale and follow manifolds.
- Ward linkage requires Euclidean distance. The other linkages accept any metric (`metric="cosine"`, `"manhattan"`, or a precomputed matrix), which is a real advantage over k-means for text or categorical data.

## Exercises

1. **Easy.** Add a single point at the midpoint between the two moons and rerun single linkage with `k = 2`. What happens to the ARI and why?
2. **Medium.** Implement `cut_at_height(merges, n, h)` that returns labels for all merges with height below `h`. Use it to find the range of heights that gives 3 clusters on the blobs.
3. **Hard.** Let `agglomerative` accept a precomputed distance matrix and use cosine distance (`1 - cosine similarity`) on 300 images of digits 0, 1 and 7 from `sklearn.datasets.load_digits`, with average linkage. Compare the ARI with Euclidean average linkage and with k-means on the same data. Why does Ward not accept cosine distance?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Agglomerative clustering | "Bottom-up clustering" | Start with singletons and repeatedly merge the two closest clusters until one remains |
| Dendrogram | "The cluster tree" | A tree recording every merge and its height; cutting it horizontally yields a flat clustering |
| Linkage | "How clusters are compared" | The rule defining distance between two sets of points: single (min), complete (max), average, or Ward (variance increase) |
| Ward linkage | "The k-means-like linkage" | Merge the pair whose union increases the total within-cluster sum of squares the least |
| Chaining | "Single linkage snakes" | Single linkage joining distinct groups through a sequence of close intermediate points |
| Lance-Williams formula | "The update rule" | A recurrence that computes the distance from a newly merged cluster to every other cluster using only previous distances |

## Further Reading

- [SciPy hierarchical clustering documentation](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) -- `linkage`, `dendrogram`, `fcluster` and the merge matrix format
- [scikit-learn User Guide: Hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) -- linkage comparison and connectivity constraints
- [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo) -- visual explanation of merges and dendrograms
- [An Introduction to Statistical Learning (ISLR), Section 12.4](https://www.statlearning.com/) -- interpreting dendrograms and choosing linkage and dissimilarity
