import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

OUTPUT_PATH = Path(__file__).parent / "out.png"


def lp_norm(vector, p):
    absolute = [abs(component) for component in vector]
    if p == math.inf:
        return max(absolute)
    return sum(value ** p for value in absolute) ** (1 / p)


def minkowski_distance(first, second, p):
    return lp_norm(np.asarray(first) - np.asarray(second), p)


def cosine_distance(first, second):
    return 1 - (first @ second) / (lp_norm(first, 2) * lp_norm(second, 2))


def mahalanobis_distance(first, second, inverse_covariance):
    difference = np.asarray(first) - np.asarray(second)
    return math.sqrt(difference @ inverse_covariance @ difference)


def pairwise_euclidean(points_a, points_b):
    squared_a = (points_a ** 2).sum(axis=1)[:, np.newaxis]
    squared_b = (points_b ** 2).sum(axis=1)[np.newaxis, :]
    squared = squared_a + squared_b - 2 * points_a @ points_b.T
    return np.sqrt(np.maximum(squared, 0.0))


def nearest_index(query, points, distance):
    distances = [distance(query, point) for point in points]
    return int(np.argmin(distances))


print("Step 1: one vector, many lengths")
vector = np.array([3.0, -4.0, 1.0])
for name, p in [("L1 (Manhattan)", 1), ("L2 (Euclidean)", 2), ("L3", 3), ("Linf (max)", math.inf)]:
    print(f"  {name:<15} |{vector.tolist()}| = {lp_norm(vector, p):.4f}")
print("  as p grows, the norm approaches the largest absolute component")

print()
print("Step 2: norm axioms")
generator = np.random.default_rng(0)
u, w = generator.normal(size=4), generator.normal(size=4)
for p in (1, 2, math.inf):
    triangle = lp_norm(u + w, p) <= lp_norm(u, p) + lp_norm(w, p) + 1e-12
    homogeneous = math.isclose(lp_norm(-3 * u, p), 3 * lp_norm(u, p))
    print(f"  p={p}: triangle inequality {triangle}, |c v| = |c| |v| {homogeneous}")
half = np.array([1.0, 0.0]), np.array([0.0, 1.0])
broken = lp_norm(half[0] + half[1], 0.5) <= lp_norm(half[0], 0.5) + lp_norm(half[1], 0.5)
print(f"  p=0.5: triangle inequality {broken} -> 'L0.5' is not a norm (|e1 + e2| = {lp_norm(half[0] + half[1], 0.5):.0f} > 2)")

print()
print("Step 3: the metric decides who your nearest neighbor is")
query = np.array([0.0, 0.0])
candidates = np.array([[3.9, 0.0], [3.0, 2.0], [2.9, 2.9]])
for name, distance in [("L1", lambda a, b: minkowski_distance(a, b, 1)),
                       ("L2", lambda a, b: minkowski_distance(a, b, 2)),
                       ("Linf", lambda a, b: minkowski_distance(a, b, math.inf))]:
    values = [distance(query, candidate) for candidate in candidates]
    print(f"  {name:<5} distances {np.round(values, 3)} -> nearest is {candidates[nearest_index(query, candidates, distance)]}")

print()
print("Step 4: cosine distance ignores length")
short_document, long_document, other_topic = np.array([1.0, 2.0, 0.0]), np.array([10.0, 20.0, 0.0]), np.array([1.0, 0.0, 2.0])
print(f"  euclidean(short, long) = {minkowski_distance(short_document, long_document, 2):.3f}, "
      f"euclidean(short, other) = {minkowski_distance(short_document, other_topic, 2):.3f}")
print(f"  cosine(short, long)    = {cosine_distance(short_document, long_document):.3f}, "
      f"cosine(short, other)    = {cosine_distance(short_document, other_topic):.3f}")

print()
print("Step 5: feature scale dominates Euclidean distance")
houses = np.array([[120.0, 3.0], [121.0, 1.0], [150.0, 3.0]])
print(f"  raw [m2, rooms]: d(house0, house1) = {minkowski_distance(houses[0], houses[1], 2):.2f}, "
      f"d(house0, house2) = {minkowski_distance(houses[0], houses[2], 2):.2f}")
standardized = (houses - houses.mean(axis=0)) / houses.std(axis=0)
print(f"  standardized:    d(house0, house1) = {minkowski_distance(standardized[0], standardized[1], 2):.2f}, "
      f"d(house0, house2) = {minkowski_distance(standardized[0], standardized[2], 2):.2f}")

print()
print("Step 6: Mahalanobis distance follows the shape of the data")
angle = math.radians(30)
rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
cloud = (generator.normal(size=(2000, 2)) * [3.0, 0.5]) @ rotation.T
inverse_covariance = np.linalg.inv(np.cov(cloud, rowvar=False))
along_axis = 3.0 * rotation[:, 0]
across_axis = 3.0 * rotation[:, 1]
center = np.zeros(2)
print(f"  two points at Euclidean distance 3: along the cloud {along_axis.round(3)}, across it {across_axis.round(3)}")
print(f"  Mahalanobis: along = {mahalanobis_distance(along_axis, center, inverse_covariance):.2f}, "
      f"across = {mahalanobis_distance(across_axis, center, inverse_covariance):.2f}")

print()
print("Step 7: pairwise distances without the (n, n, d) array")
points = generator.normal(size=(500, 20))
fast = pairwise_euclidean(points, points)
broadcast = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis=-1))
print(f"  |a - b|^2 = |a|^2 + |b|^2 - 2 a.b matches broadcasting: {np.allclose(fast, broadcast, atol=1e-6)}")
print(f"  largest diagonal entry (should be 0): {np.diag(fast).max():.1e}, rounding error from cancellation")
print(f"  memory: dot-product version {fast.nbytes / 1e6:.0f} MB vs broadcast intermediate {points.shape[0] ** 2 * points.shape[1] * 8 / 1e6:.0f} MB")

print()
print("Step 8: in high dimensions, everything is about equally far")
dimension_list = [2, 10, 100, 1000]
contrast_by_dimension = []
for dimension in dimension_list:
    sample = generator.uniform(size=(300, dimension))
    distances_from_first = pairwise_euclidean(sample[:1], sample[1:])[0]
    contrast = (distances_from_first.max() - distances_from_first.min()) / distances_from_first.min()
    contrast_by_dimension.append(contrast)
    print(f"  d={dimension:<5} (max - min) / min distance = {contrast:.3f}")

print()
print("Use It: NumPy, SciPy, scikit-learn")
print(f"  np.linalg.norm ord=1,2,inf: {[round(float(np.linalg.norm(vector, order)), 4) for order in (1, 2, np.inf)]}")
for metric, p in [("cityblock", 1), ("euclidean", 2), ("chebyshev", math.inf)]:
    ours = np.array([[minkowski_distance(a, b, p) for b in candidates] for a in candidates])
    print(f"  scipy cdist '{metric}' matches: {np.allclose(cdist(candidates, candidates, metric=metric), ours)}")
print(f"  scipy cdist 'mahalanobis' matches: "
      f"{math.isclose(cdist([along_axis], [center], metric='mahalanobis', VI=inverse_covariance)[0, 0], mahalanobis_distance(along_axis, center, inverse_covariance))}")
print(f"  sklearn pairwise_distances matches: {np.allclose(pairwise_distances(points), fast, atol=1e-6)}")
print(f"  sklearn cosine distance matches: "
      f"{math.isclose(pairwise_distances([short_document], [other_topic], metric='cosine')[0, 0], cosine_distance(short_document, other_topic))}")
for metric in ("manhattan", "euclidean", "chebyshev"):
    neighbors = NearestNeighbors(n_neighbors=1, metric=metric).fit(candidates)
    index = int(neighbors.kneighbors([query])[1][0, 0])
    print(f"  NearestNeighbors(metric='{metric}') nearest: {candidates[index]}")

figure, (ball_axes, curse_axes) = plt.subplots(1, 2, figsize=(11, 5))
angles = np.linspace(0, 2 * math.pi, 400)
for p, style in [(0.5, ":"), (1, "-"), (2, "-"), (4, "--"), (math.inf, "-")]:
    boundary = []
    for theta in angles:
        direction = np.array([math.cos(theta), math.sin(theta)])
        boundary.append(direction / lp_norm(direction, p))
    boundary = np.array(boundary)
    label = "p = inf" if p == math.inf else f"p = {p}" + (" (not a norm)" if p < 1 else "")
    ball_axes.plot(boundary[:, 0], boundary[:, 1], linestyle=style, label=label)
ball_axes.set(title="Unit circles: all points with |v|_p = 1", xlim=(-1.3, 2.6), ylim=(-1.3, 1.3))
ball_axes.set_aspect("equal")
ball_axes.legend(fontsize=8, loc="upper right")
curse_axes.plot(dimension_list, contrast_by_dimension, marker="o")
curse_axes.set(xscale="log", yscale="log", xlabel="dimension", ylabel="(max - min) / min distance",
               title="Distance contrast collapses as dimension grows")
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
print(f"  saved {OUTPUT_PATH.name}")
