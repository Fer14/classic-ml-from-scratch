import math
import random
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

random.seed(0)
N_SAMPLES = 400
N_FEATURES = 5
data_as_lists = [[random.gauss(10 * (j + 1), j + 1) for j in range(N_FEATURES)] for _ in range(N_SAMPLES)]


def standardize_lists(rows):
    n_rows = len(rows)
    n_columns = len(rows[0])
    means = [sum(row[j] for row in rows) / n_rows for j in range(n_columns)]
    stds = [math.sqrt(sum((row[j] - means[j]) ** 2 for row in rows) / n_rows) for j in range(n_columns)]
    return [[(row[j] - means[j]) / stds[j] for j in range(n_columns)] for row in rows]


def pairwise_distances_lists(rows):
    distances = []
    for first in rows:
        distance_row = []
        for second in rows:
            squared = 0.0
            for a, b in zip(first, second):
                squared += (a - b) ** 2
            distance_row.append(math.sqrt(squared))
        distances.append(distance_row)
    return distances


def standardize_numpy(matrix):
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)


def pairwise_distances_numpy(matrix):
    differences = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
    return np.sqrt((differences ** 2).sum(axis=-1))


def timed(function, *arguments):
    start = time.perf_counter()
    result = function(*arguments)
    return result, time.perf_counter() - start


print("Step 1: arrays, shape, dtype")
data = np.array(data_as_lists)
print(f"  shape={data.shape} ndim={data.ndim} dtype={data.dtype} size={data.size}")
print(f"  first row        {np.round(data[0], 2)}")
print(f"  column 2, rows 0-3 {np.round(data[:4, 2], 2)}")
print(f"  np.arange(6).reshape(2, 3) =\n{np.arange(6).reshape(2, 3)}")

print()
print("Step 2: vectorized math and axis reductions")
print(f"  column means (axis=0)  {np.round(data.mean(axis=0), 2)}")
print(f"  column stds  (axis=0)  {np.round(data.std(axis=0), 2)}")
print(f"  first 3 row sums (axis=1) {np.round(data.sum(axis=1)[:3], 2)}")

print()
print("Step 3: boolean masks")
mask = data[:, 0] > 11
print(f"  rows with feature 0 > 11: {mask.sum()} of {len(mask)}")
print(f"  mean of feature 1 on those rows: {data[mask, 1].mean():.3f}")
clipped = np.where(data > 40, 40.0, data)
print(f"  values above 40 before/after np.where clip: {(data > 40).sum()} / {(clipped > 40).sum()}")

print()
print("Step 4: broadcasting")
row_vector = np.array([1.0, 2.0, 3.0])
column_vector = np.array([[10.0], [20.0]])
print(f"  (2,1) + (3,) -> {(column_vector + row_vector).shape}\n{column_vector + row_vector}")

standardized_lists, list_seconds = timed(standardize_lists, data_as_lists)
standardized_numpy, numpy_seconds = timed(standardize_numpy, data)
print(f"  standardize: lists {list_seconds * 1000:.2f} ms, numpy {numpy_seconds * 1000:.2f} ms, "
      f"same result: {np.allclose(standardized_lists, standardized_numpy)}")

print()
print("Step 5: the speed of vectorization")
distances_lists, list_seconds = timed(pairwise_distances_lists, data_as_lists)
distances_numpy, numpy_seconds = timed(pairwise_distances_numpy, data)
print(f"  pairwise distances {N_SAMPLES}x{N_SAMPLES}: lists {list_seconds:.3f} s, numpy {numpy_seconds:.4f} s "
      f"(~{list_seconds / numpy_seconds:.0f}x faster)")
print(f"  same result: {np.allclose(distances_lists, distances_numpy)}")

print()
print("Use It: scikit-learn and SciPy")
sklearn_standardized = StandardScaler().fit_transform(data)
scipy_distances = cdist(data, data)
print(f"  StandardScaler matches ours: {np.allclose(sklearn_standardized, standardized_numpy)}")
print(f"  scipy cdist matches ours:    {np.allclose(scipy_distances, distances_numpy)}")
print(f"  standardized means ~0: {np.allclose(standardized_numpy.mean(axis=0), 0)}, stds ~1: {np.allclose(standardized_numpy.std(axis=0), 1)}")
generator = np.random.default_rng(42)
print(f"  modern random API: default_rng(42).normal(size=3) = {np.round(generator.normal(size=3), 4)}")
