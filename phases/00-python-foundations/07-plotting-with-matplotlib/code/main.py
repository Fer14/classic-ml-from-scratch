import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

random.seed(5)
OUTPUT_PATH = Path(__file__).parent / "out.png"

heights_cm = [random.gauss(170, 9) for _ in range(500)]


def histogram(values, n_bins):
    low = min(values)
    high = max(values)
    width = (high - low) / n_bins
    edges = [low + i * width for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for value in values:
        index = int((value - low) / width)
        if index == n_bins:
            index -= 1
        counts[index] += 1
    return counts, edges


def text_histogram(counts, edges, max_width=40):
    peak = max(counts)
    lines = []
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * round(max_width * count / peak)
        lines.append(f"  [{left:6.1f}, {right:6.1f}) {count:4d} {bar}")
    return "\n".join(lines)


def data_to_pixels(value, data_min, data_max, pixel_min, pixel_max):
    fraction = (value - data_min) / (data_max - data_min)
    return pixel_min + fraction * (pixel_max - pixel_min)


print("Build It: a histogram from scratch")
our_counts, our_edges = histogram(heights_cm, n_bins=12)
print(text_histogram(our_counts, our_edges))
print(f"  total counted: {sum(our_counts)} of {len(heights_cm)}")

print()
print("Build It: mapping data coordinates to pixels")
for height in (min(heights_cm), 170.0, max(heights_cm)):
    pixel_x = data_to_pixels(height, min(heights_cm), max(heights_cm), 80, 560)
    print(f"  height {height:6.1f} cm -> x = {pixel_x:6.1f} px")

print()
print("Use It: numpy.histogram and matplotlib")
numpy_counts, numpy_edges = np.histogram(heights_cm, bins=12)
print(f"  counts match np.histogram: {list(numpy_counts) == our_counts}")
print(f"  edges match np.histogram:  {np.allclose(numpy_edges, our_edges)}")

generator = np.random.default_rng(5)
epochs = np.arange(1, 51)
training_loss = 2.0 * np.exp(-epochs / 10) + 0.15 + generator.normal(0, 0.02, size=epochs.size)
validation_loss = 2.0 * np.exp(-epochs / 10) + 0.25 + 0.004 * np.maximum(epochs - 25, 0) ** 1.3
class_a = generator.normal([-1.5, -1.0], 0.8, size=(80, 2))
class_b = generator.normal([1.5, 1.0], 0.8, size=(80, 2))
model_names = ["mean baseline", "linear", "tree", "forest"]
model_scores = [0.00, 0.71, 0.78, 0.86]

figure, axes = plt.subplots(2, 2, figsize=(10, 8))

loss_axes = axes[0, 0]
loss_axes.plot(epochs, training_loss, label="training")
loss_axes.plot(epochs, validation_loss, label="validation", linestyle="--")
best_epoch = int(epochs[np.argmin(validation_loss)])
loss_axes.axvline(best_epoch, color="gray", linewidth=1)
loss_axes.annotate(f"best epoch {best_epoch}", xy=(best_epoch, validation_loss.min()), xytext=(best_epoch + 3, 1.2),
                   arrowprops={"arrowstyle": "->"})
loss_axes.set(title="Line: loss curves", xlabel="epoch", ylabel="loss")
loss_axes.legend()

scatter_axes = axes[0, 1]
scatter_axes.scatter(class_a[:, 0], class_a[:, 1], label="class A", alpha=0.7)
scatter_axes.scatter(class_b[:, 0], class_b[:, 1], label="class B", alpha=0.7, marker="^")
scatter_axes.set(title="Scatter: two classes", xlabel="feature 1", ylabel="feature 2")
scatter_axes.set_aspect("equal")
scatter_axes.legend()

histogram_axes = axes[1, 0]
histogram_axes.hist(heights_cm, bins=numpy_edges, edgecolor="white")
histogram_axes.axvline(np.mean(heights_cm), color="black", linestyle=":", label=f"mean {np.mean(heights_cm):.1f}")
histogram_axes.set(title="Histogram: heights", xlabel="height (cm)", ylabel="count")
histogram_axes.legend()

bar_axes = axes[1, 1]
bars = bar_axes.bar(model_names, model_scores)
bar_axes.bar_label(bars, fmt="%.2f")
bar_axes.set(title="Bar: model comparison", ylabel="R^2", ylim=(0, 1))

figure.suptitle("Four plots you will make in every ML project")
figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)

print(f"  best validation epoch: {best_epoch}")
print(f"  saved figure to {OUTPUT_PATH.name} ({OUTPUT_PATH.stat().st_size // 1024} KB)")
