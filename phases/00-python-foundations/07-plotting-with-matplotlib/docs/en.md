# Plotting with Matplotlib

> A model you cannot see is a model you cannot debug. Plot the data, plot the loss, plot the errors.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 00 Lesson 05 (NumPy)
**Time:** ~45 minutes

## Learning Objectives

- Compute a histogram from scratch (bin edges and counts) and verify it against `np.histogram`
- Explain how a plot maps data coordinates to pixel coordinates
- Use matplotlib's object-oriented API (`fig, axes = plt.subplots(...)`) to build line, scatter, histogram, and bar plots in one figure
- Save figures to files with the non-interactive `Agg` backend so scripts run anywhere

## The Problem

Summary statistics lie by omission. Four datasets can share the same mean, variance, and correlation and look completely different (Anscombe's quartet). A training loop can print a decreasing loss while the validation loss quietly climbs. A classifier's accuracy hides which classes it confuses.

The only reliable defense is to look. Across this course you will plot four things over and over: loss curves (is training working, is it overfitting?), scatter plots of features colored by class (is this even separable?), histograms (what does this feature's distribution look like?), and bar charts (which model wins?). This lesson builds all four.

## The Concept

### A histogram is counting in buckets

Pick `k` equal-width bins between the minimum and the maximum:

```
width  = (max - min) / k
edges  = [min, min + width, min + 2*width, ..., max]          k + 1 edges
bin(x) = floor((x - min) / width)                             0 .. k-1
```

The maximum value lands exactly on the last edge and would compute bin `k`, so it is put into bin `k - 1`. All bins are half-open `[left, right)` except the last, which is closed `[left, right]`. NumPy uses exactly this rule.

The number of bins changes the story: too few hides structure, too many turns noise into spikes. Try several.

### A plot is a coordinate transform

Every point you plot goes through a linear map from data space to screen space:

```
pixel = pixel_min + (value - data_min) / (data_max - data_min) * (pixel_max - pixel_min)
```

```
data:   141.8 ............ 170.0 ............ 201.9   (cm)
          |                  |                  |
pixels:   80 .............. 305 .............. 560    (px)
```

Axis limits (`set_xlim`), log scales, and `set_aspect("equal")` all change this transform. `equal` aspect matters in Phase 01: without it, perpendicular vectors will not look perpendicular.

### Figure, Axes, Artist

```
+------------------------- Figure --------------------------+
|  suptitle                                                 |
|  +------- Axes [0,0] -------+  +------- Axes [0,1] ------+ |
|  | title, xlabel, ylabel    |  |                         | |
|  | Line2D artists, legend   |  |  PathCollection (dots)  | |
|  +--------------------------+  +-------------------------+ |
|  +------- Axes [1,0] -------+  +------- Axes [1,1] ------+ |
|  |  Rectangle artists       |  |  Rectangle artists      | |
|  +--------------------------+  +-------------------------+ |
+-----------------------------------------------------------+
```

- **Figure**: the whole canvas, saved with `figure.savefig`.
- **Axes**: one plot area with its own coordinate system. `plt.subplots(2, 2)` returns a 2x2 NumPy array of them.
- **Artists**: everything drawn (lines, markers, bars, text).

Always use this object-oriented style (`axes.plot`, `axes.set(...)`) rather than the implicit `plt.plot` state machine. With several subplots, explicit is much less confusing.

### Backends

matplotlib renders through a *backend*. Interactive backends open windows; `Agg` renders to image files and needs no display. Selecting it with `matplotlib.use("Agg")` before importing `pyplot` makes your script work on servers, in CI, and over SSH. Every script in this course does this and saves to a file next to the script instead of calling `plt.show()`.

## Build It

### Step 1: Histogram from scratch

```python
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
```

### Step 2: Draw it in the terminal

```python
def text_histogram(counts, edges, max_width=40):
    peak = max(counts)
    lines = []
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * round(max_width * count / peak)
        lines.append(f"  [{left:6.1f}, {right:6.1f}) {count:4d} {bar}")
    return "\n".join(lines)
```

On 500 heights drawn from a normal distribution with mean 170 and standard deviation 9:

```
  [ 156.8,  161.8)   55 ####################
  [ 161.8,  166.8)  102 #####################################
  [ 166.8,  171.8)  110 ########################################
  [ 171.8,  176.8)   96 ###################################
  [ 176.8,  181.9)   56 ####################
```

That bell shape is the normal distribution you will study in Phase 03.

### Step 3: Data to pixels

```python
def data_to_pixels(value, data_min, data_max, pixel_min, pixel_max):
    fraction = (value - data_min) / (data_max - data_min)
    return pixel_min + fraction * (pixel_max - pixel_min)
```

This is all a plotting library does to place a point, before it worries about ticks, fonts, and anti-aliasing.

## Use It

First confirm the from-scratch histogram matches NumPy:

```python
numpy_counts, numpy_edges = np.histogram(heights_cm, bins=12)
print(list(numpy_counts) == our_counts)
print(np.allclose(numpy_edges, our_edges))
```

Both `True`. Then build the four-panel figure:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

figure, axes = plt.subplots(2, 2, figsize=(10, 8))

loss_axes = axes[0, 0]
loss_axes.plot(epochs, training_loss, label="training")
loss_axes.plot(epochs, validation_loss, label="validation", linestyle="--")
best_epoch = int(epochs[np.argmin(validation_loss)])
loss_axes.axvline(best_epoch, color="gray", linewidth=1)
loss_axes.set(title="Line: loss curves", xlabel="epoch", ylabel="loss")
loss_axes.legend()

scatter_axes = axes[0, 1]
scatter_axes.scatter(class_a[:, 0], class_a[:, 1], label="class A", alpha=0.7)
scatter_axes.scatter(class_b[:, 0], class_b[:, 1], label="class B", alpha=0.7, marker="^")
scatter_axes.set_aspect("equal")

histogram_axes = axes[1, 0]
histogram_axes.hist(heights_cm, bins=numpy_edges, edgecolor="white")

bar_axes = axes[1, 1]
bars = bar_axes.bar(model_names, model_scores)
bar_axes.bar_label(bars, fmt="%.2f")

figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=100)
plt.close(figure)
```

Open `code/out.png` after running. Read each panel like a practitioner:

- **Loss curves.** Training loss keeps falling, but validation loss bottoms out at epoch 31 and then rises. That gap is overfitting (Phase 04); the vertical line marks where you would stop training.
- **Scatter.** The two classes form separate clouds with a little overlap. A straight line will separate them almost perfectly, which is what logistic regression will find in Phase 06.
- **Histogram.** Passing `bins=numpy_edges` reuses the exact edges you computed, so the bars match the terminal version.
- **Bar chart.** The baseline sits at R^2 = 0 (Lesson 04), giving every other bar a meaningful reference.

`plt.close(figure)` frees the memory. In a loop that makes hundreds of figures, forgetting it leaks memory until the process dies.

## Exercises

1. **Easy.** Change `n_bins` to 4 and to 60. Describe what each histogram suggests about the data that the other does not.
2. **Medium.** Add a fifth subplot (make the grid 2x3) showing a *residual plot*: generate `y = 3x + noise`, fit a line with `np.polyfit(x, y, 1)`, and scatter `x` against `y - prediction`. A good fit shows residuals as a shapeless band around zero.
3. **Hard.** Implement `render_scatter_ascii(points, width=60, height=20)` that uses `data_to_pixels` on both axes to place `*` characters in a grid of strings, with y flipped so larger values are at the top. Render `class_a` and `class_b` with different characters.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Histogram | "Distribution plot" | Counts of values falling in each of a set of adjacent intervals (bins) |
| Bin edges | "The buckets" | The k + 1 boundaries defining k bins; each bin is half-open except the last |
| Figure | "The plot" | The top-level matplotlib container holding one or more Axes, saved as one image |
| Axes | "A subplot" | A single plotting region with its own x and y coordinate system, ticks, and labels |
| Backend | "The renderer" | The component that turns artists into pixels or vector output; `Agg` renders to files with no display |
| Aspect ratio | "Equal axes" | Ratio of screen units per data unit on y versus x; `equal` makes one unit the same length on both axes |
| Loss curve | "Training plot" | Loss as a function of training epochs, usually for both training and validation data |

## Further Reading

- [Matplotlib Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html) -- the official introduction to Figure, Axes, and the object-oriented API
- [numpy.histogram documentation](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) -- the exact bin-edge rules reproduced in this lesson
- [Python Data Science Handbook, chapter 4](https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html) -- free chapter covering every plot type used in this course
