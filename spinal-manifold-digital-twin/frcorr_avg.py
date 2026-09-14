"""
Comparison of pairwise neuron-to-neuron spike correlations against a
time-shuffled control, supporting Figure 2D.

For each animal, a cumulative distribution of |pairwise correlation
coefficient| is compared between the real spike data ("original") and
a control built from randomly shuffled spike timings ("dummy"), which
preserves firing rates but destroys temporal structure. Both the
distributions themselves and the 90th-percentile correlation per animal
are compared, along with the fraction of pairs exceeding a range of
correlation thresholds (including the 0.3 threshold reported in the
manuscript: ~31.7% of pairs across animals).
"""

import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# One .mat file per animal, each containing:
#   sorted_correlation / cumulative_fraction             - real spike data
#   sorted_correlation_dummy / cumulative_fraction_dummy  - shuffled control
FILE_PATHS = glob.glob("save_frac/*.mat")

sorted_correlation = []
cumulative_fraction = []
sorted_correlation_dummy = []
cumulative_fraction_dummy = []

for file_path in FILE_PATHS:
    data = io.loadmat(file_path)
    sorted_correlation.append(data["sorted_correlation"][0])
    cumulative_fraction.append(data["cumulative_fraction"][0])
    sorted_correlation_dummy.append(data["sorted_correlation_dummy"][0])
    cumulative_fraction_dummy.append(data["cumulative_fraction_dummy"][0])

n_animals = len(FILE_PATHS)


# ---------------------------------------------------------------------------
# Interpolate every animal's curve onto a common x-axis, then average
# ---------------------------------------------------------------------------

def interpolate_to_common_length(curves: list, length: int) -> list:
    """Resample each 1D curve in `curves` onto `length` evenly spaced
    points, so curves of different original lengths can be averaged.
    """
    return [
        np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(curve)), curve)
        for curve in curves
    ]


max_len = max(
    len(max(sorted_correlation, key=len)),
    len(max(cumulative_fraction, key=len)),
    len(max(sorted_correlation_dummy, key=len)),
    len(max(cumulative_fraction_dummy, key=len)),
)

sorted_correlation_interp = interpolate_to_common_length(sorted_correlation, max_len)
cumulative_fraction_interp = interpolate_to_common_length(cumulative_fraction, max_len)
sorted_correlation_dummy_interp = interpolate_to_common_length(sorted_correlation_dummy, max_len)
cumulative_fraction_dummy_interp = interpolate_to_common_length(cumulative_fraction_dummy, max_len)

mean_sorted_correlation = np.mean(sorted_correlation_interp, axis=0)
mean_cumulative_fraction = np.mean(cumulative_fraction_interp, axis=0)
mean_sorted_correlation_dummy = np.mean(sorted_correlation_dummy_interp, axis=0)
mean_cumulative_fraction_dummy = np.mean(cumulative_fraction_dummy_interp, axis=0)


# ---------------------------------------------------------------------------
# Figure 2D (middle) - cumulative distribution, original vs. shuffled
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(n_animals):
    ax.plot(sorted_correlation[i], cumulative_fraction[i], color="red", alpha=0.3, linewidth=2)
    ax.plot(sorted_correlation_dummy[i], cumulative_fraction_dummy[i], color="gray", alpha=0.3, linewidth=2)

ax.plot(mean_sorted_correlation, mean_cumulative_fraction, color="red", linewidth=4, label="Original Data")
ax.plot(mean_sorted_correlation_dummy, mean_cumulative_fraction_dummy, color="black", linewidth=4, label="Dummy Data")

ax.set_title("Average Cumulative Distribution of Pair-wise Correlation Coefficients")
ax.set_xlabel("Absolute Correlation Coefficient")
ax.set_ylabel("Cumulative Fraction")
ax.legend()
fig.tight_layout()


# ---------------------------------------------------------------------------
# Figure 2D (right) - 90th-percentile correlation, original vs. shuffled
# ---------------------------------------------------------------------------

CUMULATIVE_PERCENTILE = 90

original_percentiles = [
    np.percentile(curve, CUMULATIVE_PERCENTILE) for curve in sorted_correlation_interp
]
dummy_percentiles = [
    np.percentile(curve, CUMULATIVE_PERCENTILE) for curve in sorted_correlation_dummy_interp
]

for i, value in enumerate(original_percentiles):
    print(f"Original Data {i + 1}, {CUMULATIVE_PERCENTILE}th percentile: {value:.4f}")
for i, value in enumerate(dummy_percentiles):
    print(f"Dummy Data {i + 1}, {CUMULATIVE_PERCENTILE}th percentile: {value:.4f}")

fig, ax = plt.subplots(figsize=(5, 5))

boxprops = dict(linewidth=2, color="black", alpha=1)
medianprops = dict(color="black", linewidth=2)
bplot = ax.boxplot(
    [dummy_percentiles, original_percentiles], patch_artist=True, vert=True,
    labels=["Dummy", "Original"], widths=0.6, boxprops=boxprops, medianprops=medianprops,
)
for patch, color in zip(bplot["boxes"], ["gray", "indianred"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.scatter([1] * n_animals, dummy_percentiles, color="white", edgecolors="black", linewidths=2, zorder=3)
ax.scatter([2] * n_animals, original_percentiles, color="white", edgecolors="black", linewidths=2, zorder=3)
for i in range(n_animals):
    ax.plot([1, 2], [dummy_percentiles[i], original_percentiles[i]], color="gray", alpha=0.5, zorder=1)

fig.tight_layout()


# ---------------------------------------------------------------------------
# Fraction of neuron pairs exceeding a range of correlation thresholds
# (includes the 0.3 threshold reported in the manuscript)
# ---------------------------------------------------------------------------

def fraction_above_threshold(sorted_curves, fraction_curves, threshold: float) -> list:
    """For each animal's (sorted correlation, cumulative fraction) curve,
    return the fraction of pairs with correlation above `threshold`.
    """
    fractions = []
    for x, y in zip(sorted_curves, fraction_curves):
        cumulative_at_threshold = np.interp(threshold, x, y)
        fractions.append(1 - cumulative_at_threshold)
    return fractions


thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

print(f"\n{'Threshold':>10} | {'Original mean %':>16} | {'Dummy mean %':>13} | {'Gap (pp)':>9}")
print("-" * 58)
for threshold in thresholds:
    original_fractions = fraction_above_threshold(
        sorted_correlation_interp, cumulative_fraction_interp, threshold
    )
    dummy_fractions = fraction_above_threshold(
        sorted_correlation_dummy_interp, cumulative_fraction_dummy_interp, threshold
    )
    original_mean = np.mean(original_fractions) * 100
    dummy_mean = np.mean(dummy_fractions) * 100
    print(f"{threshold:>10.2f} | {original_mean:>16.1f} | {dummy_mean:>13.1f} | {original_mean - dummy_mean:>9.1f}")

plt.show()