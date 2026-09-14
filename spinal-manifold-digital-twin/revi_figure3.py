"""
Identification of neurons most strongly correlated with each manifold
dimension, supporting Figure 3E.

For a single trial, this script correlates every neuron's firing rate
with the two Isomap latent dimensions (Dim1, Dim2) and highlights the
five neurons most strongly aligned with each dimension. Neurons that
rank in the top five for both dimensions are marked with a bicolor
(half Dim1, half Dim2) marker.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.io import loadmat
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_PATH = "save_iso&ref_fr/0214_SC_1_40Hz.mat"
TOP_N = 5   # number of top-aligned neurons highlighted per dimension

COLOR_DIM1 = "indianred"
COLOR_DIM2 = "#4D4D4D"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

data = loadmat(FILE_PATH)
fr = np.array(data["frRef"])          # shape: (n_neurons, T)
iso_0 = np.array(data["iso_0"]).flatten()   # Dim1
iso_1 = np.array(data["iso_1"]).flatten()   # Dim2


# ---------------------------------------------------------------------------
# Per-neuron correlation with each manifold dimension
# ---------------------------------------------------------------------------

correlations = np.array(
    [[pearsonr(unit, iso_0)[0], pearsonr(unit, iso_1)[0]] for unit in fr]
)

# Drop neurons with an undefined correlation (e.g., constant firing rate)
valid = ~np.isnan(correlations).any(axis=1)
correlations = np.abs(correlations[valid])
fr = fr[valid]


# ---------------------------------------------------------------------------
# Top-aligned neurons per dimension, with overlap handled separately
# ---------------------------------------------------------------------------

def half_circle_marker(start_angle: float, end_angle: float, n: int = 50) -> mpath.Path:
    """Pie-slice marker path, used to build a half-Dim1/half-Dim2 split
    marker for neurons that rank in the top N for both dimensions.
    """
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), n)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    verts = np.vstack([[0, 0], verts, [0, 0]])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts) - 2) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)


dim1_marker = half_circle_marker(90, 270)
dim2_marker = half_circle_marker(-90, 90)

top_dim1 = set(np.argsort(correlations[:, 0])[-TOP_N:].tolist())
top_dim2 = set(np.argsort(correlations[:, 1])[-TOP_N:].tolist())

overlap_idx = np.array(sorted(top_dim1 & top_dim2))
only_dim1_idx = np.array(sorted(top_dim1 - top_dim2))
only_dim2_idx = np.array(sorted(top_dim2 - top_dim1))

print("Dim1-only:", only_dim1_idx)
print("Dim2-only:", only_dim2_idx)
print("Top in both:", overlap_idx)


# ---------------------------------------------------------------------------
# Figure 3E - neurons correlated with each manifold dimension
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(4.5, 4.5))

ax.scatter(
    correlations[:, 0], correlations[:, 1],
    color="lightgray", s=50, label="All neurons",
)
ax.scatter(
    correlations[only_dim1_idx, 0], correlations[only_dim1_idx, 1],
    color=COLOR_DIM1, s=200, label="Top DIM1-aligned",
)
ax.scatter(
    correlations[only_dim2_idx, 0], correlations[only_dim2_idx, 1],
    color=COLOR_DIM2, s=200, label="Top DIM2-aligned",
)
if len(overlap_idx) > 0:
    ax.scatter(
        correlations[overlap_idx, 0], correlations[overlap_idx, 1],
        marker=dim1_marker, color=COLOR_DIM1, s=200, label="Top in both",
    )
    ax.scatter(
        correlations[overlap_idx, 0], correlations[overlap_idx, 1],
        marker=dim2_marker, color=COLOR_DIM2, s=200,
    )

for idx in np.unique(np.concatenate([list(top_dim1), list(top_dim2)])):
    ax.text(
        correlations[idx, 0], correlations[idx, 1], str(idx),
        fontsize=9, ha="center", va="center", color="white", weight="bold",
    )

ax.set_xlabel("Correlation with DIM1")
ax.set_ylabel("Correlation with DIM2")
ax.set_title("IML Neurons Correlated with DIMs")
fig.tight_layout()
plt.show()