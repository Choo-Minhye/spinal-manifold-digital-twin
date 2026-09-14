"""
Side-by-side comparison of the raw (unaligned) circular manifolds
from all seven animals, supporting Figure 5A.

Each animal's 2D Isomap trajectory is plotted on its own, spaced out
along the x-axis so that differences in size, orientation, and
position across animals are visible without 3D perspective
distortion or overlapping trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_PATHS = [
    "save_iso&ref_fr/0908_SC_4_40Hz.mat",
    "save_iso&ref_fr/0214_SC_1_40Hz.mat",
    "save_iso&ref_fr/0818_SC_1_40Hz.mat",
    "save_iso&ref_fr/1101_SC_1_40Hz.mat",
    "save_iso&ref_fr/1108_SC_1_40Hz.mat",
    "save_iso&ref_fr/0927_SC_1_40Hz.mat",
    "save_iso&ref_fr/0402_SC_2_40Hz.mat",
]

# Original color assigned to each animal (R1 through R7), kept
# consistent with the other manifold figures in the manuscript.
ANIMAL_COLORS = [
    "steelblue", "royalblue", "gold", "moccasin", "skyblue", "powderblue", "lightseagreen",
]

# Left-to-right display order, given as animal numbers (R1 = first file, etc.),
# chosen so that visually similar manifolds are not placed next to each other.
DISPLAY_ORDER = [2, 1, 7, 4, 3, 5, 6]

OFFSET_STEP = 60   # x-axis spacing between animals, in Isomap coordinate units


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

iso_0_by_animal = []
iso_1_by_animal = []

for file_path in FILE_PATHS:
    loaded = loadmat(file_path)
    iso_0_by_animal.append(np.array(loaded["iso_0"])[0])
    iso_1_by_animal.append(np.array(loaded["iso_1"])[0])


# ---------------------------------------------------------------------------
# Figure 5A - raw manifolds, side by side
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 4))

for position, animal_number in enumerate(DISPLAY_ORDER):
    animal_idx = animal_number - 1   # R1 -> index 0, R2 -> index 1, ...
    offset = position * OFFSET_STEP
    color = ANIMAL_COLORS[animal_idx]

    x = iso_0_by_animal[animal_idx] + offset
    y = iso_1_by_animal[animal_idx]

    ax.scatter(x, y, color=color, edgecolors="black", linewidths=0.1, s=50)
    ax.plot(x, y, color="black", linewidth=0.3, alpha=0.5)

ax.set_aspect("equal")   # keeps each circular trajectory visually circular
ax.axis("off")

legend_handles = [
    plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor=ANIMAL_COLORS[r - 1], markersize=8, label=f"R{r}",
    )
    for r in range(1, len(FILE_PATHS) + 1)
]
ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.tight_layout()
plt.show()