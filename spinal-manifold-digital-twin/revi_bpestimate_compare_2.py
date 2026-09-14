"""
Minimal-manifold reconstruction and distance-correlation analysis
supporting Figures 3A and 4B-D.

Figure 3A  - Full-population Isomap manifold, colored by blood
             pressure (BP).
Figure 4B  - A minimal circular manifold reconstructed from six
             selected neurons, and the neural phase derived from it.
Figure 4C  - Distance correlation (dCor) between BP and either the
             manifold phase or the six neurons' firing rates, computed
             in each of six time bins.
Figure 4D  - Linear regression prediction of BP from the manifold
             (2D Isomap coordinates of the six neurons) versus from
             the six neurons' firing rates individually.

Note: the six neurons below were selected, as described in the
Method Details, for their high firing rates and strong contribution
to the Isomap Dim1/Dim2 axes; they are not re-derived in this script.
"""

import numpy as np
import matplotlib.pyplot as plt
import dcor
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_PATH = "save_iso&ref_fr/1108_SC_1_40Hz.mat"

# Six neurons selected for high firing rate and strong Dim1/Dim2
# contribution (see Method Details, Figure 4A).
SELECTED_NEURON_INDICES = [6, 12, 24, 11, 13, 2]

LATENCY_SHIFT = 10   # samples; corrects for neuro-hemodynamic latency
N_TIME_BINS = 6      # matches the six temporal bins used for dCor in Figure 4C

COLOR_MANIFOLD = "#c22e63"
COLOR_NEURON_MEAN = "#1b0c41"
COLOR_NEURON_RANGE = "#999999"
COLOR_HIGHLIGHT_LINE = "#fff6e5"   # light ivory outline drawn under the mean line


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def shift_forward(signal: np.ndarray, shift: int) -> np.ndarray:
    """Shift a 1D signal forward by `shift` samples to correct for the
    latency between neural activity and BP, padding with its own first
    values and trimming the tail so length is preserved.
    """
    return np.concatenate((signal[:shift], signal))[:-shift]


data = loadmat(FILE_PATH)
bp = np.array(data["bpRef"])[0]
fr = np.array(data["frRef"])          # shape: (n_neurons, T)

bp_norm = MinMaxScaler().fit_transform(bp.reshape(-1, 1)).flatten()
selected_fr = fr[SELECTED_NEURON_INDICES, :]   # shape: (6, T)


# ---------------------------------------------------------------------------
# Figure 3A - full-population Isomap manifold colored by BP
# ---------------------------------------------------------------------------

def plot_population_manifold(fr: np.ndarray, bp: np.ndarray):
    """3D Isomap embedding of every recorded neuron's firing rate,
    colored by blood pressure.
    """
    isomap = Isomap(n_components=3)
    embedding = isomap.fit_transform(fr.T)   # (T, 3)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=bp, cmap="gist_heat", edgecolors="black", linewidths=0.3,
    )
    ax.set_title("Isomap Manifold Colored by Blood Pressure")
    ax.axis("off")
    fig.colorbar(scatter, label="BP")
    fig.tight_layout()
    return embedding


population_embedding = plot_population_manifold(fr, bp)


# ---------------------------------------------------------------------------
# Figure 4B - minimal manifold reconstructed from the six selected neurons
# ---------------------------------------------------------------------------

def reconstruct_minimal_manifold(selected_fr: np.ndarray, n_components: int = 6):
    """Isomap embedding built from only the six selected neurons, plus
    the manifold's angular phase in the first two embedding dimensions.
    """
    normalized = np.array(
        [MinMaxScaler().fit_transform(unit.reshape(-1, 1)).flatten() for unit in selected_fr]
    )
    isomap = Isomap(n_components=n_components)
    embedding = isomap.fit_transform(normalized.T)   # (T, n_components)

    dim0 = shift_forward(
        MinMaxScaler().fit_transform(embedding[:, 0].reshape(-1, 1)).flatten(), LATENCY_SHIFT
    )
    dim1 = shift_forward(
        MinMaxScaler().fit_transform(embedding[:, 1].reshape(-1, 1)).flatten(), LATENCY_SHIFT
    )
    phase = np.arctan2(dim0, dim1)
    return embedding, dim0, dim1, phase


minimal_embedding, manifold_dim0, manifold_dim1, manifold_phase = reconstruct_minimal_manifold(
    selected_fr
)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    minimal_embedding[:, 0], minimal_embedding[:, 1], minimal_embedding[:, 2],
    c=bp, cmap="magma", edgecolors="black", linewidths=0.3, s=60,
)
ax.set_title("Minimal Manifold Reconstructed from Six Selected Neurons")
ax.axis("off")
fig.colorbar(scatter, label="BP")
fig.tight_layout()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(manifold_dim0, manifold_dim1, c=manifold_phase, cmap="twilight", s=15)
ax.set_title("Phase Trajectory of the Minimal Manifold (Color: Angle in Radians)")
ax.set_xlabel("Isomap Dimension 1")
ax.set_ylabel("Isomap Dimension 2")
ax.axis("equal")
fig.tight_layout()


# ---------------------------------------------------------------------------
# Figure 4C - distance correlation with BP, per time bin
# ---------------------------------------------------------------------------

def bin_edges(length: int, n_bins: int):
    bin_size = length // n_bins
    return [(i * bin_size, (i + 1) * bin_size) for i in range(n_bins)]


bins = bin_edges(len(bp_norm), N_TIME_BINS)

dcor_phase = []
dcor_by_neuron = {idx: [] for idx in SELECTED_NEURON_INDICES}

for start, end in bins:
    bp_bin = bp_norm[start:end]
    phase_bin = manifold_phase[start:end]
    dcor_phase.append(dcor.distance_correlation(phase_bin, bp_bin))
    for i, idx in enumerate(SELECTED_NEURON_INDICES):
        fr_bin_norm = MinMaxScaler().fit_transform(
            selected_fr[i, start:end].reshape(-1, 1)
        ).flatten()
        dcor_by_neuron[idx].append(dcor.distance_correlation(fr_bin_norm, bp_bin))

dcor_matrix = np.array([dcor_by_neuron[idx] for idx in SELECTED_NEURON_INDICES])
mean_dcor_neurons = np.nanmean(dcor_matrix, axis=0)
std_dcor_neurons = np.nanstd(dcor_matrix, axis=0)

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=False)

axes[0].plot(bp_norm, color="black", linewidth=1.5, label="Normalized BP")
for start, _ in bins:
    axes[0].axvline(start, color="gray", alpha=0.4)
axes[0].set_ylabel("Normalized BP")
axes[0].set_title("Blood Pressure Trace with Time Bins")

bin_x = np.arange(1, N_TIME_BINS + 1)
axes[1].bar(bin_x - 0.15, dcor_phase, width=0.3, color=COLOR_MANIFOLD, label="Manifold phase")
axes[1].bar(
    bin_x + 0.15, mean_dcor_neurons, width=0.3, color=COLOR_NEURON_MEAN,
    yerr=std_dcor_neurons, capsize=3, label="Mean of 6 neurons",
    error_kw=dict(ecolor="gray", alpha=0.5, elinewidth=1.2, capthick=1.2),
)
axes[1].set_xlabel("Time Bin")
axes[1].set_ylabel("Distance Correlation with BP")
axes[1].set_title("Distance Correlation with BP per Time Bin")
axes[1].legend(loc="upper right", frameon=False)
fig.tight_layout()


# ---------------------------------------------------------------------------
# Figure 4D - BP prediction from the manifold vs. from individual neurons
# ---------------------------------------------------------------------------

manifold_coords = np.column_stack((manifold_dim0, manifold_dim1))
manifold_model = LinearRegression().fit(manifold_coords, bp)
manifold_prediction = manifold_model.predict(manifold_coords)

neuron_predictions = []
for unit in selected_fr:
    unit_norm = shift_forward(
        MinMaxScaler().fit_transform(unit.reshape(-1, 1)).flatten(), LATENCY_SHIFT
    ).reshape(-1, 1)
    model = LinearRegression().fit(unit_norm, bp)
    neuron_predictions.append(model.predict(unit_norm))
neuron_predictions = np.array(neuron_predictions)   # (6, T)
mean_neuron_prediction = neuron_predictions.mean(axis=0)

print(f"Manifold R2:        {r2_score(bp, manifold_prediction):.4f}")
print(f"Mean-neuron R2:     {r2_score(bp, mean_neuron_prediction):.4f}")

# Panel 1: prediction from the manifold
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(manifold_prediction, color=COLOR_HIGHLIGHT_LINE, linewidth=13, label="_nolegend_")
ax.plot(manifold_prediction, color=COLOR_MANIFOLD, linewidth=5, label="Mean Prediction")
ax.plot(bp, "k--", linewidth=2, label="Actual BP")
ax.set_title("Prediction of BP from Manifold")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized BP")
ax.legend()
fig.tight_layout()

# Panel 2: prediction from the six individual neurons
fig, ax = plt.subplots(figsize=(6, 5))
lower, upper = neuron_predictions.min(axis=0), neuron_predictions.max(axis=0)
ax.fill_between(
    np.arange(len(lower)), lower, upper, color=COLOR_NEURON_RANGE, alpha=0.3,
    label="Range across 6 neurons",
)
ax.plot(mean_neuron_prediction, color=COLOR_HIGHLIGHT_LINE, linewidth=13, label="_nolegend_")
ax.plot(mean_neuron_prediction, color=COLOR_NEURON_MEAN, linewidth=5, label="Mean Prediction")
ax.plot(bp, "k--", linewidth=2, label="Actual BP")
ax.set_title("Predicted BP from Selected Neurons (Mean in Purple)")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized BP")
ax.legend()
fig.tight_layout()

plt.show()