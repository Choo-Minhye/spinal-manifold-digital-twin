"""
Comparison of Isomap against PCA, LLE, and an autoencoder as the
dimensionality-reduction method for IML population activity, supporting
the Supplementary Figure comparing dimensionality-reduction methods.

Panel A - 3D latent trajectories from each method, colored by time and
           by blood pressure (BP).
Panel B - Persistent homology (H1/H2) of each method's latent space,
           evaluated on a single representative trial.
Panels D/E - Geodesic distance preservation: how well each method's
           embedded Euclidean distances match the geodesic distances
           of the original high-dimensional firing-rate data, both as
           a single summary number and as a function of the number of
           latent dimensions retained.
Panels C/F - Dim1+Dim2 -> BP: an in-sample linear regression of blood
           pressure on each method's first two latent dimensions.

Two earlier candidate metrics for the "why Isomap" comparison -
trustworthiness/continuity (local neighborhood preservation) and a
circularity / circular-linear phase-alignment score - were explored
but are not included here: local-neighborhood scores were similar
across all four methods (they do not distinguish "clean loop" from
"broken path"), and circularity/phase alignment was a secondary,
single-trial-only illustration. Geodesic preservation directly tests
what Isomap is designed to optimize and is the metric that was kept.

The geodesic-preservation and BP-fit sections each save their result
for one trial to disk; run this script once per trial/animal, then
aggregate the saved files across trials (paired comparison across
n = 7 animals, as reported in the manuscript).
"""

import os
import pickle
import itertools
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from ripser import ripser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_PATH = "save_iso&ref_fr/0214_SC_1_40Hz.mat"   # trial being analyzed
TRIAL_ID = "0214_SC_1"

N_LATENT = 6            # dimensionality used for Isomap/PCA/LLE/Autoencoder
ANALYSIS_DIMS = 6       # dimensions used for persistent homology / geodesic analysis
GEODESIC_N_NEIGHBORS = 5   # matches Isomap's default n_neighbors
LATENCY_SHIFT = 10         # samples; corrects neuro-hemodynamic latency

AE_RANDOM_SEED = 42
AE_HIDDEN_DIM = 32
AE_EPOCHS = 1000
AE_LEARNING_RATE = 1e-3

METHOD_ORDER = ["Isomap", "PCA", "LLE", "Autoencoder"]
HIGHLIGHT_COLOR = "#2f6fb3"   # Isomap
GRAY_COLOR = "#a6a6a6"        # every other method

RESULTS_DIR_GEODESIC = os.path.join("..", "RESULTS", "geodesic_preservation")
RESULTS_DIR_BP_FIT = os.path.join("..", "RESULTS", "bp_dim2_regression")


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.linewidth": 1.1,
        "axes.edgecolor": "#000000",
    }
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

data = loadmat(FILE_PATH)
fr = np.array(data["frRef"])            # shape: (n_neurons, T)
bp = np.array(data["bpRef"]).flatten()
t = np.array(data["tRef"]).flatten()

firing_rate = fr.T   # Isomap/PCA/LLE/Autoencoder expect (time, neurons)


# ---------------------------------------------------------------------------
# Fit each dimensionality-reduction method
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def fit_autoencoder(X: np.ndarray, latent_dim: int) -> np.ndarray:
    """Train a simple autoencoder and return its latent representation."""
    np.random.seed(AE_RANDOM_SEED)
    torch.manual_seed(AE_RANDOM_SEED)
    device = torch.device("cpu")

    model = Autoencoder(X.shape[1], AE_HIDDEN_DIM, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LEARNING_RATE)
    criterion = nn.MSELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    model.train()
    for epoch in range(AE_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), X_tensor)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"AE epoch {epoch + 1}/{AE_EPOCHS}, loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        return model.encoder(X_tensor).cpu().numpy()


fr_isomap = Isomap(n_components=N_LATENT).fit_transform(firing_rate)
fr_pca = PCA(n_components=N_LATENT).fit_transform(firing_rate)
fr_lle = LocallyLinearEmbedding(
    n_neighbors=5, n_components=N_LATENT, method="standard", eigen_solver="auto"
).fit_transform(firing_rate)
fr_ae = fit_autoencoder(firing_rate, N_LATENT)

methods = {"Isomap": fr_isomap, "PCA": fr_pca, "LLE": fr_lle, "Autoencoder": fr_ae}

print("Latent representation shapes:")
for name, latent in methods.items():
    print(f"  {name:12s}: {latent.shape}")


# ---------------------------------------------------------------------------
# Panel A - 3D latent trajectories, colored by time and by BP
# ---------------------------------------------------------------------------

def plot_trajectories(color_values: np.ndarray, cmap: str, colorbar_label: str, title: str):
    fig = plt.figure(figsize=(24, 6))
    for i, (name, latent) in enumerate(methods.items()):
        ax = fig.add_subplot(1, len(methods), i + 1, projection="3d")
        scatter = ax.scatter(
            latent[:, 0], latent[:, 1], latent[:, 2],
            c=color_values, cmap=cmap, s=50, edgecolors="black", linewidths=0.2, alpha=0.8,
        )
        ax.view_init(elev=90, azim=45)
        ax.set_title(name, fontsize=16)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()
    fig.colorbar(scatter, ax=fig.axes, shrink=0.55, pad=0.02, label=colorbar_label)
    fig.suptitle(title, fontsize=18)
    return fig


plot_trajectories(t, "viridis", "Time (sec)", "3-D Latent Trajectories Colored by Time")
plot_trajectories(bp, "viridis", "Blood Pressure", "3-D Latent Trajectories Colored by BP")


# ---------------------------------------------------------------------------
# Panel B - persistent homology, single trial
# ---------------------------------------------------------------------------

def normalize_latent_space(latent: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Center and scale a latent representation so that different methods'
    persistence diagrams are comparable regardless of their native scale.
    """
    centered = latent - np.mean(latent, axis=0, keepdims=True)
    rms_radius = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    return centered if rms_radius < eps else centered / rms_radius


homology_results = {}
for name in METHOD_ORDER:
    latent_normalized = normalize_latent_space(methods[name][:, :ANALYSIS_DIMS])
    diagrams = ripser(latent_normalized, maxdim=2, coeff=2)["dgms"]
    h1 = diagrams[1]

    if len(h1) > 0:
        lifetimes = h1[:, 1] - h1[:, 0]
        lifetimes = np.sort(lifetimes[np.isfinite(lifetimes)])[::-1]
    else:
        lifetimes = np.array([])

    longest_h1 = lifetimes[0] if len(lifetimes) > 0 else 0.0
    second_h1 = lifetimes[1] if len(lifetimes) > 1 else 0.0

    homology_results[name] = {
        "h0": diagrams[0], "h1": h1, "h2": diagrams[2],
        "longest_h1": longest_h1, "h1_gap": longest_h1 - second_h1,
    }

print("\nPersistent homology (single trial):")
for name in METHOD_ORDER:
    r = homology_results[name]
    print(f"  {name:12s}: longest H1 = {r['longest_h1']:.4f}, H1 gap = {r['h1_gap']:.4f}")

# Common x-axis so all four barcodes are directly comparable
all_finite = [
    v for name in METHOD_ORDER for dim in ("h1", "h2")
    for v in homology_results[name][dim][np.isfinite(homology_results[name][dim][:, 1])].flatten()
    if len(homology_results[name][dim]) > 0
]
common_xmax = max(all_finite) * 1.05 if all_finite else 1.0

fig, axes = plt.subplots(2, len(METHOD_ORDER), figsize=(20, 8), sharex=True)
for col, name in enumerate(METHOD_ORDER):
    for row, (dim_key, color) in enumerate((("h1", "indianred"), ("h2", "maroon"))):
        ax = axes[row, col]
        barcode = homology_results[name][dim_key]
        finite = barcode[np.isfinite(barcode[:, 1])] if len(barcode) > 0 else barcode
        if len(finite) > 0:
            order = np.argsort(finite[:, 1] - finite[:, 0])
            for bar_idx, interval in enumerate(finite[order]):
                ax.plot([interval[0], interval[1]], [bar_idx, bar_idx], color=color, linewidth=2.5)
        ax.set_xlim(0, common_xmax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if row == 0:
            ax.set_title(name, fontsize=14)
        if col == 0:
            ax.set_ylabel(f"{dim_key.upper()} features")
axes[1, 0].set_xlabel("Filtration value")
fig.suptitle("Persistent Homology of Latent Representations", fontsize=17)
fig.tight_layout(rect=[0, 0, 1, 0.95])


# ---------------------------------------------------------------------------
# Panels D/E - geodesic distance preservation
# ---------------------------------------------------------------------------

def geodesic_preservation(latent: np.ndarray, geo_flat: np.ndarray, finite_mask: np.ndarray, upper_idx):
    """Pearson correlation and residual variance between geodesic distance
    (computed once from the original high-dimensional data) and each
    method's embedded Euclidean distance. Higher correlation / r^2 means
    the embedding better preserves the original geodesic structure.
    """
    embed_flat = squareform(pdist(latent, metric="euclidean"))[upper_idx]
    r = np.corrcoef(geo_flat[finite_mask], embed_flat[finite_mask])[0, 1]
    return r, 1 - r ** 2


# Geodesic ground truth: built once from the original firing-rate data,
# independent of any dimensionality-reduction method.
knn_graph = kneighbors_graph(
    firing_rate, n_neighbors=GEODESIC_N_NEIGHBORS, mode="distance", include_self=False
)
knn_graph = knn_graph.maximum(knn_graph.T)
geo_distances = shortest_path(knn_graph, method="D", directed=False)

n_points = geo_distances.shape[0]
upper_idx = np.triu_indices(n_points, k=1)
geo_flat = geo_distances[upper_idx]
finite_mask = np.isfinite(geo_flat)
n_unreachable = np.sum(np.isinf(geo_flat))
if n_unreachable > 0:
    print(
        f"WARNING: {n_unreachable} point pairs are disconnected in the "
        f"k={GEODESIC_N_NEIGHBORS} neighbor graph and excluded from the correlation below."
    )

geo_residvar = {}
for name in METHOD_ORDER:
    _, resid_var = geodesic_preservation(
        methods[name][:, :ANALYSIS_DIMS], geo_flat, finite_mask, upper_idx
    )
    geo_residvar[name] = resid_var

print(f"\nGeodesic distance preservation (d={ANALYSIS_DIMS}):")
for name in METHOD_ORDER:
    print(f"  {name:12s}: preservation r^2 = {1 - geo_residvar[name]:.4f}")

method_colors = [HIGHLIGHT_COLOR if m == "Isomap" else GRAY_COLOR for m in METHOD_ORDER]
x_positions = np.arange(len(METHOD_ORDER))

# Panel D: single-number summary at the full analysis dimensionality
preservation_values = np.array([1 - geo_residvar[m] for m in METHOD_ORDER])
fig, ax = plt.subplots(figsize=(4.2, 4.5))
ax.bar(x_positions, preservation_values, color=method_colors, edgecolor="black", linewidth=1.2, width=0.62)
for x, v in zip(x_positions, preservation_values):
    ax.text(x, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
ax.set_xticks(x_positions); ax.set_xticklabels(METHOD_ORDER)
ax.set_ylabel("Geodesic preservation ($r^2$)")
ax.set_title("Geodesic distance\npreservation", fontweight="bold")
ax.set_ylim(0, 1.12)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()

# Panel E: preservation vs. number of latent dimensions retained
dim_range = np.arange(1, ANALYSIS_DIMS + 1)
line_styles = {"Isomap": "-", "PCA": "--", "LLE": ":", "Autoencoder": "-."}
markers = {"Isomap": "o", "PCA": "s", "LLE": "^", "Autoencoder": "D"}
gray_shades = {"PCA": "#8c8c8c", "LLE": "#bdbdbd", "Autoencoder": "#6e6e6e"}

curves = {name: [] for name in METHOD_ORDER}
for d in dim_range:
    for name in METHOD_ORDER:
        _, resid_var = geodesic_preservation(methods[name][:, :d], geo_flat, finite_mask, upper_idx)
        curves[name].append(1 - resid_var)

fig, ax = plt.subplots(figsize=(5.6, 4.6))
for name in METHOD_ORDER:
    is_isomap = name == "Isomap"
    ax.plot(
        dim_range, curves[name], marker=markers[name],
        markersize=6 if is_isomap else 5, linestyle=line_styles[name],
        color=HIGHLIGHT_COLOR if is_isomap else gray_shades[name],
        linewidth=2.4 if is_isomap else 1.6, markeredgecolor="black", markeredgewidth=0.6,
    )
    ax.text(
        dim_range[-1] + 0.12, curves[name][-1], name,
        color=HIGHLIGHT_COLOR if is_isomap else "#4d4d4d",
        va="center", fontweight="bold" if is_isomap else "normal", fontsize=10.5,
    )
ax.set_xlabel("Number of latent dimensions")
ax.set_ylabel("Geodesic preservation ($r^2$)")
ax.set_title("Preservation vs. dimensionality", fontweight="bold")
ax.set_xticks(dim_range)
ax.set_xlim(dim_range[0] - 0.3, dim_range[-1] + 1.5)
ax.set_ylim(0, 1.12)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()

# Save this trial's result for cross-trial aggregation (n = 7 paired comparison)
os.makedirs(RESULTS_DIR_GEODESIC, exist_ok=True)
with open(os.path.join(RESULTS_DIR_GEODESIC, f"{TRIAL_ID}.pkl"), "wb") as f:
    pickle.dump(
        {
            "trial_id": TRIAL_ID,
            "geodesic_n_neighbors": GEODESIC_N_NEIGHBORS,
            "analysis_dims": ANALYSIS_DIMS,
            "preservation_summary": {m: 1 - geo_residvar[m] for m in METHOD_ORDER},
            "dim_range": dim_range.tolist(),
            "dimensionality_curves": curves,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
        f,
    )


# ---------------------------------------------------------------------------
# Panels C/F - Dim1+Dim2 -> BP (in-sample linear regression)
#
# In-sample R^2 measures how well a straight line through each method's
# first two latent dimensions tracks BP shape-wise; it is not a test of
# generalization to unseen timepoints.
# ---------------------------------------------------------------------------

def shift_forward(signal: np.ndarray, shift: int) -> np.ndarray:
    """Correct the timing offset between the neural trajectory and BP."""
    return np.concatenate((signal[:shift], signal))[:-shift]


bp_fit_results = {}
for name in METHOD_ORDER:
    dim1 = MinMaxScaler().fit_transform(methods[name][:, 0].reshape(-1, 1)).flatten()
    dim2 = MinMaxScaler().fit_transform(methods[name][:, 1].reshape(-1, 1)).flatten()
    bp_scaled = MinMaxScaler().fit_transform(bp.reshape(-1, 1)).flatten()

    dim1 = shift_forward(dim1, LATENCY_SHIFT)
    dim2 = shift_forward(dim2, LATENCY_SHIFT)

    X = np.column_stack((dim1, dim2))
    model = LinearRegression().fit(X, bp_scaled)
    prediction = model.predict(X)

    bp_fit_results[name] = {
        "r2": r2_score(bp_scaled, prediction),
        "mse": mean_squared_error(bp_scaled, prediction),
        "y_true": bp_scaled,
        "y_pred": prediction,
    }

print("\nDim1+Dim2 -> BP (in-sample linear regression):")
for name in METHOD_ORDER:
    r = bp_fit_results[name]
    print(f"  {name:12s}: R^2 = {r['r2']:.4f}, MSE = {r['mse']:.4f}")

# Panel F: actual vs. predicted BP, one panel per method
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, name in zip(axes.flatten(), METHOD_ORDER):
    r = bp_fit_results[name]
    line_color = HIGHLIGHT_COLOR if name == "Isomap" else GRAY_COLOR
    ax.plot(r["y_true"], color="black", linewidth=1.2, label="Actual BP")
    ax.plot(r["y_pred"], color=line_color, linewidth=1.8, label="Predicted BP")
    ax.set_title(f"{name}\n$R^2$ = {r['r2']:.3f}, MSE = {r['mse']:.4f}")
    ax.set_xlabel("Time (samples)"); ax.set_ylabel("BP (scaled 0-1)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("Dim1+Dim2 -> BP (in-sample fit)", fontsize=14, fontweight="bold")
fig.tight_layout()

# Panel C: R^2 summary bar plot
r2_values = np.array([bp_fit_results[m]["r2"] for m in METHOD_ORDER])
fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(x_positions, r2_values, color=method_colors, edgecolor="black", linewidth=1.2, width=0.6)
for x, v in zip(x_positions, r2_values):
    ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x_positions); ax.set_xticklabels(METHOD_ORDER)
ax.set_ylabel("In-sample $R^2$ (Dim1+Dim2 -> BP)")
ax.set_title("This trial", fontweight="bold")
ax.set_ylim(0, 1.1)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()

# Save this trial's result for cross-trial aggregation (n = 7 paired comparison)
os.makedirs(RESULTS_DIR_BP_FIT, exist_ok=True)
with open(os.path.join(RESULTS_DIR_BP_FIT, f"{TRIAL_ID}.pkl"), "wb") as f:
    pickle.dump(
        {
            "trial_id": TRIAL_ID,
            "bp_r2": {m: bp_fit_results[m]["r2"] for m in METHOD_ORDER},
            "bp_mse": {m: bp_fit_results[m]["mse"] for m in METHOD_ORDER},
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
        f,
    )

plt.show()