"""
Manifold-based vs. single-neuron blood pressure (BP) decoding.

For each trial (one .mat file per animal), this script compares three
ways of predicting the blood pressure trace from IML neural activity:

    1. Manifold decoding  - linear regression on the 2D Isomap
                              coordinates (iso_0, iso_1).
    2. Phase decoding      - linear regression on the angular phase
                              (arctan2) of the manifold trajectory.
    3. Single-neuron decoding - linear regression fit independently
                              on each neuron's firing rate, then
                              averaged across neurons.

Decoding accuracy is compared using R^2, mean absolute error (MAE),
mean squared error (MSE), and mutual information (MI) with BP,
using paired t-tests across trials (manifold vs. single-neuron).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# One .mat file per animal/trial. Each file is expected to contain:
#   bpRef  - blood pressure trace,        shape (1, T)
#   iso_0  - Isomap dimension 1,          shape (1, T)
#   iso_1  - Isomap dimension 2,          shape (1, T)
#   frRef  - per-neuron firing rate,      shape (n_units, T)
FILE_PATHS = [
    "save_iso&ref_fr/1108_SC_1_40Hz.mat",
    "save_iso&ref_fr/1126_SC_1_40Hz.mat",
    "save_iso&ref_fr/0220_SC_1_40Hz.mat",
    "save_iso&ref_fr/0220_SC_2_40Hz.mat",
    "save_iso&ref_fr/1101_SC_1_40Hz.mat",
    "save_iso&ref_fr/1108_SC_1_40Hz.mat",
    "save_iso&ref_fr/0927_SC_1_40Hz.mat",
    "save_iso&ref_fr/0402_SC_2_40Hz.mat",
]

# Number of samples the neural signal is shifted forward to correct for
# a fixed neuro-hemodynamic latency relative to BP.
LATENCY_SHIFT = 10

COLOR_MANIFOLD = "#c22e63"
COLOR_NEURON = "#2d0052"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shift_forward(signal: np.ndarray, shift: int) -> np.ndarray:
    """Shift a 1D signal forward by `shift` samples, padding with its own
    first `shift` values and trimming the tail so the length is preserved.
    Used to correct for the latency between neural activity and BP.
    """
    return np.concatenate((signal[:shift], signal))[:-shift]


def load_trial(file_path: str):
    """Load one trial and return latency-corrected, min-max normalized
    BP, manifold coordinates, and per-neuron firing rates.
    """
    data = loadmat(file_path)
    scaler = MinMaxScaler()

    bp = scaler.fit_transform(data["bpRef"][0].reshape(-1, 1)).flatten()
    iso_0 = scaler.fit_transform(data["iso_0"][0].reshape(-1, 1)).flatten()
    iso_1 = scaler.fit_transform(data["iso_1"][0].reshape(-1, 1)).flatten()
    fr = np.array(data["frRef"])

    iso_0 = shift_forward(iso_0, LATENCY_SHIFT)
    iso_1 = shift_forward(iso_1, LATENCY_SHIFT)

    return bp, iso_0, iso_1, fr


def decode_from_manifold(iso_0: np.ndarray, iso_1: np.ndarray, bp: np.ndarray):
    """Linear regression of BP on the 2D manifold coordinates."""
    x = np.column_stack((iso_0, iso_1))
    model = LinearRegression().fit(x, bp)
    return model.predict(x)


def decode_from_phase(iso_0: np.ndarray, iso_1: np.ndarray, bp: np.ndarray):
    """Linear regression of BP on the manifold's angular phase."""
    phase = np.arctan2(iso_1, iso_0).reshape(-1, 1)
    model = LinearRegression().fit(phase, bp)
    return model.predict(phase)


def decode_from_single_neurons(fr: np.ndarray, bp: np.ndarray):
    """Fit one linear regression per neuron, then average the predictions
    and the per-neuron mutual information with BP.
    """
    predictions, mutual_infos = [], []
    for unit_rate in fr:
        unit_rate = unit_rate.reshape(-1, 1)
        model = LinearRegression().fit(unit_rate, bp)
        predictions.append(model.predict(unit_rate))
        mutual_infos.append(
            mutual_info_regression(unit_rate, bp, discrete_features=False)[0]
        )
    mean_prediction = np.mean(predictions, axis=0)
    mean_mutual_info = np.mean(mutual_infos)
    return mean_prediction, mean_mutual_info


def star_label(p_value: float) -> str:
    """Convert a p-value to a conventional significance label."""
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Run decoding for every trial
# ---------------------------------------------------------------------------

records = []
predictions_by_file = {}

for file_path in FILE_PATHS:
    bp, iso_0, iso_1, fr = load_trial(file_path)

    pred_manifold = decode_from_manifold(iso_0, iso_1, bp)
    pred_phase = decode_from_phase(iso_0, iso_1, bp)
    pred_neurons, mean_mi_neurons = decode_from_single_neurons(fr, bp)
    mi_manifold = np.mean(
        mutual_info_regression(
            np.column_stack((iso_0, iso_1)), bp, discrete_features=False
        )
    )

    records.append(
        {
            "file": os.path.basename(file_path),
            "r2_manifold": r2_score(bp, pred_manifold),
            "r2_phase": r2_score(bp, pred_phase),
            "r2_neurons": r2_score(bp, pred_neurons),
            "mae_manifold": mean_absolute_error(bp, pred_manifold),
            "mae_phase": mean_absolute_error(bp, pred_phase),
            "mae_neurons": mean_absolute_error(bp, pred_neurons),
            "mse_manifold": mean_squared_error(bp, pred_manifold),
            "mse_neurons": mean_squared_error(bp, pred_neurons),
            "mi_manifold": mi_manifold,
            "mi_neurons": mean_mi_neurons,
        }
    )

    # Keep the raw traces so the final time series plot can be produced
    # without reloading the .mat files.
    predictions_by_file[os.path.basename(file_path)] = {
        "bp": bp,
        "pred_manifold": pred_manifold,
        "pred_neurons": pred_neurons,
    }

results = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Paired statistics: manifold vs. single-neuron decoding
# ---------------------------------------------------------------------------

p_values = {
    "Prediction Accuracy (R\u00b2)": ttest_rel(
        results["r2_manifold"], results["r2_neurons"]
    ).pvalue,
    "Prediction Error (MAE)": ttest_rel(
        results["mae_manifold"], results["mae_neurons"]
    ).pvalue,
    "Mutual Information": ttest_rel(
        results["mi_manifold"], results["mi_neurons"]
    ).pvalue,
}

for label, p_value in p_values.items():
    print(f"{label}: p = {p_value:.4g} ({star_label(p_value)})")

# ---------------------------------------------------------------------------
# Summary figure: manifold vs. single-neuron decoding across trials
# ---------------------------------------------------------------------------

def plot_paired_comparison(ax, manifold_values, neuron_values, title, ylabel):
    """Paired dot-and-line plot with a bar for the group mean, used for
    every manifold-vs-neuron metric comparison in the summary figure.
    """
    bar_positions = [0, 0.35]
    means = [np.mean(manifold_values), np.mean(neuron_values)]
    ax.bar(
        bar_positions,
        means,
        color=[COLOR_MANIFOLD, COLOR_NEURON],
        alpha=0.6,
        width=0.35,
    )
    for manifold_value, neuron_value in zip(manifold_values, neuron_values):
        ax.plot(
            bar_positions,
            [manifold_value, neuron_value],
            color="gray",
            alpha=0.3,
            zorder=1,
        )
        ax.scatter(
            bar_positions,
            [manifold_value, neuron_value],
            color=[COLOR_MANIFOLD, COLOR_NEURON],
            s=25,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            zorder=3,
        )
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(["Manifold", "Neurons"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{star_label(p_values[title])}", fontsize=12)


fig, axes = plt.subplots(1, 3, figsize=(13, 5))
plot_paired_comparison(
    axes[0], results["r2_manifold"], results["r2_neurons"],
    "Prediction Accuracy (R\u00b2)", "R\u00b2",
)
plot_paired_comparison(
    axes[1], results["mae_manifold"], results["mae_neurons"],
    "Prediction Error (MAE)", "MAE",
)
plot_paired_comparison(
    axes[2], results["mi_manifold"], results["mi_neurons"],
    "Mutual Information", "MI",
)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Per-trial time series: actual BP vs. manifold vs. mean single-neuron
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    len(predictions_by_file), 1, figsize=(10, 2 * len(predictions_by_file)),
    sharex=True,
)
for ax, (file_name, trace) in zip(axes, predictions_by_file.items()):
    ax.plot(trace["bp"], "k-", linewidth=1.5, label="Actual BP")
    ax.plot(
        trace["pred_manifold"], color=COLOR_MANIFOLD, linewidth=2,
        label="Manifold prediction",
    )
    ax.plot(
        trace["pred_neurons"], color=COLOR_NEURON, linewidth=1.5,
        label="Mean single-neuron prediction",
    )
    ax.set_title(file_name)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time")
fig.tight_layout()
plt.show()