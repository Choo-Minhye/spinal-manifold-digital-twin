"""
FORCE learning of a recurrent reservoir to decode blood pressure (BP)
from a 2D Isomap-projected latent manifold of IML population activity.

This script also reports wall-clock training time and basic hardware
info, used to support the computational-complexity analysis in the
manuscript's Methods section.
"""

import time
import platform

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def tanh_activation(x):
    """Nonlinear activation function (tanh) used for reservoir units."""
    return np.tanh(x)


def interpolate_to_length(data, target_length):
    """Linearly resample a 1D array to a fixed target length."""
    current_length = len(data)
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_old, data, kind="linear")
    return interpolator(x_new)


def force_learning(X, Y, N, Rank, tau, dt, itr):
    """
    Train a recurrent reservoir using the FORCE learning algorithm
    (recursive least squares update of the readout weights).

    Parameters
    ----------
    X : ndarray, shape (Rank, T)
        Latent input driving the reservoir (e.g., 2D Isomap coordinates).
    Y : ndarray, shape (T,)
        Target signal to be reconstructed from the reservoir (e.g., BP).
    N : int
        Number of reservoir units.
    Rank : int
        Dimensionality of the input/output (here, 2).
    tau : float
        Reservoir membrane time constant.
    dt : float
        Integration time step.
    itr : int
        Number of training iterations (full passes over the trial).

    Returns
    -------
    Wr : ndarray, shape (N, Rank)
        Learned readout weights.
    Wz : ndarray, shape (N, Rank)
        Fixed input weights.
    J : ndarray, shape (N, N)
        Fixed recurrent connectivity matrix.
    r_learning : ndarray, shape (itr * T, N)
        Reservoir activity recorded throughout training.
    x : ndarray, shape (N, T)
        Reservoir membrane potential at the end of training.
    """
    T = X.shape[1]
    x = np.zeros((N, T))
    g = 0.0  # gain factor; g = 0 corresponds to a non-chaotic reservoir
    J = np.random.randn(N, N) / np.sqrt(N)   # fixed recurrent connectivity
    Wz = np.random.randn(N, Rank)            # fixed input weights
    Wr = np.zeros((N, Rank))                 # readout weights (learned)
    P = np.eye(N) / 5.0                      # inverse correlation matrix (RLS)
    r_learning = np.zeros((itr * T, N))

    for j in range(itr):
        x[:, 1] = x[:, -1]
        x[:, 2:] = 0

        for i in range(T - 1):
            # Reservoir state update
            dx = (dt / tau) * (-x[:, i] + g * J @ tanh_activation(x[:, i]) + Wz @ X[:, i])
            x[:, i + 1] = x[:, i] + dx

            r = tanh_activation(x[:, i + 1])
            r_learning[j * T + i, :] = r

            # Recursive least squares (RLS) update of P and Wr
            P = P - P @ np.outer(r, r) @ P / (1 + r @ P @ r)
            error = Wr.T @ r - Y[i + 1:i + 1 + Rank].reshape(-1, 1).squeeze()
            Wr = Wr - P @ np.outer(r, error)

    return Wr, Wz, J, r_learning, x


# =====================================================================
# Run FORCE learning on a single trial and measure training time
# =====================================================================
FILE_PATH = "save_iso&ref_fr/0214_SC_1_40Hz.mat"  # update path as needed
TARGET_LENGTH = 240   # number of time points per trial after resampling
N_UNITS = 200          # number of reservoir units
RANK = 2               # input/output dimensionality (2D latent manifold)
N_ITERATIONS = 100     # number of training iterations

loaded_data = io.loadmat(FILE_PATH)

iso_0_data = loaded_data["iso_0"].flatten()   # latent dimension 1 (z1)
iso_1_data = loaded_data["iso_1"].flatten()   # latent dimension 2 (z2)
bp_data = loaded_data["bpRef"].flatten()      # blood pressure trace

# Resample all signals to a common length
iso_0_data = interpolate_to_length(iso_0_data, TARGET_LENGTH)
iso_1_data = interpolate_to_length(iso_1_data, TARGET_LENGTH)
bp_data = interpolate_to_length(bp_data, TARGET_LENGTH)

# Stack the two latent dimensions as the 2D input driving the reservoir
X = np.column_stack((iso_0_data, iso_1_data)).T   # shape: (Rank, T)
Y = bp_data
Y[TARGET_LENGTH - 1] = Y[TARGET_LENGTH - 2]        # guard against edge artifact

# Standardize input and target
scaler = StandardScaler()
X = scaler.fit_transform(X.T).T
Y = scaler.fit_transform(Y.reshape(-1, 1)).flatten()

# Report hardware used for the timing benchmark
print("=" * 60)
print(f"Processor: {platform.processor()}")
print(f"Machine:   {platform.machine()}")
print(f"System:    {platform.system()} {platform.release()}")
print("=" * 60)

# Measure wall-clock training time
start_time = time.perf_counter()
Wr, Wz, J, r_learning, final_reservoir_state = force_learning(
    X, Y, N=N_UNITS, Rank=RANK, tau=0.1, dt=0.1, itr=N_ITERATIONS
)
elapsed = time.perf_counter() - start_time

# Evaluate prediction quality
r_prediction = np.tanh(np.dot(Wz, X))
predicted_Y = np.dot(Wr.T, r_prediction).flatten()[:TARGET_LENGTH]

mse = mean_squared_error(Y, predicted_Y)
r2 = r2_score(Y, predicted_Y)

print(f"\nFile: {FILE_PATH}")
print(f"Reservoir size (N): {N_UNITS}, Time steps (T): {TARGET_LENGTH}, Iterations: {N_ITERATIONS}")
print(f"Training time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
print(f"R-squared: {r2:.4f}, MSE: {mse:.4f}")

# Plot actual vs. predicted BP
plt.figure(figsize=(8, 5))
plt.plot(Y, "k-", linewidth=1, label="Actual Data")
plt.plot(predicted_Y, color="red", linewidth=2, label="Predicted Data")
plt.xlabel("Time")
plt.ylabel("BP")
plt.title(f"R-squared: {r2:.4f}, Training time: {elapsed:.2f}s")
plt.legend()
plt.tight_layout()
plt.show()