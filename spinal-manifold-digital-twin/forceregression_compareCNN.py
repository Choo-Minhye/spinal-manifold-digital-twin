"""
Convolutional neural network (CNN) baseline for decoding blood pressure
(BP) from the 2D Isomap latent manifold, supporting the CNN baseline in
Figure 6D (compared there against linear regression and the FORCE-learned
recurrent reservoir model).

For each trial, a small 1D CNN is trained to map the 2D manifold
trajectory (iso_0, iso_1) to BP, with an 80/20 train/validation split
and early stopping on validation loss. Predictions, ground truth, R^2,
and MSE for every trial are saved to a single .mat file.
"""

import os

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# NOTE: '0402_SC_2_20Hz.mat' appears twice in this list (once mid-list,
# once at the end) - please confirm whether that trial should be counted
# once or twice before this goes into the repository; it has been left
# exactly as provided rather than silently deduplicated.
FILE_PATHS = [
    "save_iso&ref_fr/1126_SC_1_10Hz.mat",
    "save_iso&ref_fr/0402_SC_2_20Hz.mat",
    "save_iso&ref_fr/0908_SC_4_40Hz.mat",
    "save_iso&ref_fr/0214_SC_1_40Hz.mat",
    "save_iso&ref_fr/0818_SC_1_40Hz.mat",
    "save_iso&ref_fr/1101_SC_1_20Hz.mat",
    "save_iso&ref_fr/1108_SC_1_40Hz.mat",
    "save_iso&ref_fr/0927_SC_1_40Hz.mat",
    "save_iso&ref_fr/0402_SC_2_20Hz.mat",
]

TARGET_LENGTH = 240      # samples per trial after resampling
TRAIN_FRACTION = 0.8
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 7
EARLY_STOP_MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3

OUTPUT_DIR = "save_regression_data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "force_regression_CNN_.mat")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNNRegressor(nn.Module):
    """1D CNN mapping a 2-channel input (iso_0, iso_1) to a 1-channel BP
    trace of the same length.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):   # x: [batch, 2, T]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x.squeeze(1)   # [batch, T]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interpolate_to_length(data: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a 1D array to a fixed target length."""
    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, target_length)
    return interp1d(x_old, data, kind="linear")(x_new)


def load_trial(file_path: str, target_length: int):
    """Load one trial and resample the manifold coordinates and BP to a
    common length, then standardize both.
    """
    data = io.loadmat(file_path)

    iso_0 = interpolate_to_length(data["iso_0"].flatten(), target_length)
    iso_1 = interpolate_to_length(data["iso_1"].flatten(), target_length)
    bp = interpolate_to_length(data["bpRef"].flatten(), target_length)

    x = np.stack([iso_0, iso_1], axis=0)   # shape: (2, T)
    x = StandardScaler().fit_transform(x.T).T
    y = StandardScaler().fit_transform(bp.reshape(-1, 1)).flatten()

    return x, y


def train_with_early_stopping(model: nn.Module, x_train, y_train, x_val, y_val):
    """Train `model` with Adam + MSE loss, stopping when validation loss
    fails to improve by at least EARLY_STOP_MIN_DELTA for
    EARLY_STOP_PATIENCE consecutive epochs.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    stopped_epoch = MAX_EPOCHS - 1

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(x_train).squeeze(), y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val).squeeze(), y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if best_val_loss - val_loss.item() > EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss.item()
            best_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                stopped_epoch = epoch
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, best_val_loss, stopped_epoch


# ---------------------------------------------------------------------------
# Train and evaluate the CNN baseline on every trial
# ---------------------------------------------------------------------------

n_trials = len(FILE_PATHS)
fig_pred, axes_pred = plt.subplots(3, 3, figsize=(15, 10))
fig_loss, axes_loss = plt.subplots(3, 3, figsize=(15, 10))
axes_pred, axes_loss = axes_pred.flatten(), axes_loss.flatten()

predictions, ground_truths, r2_scores, mse_scores = [], [], [], []
stopped_epochs, best_val_losses = [], []

for idx, file_path in enumerate(FILE_PATHS):
    x, y = load_trial(file_path, TARGET_LENGTH)

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    train_len = int(TRAIN_FRACTION * x_tensor.shape[-1])
    x_train, y_train = x_tensor[:, :, :train_len], y_tensor[:, :train_len]
    x_val, y_val = x_tensor[:, :, train_len:], y_tensor[:, train_len:]

    model = CNNRegressor()
    train_losses, val_losses, best_val_loss, stopped_epoch = train_with_early_stopping(
        model, x_train, y_train, x_val, y_val
    )
    stopped_epochs.append(stopped_epoch)
    best_val_losses.append(best_val_loss)

    with torch.no_grad():
        full_prediction = model(x_tensor).squeeze().numpy()

    predictions.append(full_prediction)
    ground_truths.append(y)
    r2_scores.append(r2_score(y, full_prediction))
    mse_scores.append(mean_squared_error(y, full_prediction))

    axes_pred[idx].plot(y, label="True", color="black")
    axes_pred[idx].plot(full_prediction, label="Pred", color="red")
    axes_pred[idx].set_title(f"Trial {idx + 1}")
    axes_pred[idx].legend()

    axes_loss[idx].plot(train_losses, label="Train Loss")
    axes_loss[idx].plot(val_losses, label="Val Loss")
    axes_loss[idx].set_title(f"Loss Curve {idx + 1}")
    axes_loss[idx].legend()

print("\nEarly stopping summary:")
for i, (epoch, val_loss) in enumerate(zip(stopped_epochs, best_val_losses)):
    print(f"  Trial {i + 1}: stopped at epoch {epoch}, best val loss = {val_loss:.4f}")

fig_pred.tight_layout()
fig_loss.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# Save predictions and metrics for all trials
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
io.savemat(
    OUTPUT_PATH,
    {
        "CNN_predictive": predictions,
        "CNN_groundtruth": ground_truths,
        "CNN_accuracy": r2_scores,
        "CNN_mse": mse_scores,
    },
)
print(f"Saved CNN baseline results to: {OUTPUT_PATH}")