import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import io
from sklearn.manifold import Isomap
from ripser import ripser as tda

from PYPACKAGE.plot import plot_data
from PYPACKAGE import cardiovascular_process as cdp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import io
from sklearn.manifold import Isomap

from PYPACKAGE import cardiovascular_process as cdp

# =============================================================================
# 0. Parameters
# =============================================================================
dt = 0.5
pre = 30
post = 90

if pre < 0:
    pre = -pre

# Number of samples per trial (same as the original script)
L = round(post / dt) + round(pre / dt)

# This single rat's session contains trials at all of these frequencies.
# Edit this list only to add/remove a condition from the comparison.
rat_numb = '0214_SC_1'
FREQ_LIST = [4, 10, 20, 40]

# Which matching trial to use if this rat has more than one trial at a given
# frequency. The original script used idx[1] (the second match), so that's
# kept as the default here.
TRIAL_INDEX = 1


# =============================================================================
# 1. Load this rat's data once (shared across all frequencies)
# =============================================================================
freq = io.loadmat(
    '..\\PREPROCESSED\\' + rat_numb + '\\stimInfo.mat'
)['stimFreq']
freq = freq.astype('int8')

bp_path = (
    '..\\DATA\\PROCESSED\\BP_DATA\\'
    + 'bp_resample_data_'
    + rat_numb
    + '_v7.mat'
)
bp_data, t = cdp.get_bp_with_time(bp_path)

fr = io.loadmat(
    '..\\DATA\\PROCESSED\\FR_DATA\\'
    + 'fr_resample_data_'
    + rat_numb
    + '_v7.mat'
)['FR']


def load_isomap_for_freq(select_freq, trial_index=TRIAL_INDEX):
    """Pull the trial(s) matching select_freq from this rat's session,
    and fit a 6D Isomap exactly as in the original script."""

    idx = np.where(freq[0] == select_freq)[0].astype('int8')
    if len(idx) == 0:
        raise ValueError(f'No trials at {select_freq} Hz found for rat {rat_numb}')

    use_idx = idx[trial_index] if len(idx) > trial_index else idx[0]

    tRef = t[use_idx * L:(use_idx + 1) * L]
    frRef = fr[:, use_idx * L:(use_idx + 1) * L]
    bpRef = bp_data[use_idx * L:(use_idx + 1) * L]

    firing_rate = frRef.transpose()  # Isomap input shape: [time points, neurons]

    isomap = Isomap(n_components=6)
    fr_isomap = isomap.fit_transform(firing_rate)

    return tRef, bpRef, fr_isomap


# =============================================================================
# 2. Run Isomap for every frequency
# =============================================================================
results = {}
for freq_val in FREQ_LIST:
    print(f'Running Isomap for {freq_val} Hz...')
    tRef, bpRef, fr_isomap = load_isomap_for_freq(freq_val)
    results[freq_val] = {'t': tRef, 'bp': bpRef, 'iso': fr_isomap}
    print(f'  done, {fr_isomap.shape[0]} time points')


# =============================================================================
# 3. Plot all frequencies: BP trace (top row) + Isomap projection (bottom row)
# =============================================================================
freqs_sorted = sorted(results.keys())
n = len(freqs_sorted)

# Shared y-axis for the BP traces, so amplitude is directly comparable
# across frequencies.
all_bp = np.concatenate([results[f]['bp'].ravel() for f in freqs_sorted])
bp_pad = 0.05 * (all_bp.max() - all_bp.min())
bp_ylim = (all_bp.min() - bp_pad, all_bp.max() + bp_pad)

# Common relative time axis, 0 to pre+post (= 120 s here), shared by every
# frequency instead of each trial's own absolute session time.
t_rel = np.arange(L) * dt

fig = plt.figure(figsize=(4 * n, 8))
gs = gridspec.GridSpec(2, n, height_ratios=[1, 1.4], hspace=0.4)

sc = None
iso_axes = []
for i, freq_val in enumerate(freqs_sorted):
    r = results[freq_val]

    # --- Top row: BP trace, on the common 0-120 s trial-relative axis ---
    ax_bp = fig.add_subplot(gs[0, i])
    ax_bp.plot(t_rel, r['bp'], color='black', linewidth=1)
    ax_bp.set_title(f'{freq_val} Hz')
    ax_bp.set_xlim(0, pre + post)
    ax_bp.set_ylim(bp_ylim)
    if i == 0:
        ax_bp.set_ylabel('\u0394BP')
    ax_bp.spines['top'].set_visible(False)
    ax_bp.spines['right'].set_visible(False)

    # --- Bottom row: Isomap projection, each at its own native scale, no border ---
    ax_iso = fig.add_subplot(gs[1, i])
    sc = ax_iso.scatter(
        r['iso'][:, 0], r['iso'][:, 1],
        c=r['bp'], cmap='gist_heat', s=20, edgecolors='none'
    )
    ax_iso.set_aspect('equal', adjustable='box')
    ax_iso.set_xticks([])
    ax_iso.set_yticks([])
    ax_iso.set_frame_on(False)
    iso_axes.append(ax_iso)

fig.colorbar(sc, ax=iso_axes, shrink=0.7, label='\u0394BP')
plt.suptitle('Isomap projection across stimulation frequencies')
plt.savefig('freq_comparison_isomap.png', dpi=300, bbox_inches='tight')
plt.show()