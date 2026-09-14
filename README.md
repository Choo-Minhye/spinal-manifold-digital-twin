# Spinal Manifold Digital Twin

Code accompanying "Recurrent latent dynamics in spinal circuits enable a
digital twin of autonomic state transitions" (iScience). Reconstructs a
circular neural manifold from spinal intermediolateral nucleus (IML)
population activity and embeds it as a recurrent reservoir to
generatively decode blood pressure (BP).

## Data

- `save_iso&ref_fr/` - one `.mat` file per trial, each containing
  `frRef` (per-neuron firing rate), `bpRef` (blood pressure trace),
  `tRef` (time vector), and `iso_0`/`iso_1`/`iso_2` (Isomap latent
  coordinates precomputed for that trial).
- `save_frac/` - one `.mat` file per animal with the pairwise
  spike-correlation data used for Figure 2D (`sorted_correlation`,
  `cumulative_fraction`, and their time-shuffled-control counterparts).
- `PYPACKAGE/` - shared plotting and preprocessing utilities used by
  some of the analysis scripts.

## Analysis scripts

| Script | Figure | What it does |
|---|---|---|
| `frcorr_avg.py` | 2D | Pairwise neuron correlation vs. time-shuffled control |
| `revi_figure3.py` | 3E | Neurons most correlated with each manifold dimension |
| `line3d.py` | 5A | Raw (unaligned) manifolds compared across animals |
| `revi_several_bpregression.py` / `revi_bpestimate_compare_2.py` | 4E | Manifold- vs. single-neuron BP decoding across animals |
| `revi_40.py` / `revi_forceregression.py` | 6C | FORCE-learned RNN reservoir decoding of BP from the manifold |
| `forceregression_compareCNN.py` | 6D | CNN baseline for the same decoding comparison |
| `anal_revi.py` | Supplementary | Isomap vs. PCA / LLE / Autoencoder comparison (embedding, persistent homology, geodesic preservation, BP fit) |

The script-to-figure mapping above was reconstructed from each script's
content against the figure legends; please double-check it against your
own records before publishing, and feel free to add a one-line comment
at the top of each script noting the figure it produces.

## Requirements

- Python packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`,
  `scikit-learn`, `torch`, `ripser`, `dcor`
- MATLAB is not required to run these scripts; `.mat` files are read
  directly via `scipy.io.loadmat`.

## Citation

If you use this code, please cite:

> Choo et al,. Recurrent latent dynamics in spinal circuits enable a
> digital twin of autonomic state transitions. *iScience* (2026).
> DOI: not yet
