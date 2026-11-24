# Spinal Manifold Digital Twin

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.isci.2026.117971-blue)](https://doi.org/10.1016/j.isci.2026.117971)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.23409705.svg)](https://doi.org/10.5281/zenodo.23409705)

Code accompanying "[Recurrent latent dynamics in spinal circuits enable a
digital twin of autonomic state transitions](https://doi.org/10.1016/j.isci.2026.117971)"
(*iScience* 29, 117971, 2026). Reconstructs a circular neural manifold from
spinal intermediolateral nucleus (IML) population activity and embeds it as
a recurrent reservoir to generatively decode blood pressure (BP).

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

Figure numbers refer to the published article. Supplementary analyses
correspond to the Supplementary Information of the same article.

## Requirements

- Python packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`,
  `scikit-learn`, `torch`, `ripser`, `dcor`
- MATLAB is not required to run these scripts; `.mat` files are read
  directly via `scipy.io.loadmat`.

## Citation

If you use this code, please cite:

> Choo, M., Cho, M., Lee, J., Oh, C., Hong, S., Heo, C., Shon, Y.M., and
> Park, S. (2026). Recurrent latent dynamics in spinal circuits enable a
> digital twin of autonomic state transitions. *iScience* 29, 117971.
> https://doi.org/10.1016/j.isci.2026.117971

M. Choo and M. Cho contributed equally to this work.

```bibtex
@article{choo2026recurrent,
  title   = {Recurrent latent dynamics in spinal circuits enable a digital twin of autonomic state transitions},
  author  = {Choo, Minhye and Cho, Mingeun and Lee, Jiho and Oh, Changhwa and Hong, Sungjun and Heo, Chaewon and Shon, Young-Min and Park, Sungmin},
  journal = {iScience},
  volume  = {29},
  number  = {11},
  pages   = {117971},
  year    = {2026},
  doi     = {10.1016/j.isci.2026.117971},
  pmid    = {42990874},
  pmcid   = {PMC13734804}
}
```

The exact code version used in the article is archived on Zenodo:
[10.5281/zenodo.23409705](https://doi.org/10.5281/zenodo.23409705)
