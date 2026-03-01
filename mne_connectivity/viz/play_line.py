from os import path as op

import numpy as np
from line import plot_spectral_connectivity
from matplotlib import pyplot as plt
from mne import make_fixed_length_epochs
from mne.datasets import sample
from mne.io import read_raw_fif

from mne_connectivity import SpectralConnectivity, spectral_connectivity_epochs

data_path = sample.data_path()
raw_fname = op.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif")

# Setup for reading the raw data
raw = read_raw_fif(raw_fname)

raw.pick("eeg").load_data()

epochs = make_fixed_length_epochs(raw, duration=2.0)[:10]
epochs.load_data().pick(np.arange(20))

con = spectral_connectivity_epochs(epochs, method="imcoh", fmax=40)
con = SpectralConnectivity(
    np.abs(con.get_data()), con.freqs, con.n_nodes, con.names, con.indices
)
MULTIVAR = True
if MULTIVAR:
    con = SpectralConnectivity(
        np.concatenate(
            (
                np.concatenate(
                    [
                        con.get_data().mean(0, keepdims=True)[:, None, :],
                        con.get_data().mean(0, keepdims=True)[:, None, :] * 0.5,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        con.get_data().mean(0, keepdims=True)[:, None, :] * -1,
                        con.get_data().mean(0, keepdims=True)[:, None, :] * -0.5,
                    ],
                    axis=1,
                ),
            ),
            axis=0,
        ),
        con.freqs,
        con.n_nodes,
        con.names,
        indices=(
            np.array((np.arange(0, 10), np.arange(10, 20))),
            np.array((np.arange(0, 10), np.arange(10, 20))),
        ),
        components=np.arange(2),
    )

fig = plot_spectral_connectivity(
    con, info=epochs.info, picks=None, node_selection="seeds_and_targets", show=False
)


plt.show(block=True)

print("jeff")
