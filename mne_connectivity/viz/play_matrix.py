from os import path as op

import numpy as np
from matplotlib import pyplot as plt
from mne import make_fixed_length_epochs
from mne.datasets import sample
from mne.io import read_raw_fif

from mne_connectivity import (
    Connectivity,
    SpectralConnectivity,
    spectral_connectivity_epochs,
)
from mne_connectivity.viz.matrix import plot_connectivity

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

con = Connectivity(
    con.get_data()[:, 0], con.n_nodes, con.names, con.indices, "abs(imcoh)"
)

MULTIVAR = True
if MULTIVAR:
    con = Connectivity(
        np.concatenate(
            (
                np.concatenate(
                    [
                        con.get_data().mean(0, keepdims=True)[:, None],
                        con.get_data().mean(0, keepdims=True)[:, None] * 0.5,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        con.get_data().mean(0, keepdims=True)[:, None] * -1,
                        con.get_data().mean(0, keepdims=True)[:, None] * -0.5,
                    ],
                    axis=1,
                ),
            ),
            axis=0,
        ),
        con.n_nodes,
        con.names,
        indices=(
            np.array((np.arange(0, 10), np.arange(10, 20))),
            np.array((np.arange(0, 10), np.arange(10, 20))),
        ),
        components=np.arange(2),
    )

fig = plot_connectivity(con, info=epochs.info, picks=None, show=False)


plt.show(block=True)

print("jeff")
