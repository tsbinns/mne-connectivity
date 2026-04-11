import numpy as np

from mne_connectivity import spectral_connectivity_epochs

n_epo = 5
n_ch = 3
sfreq = 50
n_times = sfreq * 2
method = "dpli"

rng = np.random.RandomState(0)
data = rng.standard_normal((n_epo, n_ch, n_times))

ind_tril = np.tril_indices(n_ch, -1)
ind_triu = np.triu_indices(n_ch, 1)
ind_all = (
    np.repeat(np.arange(n_ch), n_ch),
    np.tile(np.arange(n_ch)[None, :], n_ch).flatten(),
)

methods = [
    "coh",
    "cohy",
    "imcoh",
    "plv",
    "ciplv",
    "ppc",
    "pli",
    "dpli",
    "wpli",
    "wpli2_debiased",
]

for method in methods:
    con_tril = spectral_connectivity_epochs(
        data, method=method, indices=ind_tril, sfreq=sfreq, verbose=False
    )
    con_triu = spectral_connectivity_epochs(
        data, method=method, indices=ind_triu, sfreq=sfreq, verbose=False
    )
    con_all = spectral_connectivity_epochs(
        data, method=method, indices=ind_all, sfreq=sfreq, verbose=False
    )

    print("jeff")
