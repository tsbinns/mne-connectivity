import mne
import numpy as np
from mne.viz._mpl_figure import _line_figure
from mne.viz.evoked import _plot_lines

n_cons, n_freqs = 3, 30

con = np.random.rand(n_cons, n_freqs)
freqs = np.arange(1, n_freqs + 1)
info = mne.create_info(n_cons, sfreq=1, ch_types="misc")
picks = np.arange(n_cons)

fig, axes = _line_figure(info, None, picks=picks)
_plot_lines(
    con,
    info,
    picks=picks,
    fig=fig,
    axes=axes,
    spatial_colors=False,
    unit="",
    units={"eeg-dbs": "Connectivity (A.U.)"},
    scalings=None,
    hline=None,
    gfp=False,
    types=np.array(
        ["eeg-dbs", "eeg-dbs", "eeg-dbs"]
    ),  # np.array(info.get_channel_types()),
    zorder="std",
    xlim=(freqs[0], freqs[-1]),
    ylim=None,
    times=freqs,
    bad_ch_idx=[],
    titles={"eeg-dbs": "Connectivity"},
    ch_types_used=["eeg-dbs"],
    selectable=True,
    psd=False,
    line_alpha=0.75,
    nave=None,
    time_unit="ms",
    sphere=None,
    highlight=None,
)
print("jeff")
