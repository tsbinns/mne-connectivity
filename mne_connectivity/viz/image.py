import mne
from mne._fiff.pick import pick_info
from mne.utils.check import _check_option, _validate_type
from mne.viz.topo import _imshow_tfr

from mne_connectivity import (
    EpochSpectralConnectivity,
    EpochTemporalConnectivity,
    SpectroTemporalConnectivity,
)

from .helpers import (
    _add_comps_as_connections,
    _check_data_is_real,
    _check_info,
    _get_con_info,
    _get_node_names_and_indices,
    _handle_data_and_indices,
    _handle_picks,
)


def plot_epoch_spectral_connectivity(con):
    """Plot epoch spectral connectivity."""
    _validate_type(con, EpochSpectralConnectivity, "con", "EpochSpectralConnectivity")

    _plot_image_connectivity(
        con,
        x_vals=con.freqs,
        y_vals=con.epochs,
        x_label="Frequency (Hz)",
        y_label="Epochs",
    )


def plot_epoch_temporal_connectivity(con):
    """Plot epoch temporal connectivity."""
    _validate_type(con, EpochTemporalConnectivity, "con", "EpochTemporalConnectivity")

    _plot_image_connectivity(
        con, x_vals=con.times, y_vals=con.epochs, x_label="Time (s)", y_label="Epochs"
    )


def plot_spectro_temporal_connectivity(con):
    """Plot spectro-temporal connectivity."""
    _validate_type(
        con, SpectroTemporalConnectivity, "con", "SpectroTemporalConnectivity"
    )

    _plot_image_connectivity(
        con,
        x_vals=con.times,
        y_vals=con.freqs,
        x_label="Time (s)",
        y_label="Frequency (Hz)",
    )


def _plot_image_connectivity(
    con,
    info,
    picks,
    exclude,
    node_aliases,
    xlim,
    x_vals,
    y_vals,
    x_label,
    y_label,
    show,
):
    """Plot connectivity as image plots.

    Connectivity has dims [connections, x, y], where x and y are epochs, frequencies, or
    times.
    """
    _check_data_is_real(con.get_data())

    _check_option("con.shape", len(con.shape), [3, 4], " length")

    _validate_type(info, (mne.Info, None), "`info`", "mne.Info or None")

    _validate_type(node_aliases, (dict, None), "`node_aliases`", "dict or None")

    if not isinstance(xlim, str):
        _check_option("xlim", len(xlim), [2], " length")
    else:
        _check_option("xlim", xlim, ["tight"], " as a str")

    _validate_type(show, bool, "`show`", "bool")

    ch_names = con.names
    con_method = con.method if con.method is not None else "connectivity"
    ch_info = _check_info(info, ch_names)
    data, indices, is_multivar = _handle_data_and_indices(con, ch_info)

    # Get info about nodes and connections
    node_names, node_indices = _get_node_names_and_indices(
        ch_names, node_aliases, indices, is_multivar
    )
    con_info = _get_con_info(ch_info, node_names, indices, node_indices, is_multivar)

    # Get requested connections
    picks = _handle_picks(picks, exclude, ch_info, indices, is_multivar)
    data = data[picks]
    indices = (indices[0][picks], indices[1][picks])
    node_indices = (node_indices[0][picks], node_indices[1][picks])
    con_info = pick_info(con_info, picks)
    con_info["temp"]["con_types"] = con_info["temp"]["con_types"][picks]

    # Add multivariate components as additional connections
    if data.ndim == 3:
        data, con_info, node_indices = _add_comps_as_connections(
            data, con_info, node_indices, comps_axis=1
        )

    # con_types = con_info["temp"]["con_types"]
    # figs = []

    _imshow_tfr(
        ax=axes[ix],
        tfr=data[[ix]],
        ch_idx=0,
        tmin=x_vals[0],
        tmax=x_vals[-1],
        vmin=vmin,
        vmax=vmax,
        onselect=None,
        ylim=None,
        freq=y_vals,
        x_label=x_label,
        y_label=y_label,
        colorbar=colorbar,
        cmap=cmap,
        yscale=yscale,
        mask=mask,
        mask_style=mask_style,
        mask_cmap=mask_cmap,
        mask_alpha=mask_alpha,
        cnorm=cnorm,
    )
