import mne
import numpy as np
from matplotlib import pyplot as plt
from mne._fiff.pick import pick_info
from mne.utils.check import _check_option, _validate_type
from mne.utils.misc import _pl

from mne_connectivity import Connectivity

from .helpers import (
    _add_comps_as_connections,
    _check_data_is_real,
    _check_info,
    _get_con_info,
    _get_node_names_and_indices,
    _handle_data_and_indices,
    _handle_picks,
)


def plot_connectivity(
    con,
    picks=None,
    exclude="bads",
    info=None,
    node_aliases=None,
    colorbar=True,
    cmap="viridis",
    show=True,
):
    """Plot connectivity as a matrix.

    Parameters
    ----------
    con : Connectivity
        The connectivity object to plot.
    picks : str | array_like | slice | None (default None)
        Channels to include in the plot. All connections involving these channels will
        be included. Slices and lists of integers will be interpreted as channel
        indices. In lists, channel type strings (e.g., ``['meg', 'eeg']``) will pick
        channels of those types, channel name strings (e.g., ``['MEG0111', 'MEG2623']``
        will pick the given channels. Can also be the string values ``'all'`` to pick
        all channels, or ``'data'`` to pick data channels. None will pick any good
        channels. Note that channels in ``info['bads']`` will be included if their names
        or indices are explicitly provided.
    exclude : list of str | ``'bads'`` (default ``'bads'``)
        Channel names to exclude from plotting. All connections involving these channels
        will be excluded. If ``'bads'``, channels in ``info['bads']`` are excluded. Pass
        an empty list to include all channels (including bad channels, if any).
    info : mne.Info | None (default None)
        The :class:`mne.Info` object with information about the sensors and methods of
        measurement. Used to split the figures by channel types and identify bad
        channels.
    node_aliases : dict | None
        Mapping of node indices to node names. Keys should be seed or target indices
        found in ``con.indices``, that is, ints for bivariate connectivity, and arrays
        of ints for multivariate connectivity. If ``None`` and plotting results for
        bivariate connectivity, node names will be taken from ``con.names``. If ``None``
        and plotting results for multivariate connectivity, node names will be generated
        as ``'node {idx}'``, where ``idx`` is the order of the node in the unique set of
        indices, as determined by ``np.unique([*con.indices[0], *con.indices[1]])``.
    colorbar : bool (default True)
        Whether to display a colorbar.
    cmap : str | instance of matplotlib.colors.Colormap (default "viridis")
        The colormap to use for coloring the connectivity values.
    show : bool (default True)
        Whether to show the figure.
    """
    _validate_type(con, Connectivity, "con", "Connectivity")

    _check_data_is_real(con.get_data())

    _check_option("con.shape", len(con.shape), [1, 2], " length")

    _validate_type(info, (mne.Info, None), "`info`", "mne.Info or None")

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
    if is_multivar:
        data, con_info, node_indices = _add_comps_as_connections(
            data, con_info, node_indices, comps_axis=1
        )

    con_types = con_info["temp"]["con_types"]
    figs = []
    for con_type in np.unique(con_types):
        # Prepare connectivity info for plotting
        type_mask = con_types == con_type
        type_node_indices = tuple(idcs[type_mask] for idcs in node_indices)
        type_n_cons = len(type_node_indices[0])
        type_node_indices_unique = np.unique(type_node_indices)
        type_node_names = [node_names[idx] for idx in type_node_indices_unique]
        type_n_nodes = type_node_indices_unique.size

        # Make data square for plotting
        square_matrix = np.full((type_n_nodes, type_n_nodes), fill_value=np.nan)
        for idx, (seed_idx, target_idx) in enumerate(zip(*type_node_indices)):
            square_matrix[seed_idx, target_idx] = data[idx]

        # Create figure and axis
        fig, ax = plt.subplots(
            1, 1, figsize=(6, 6), facecolor="w", layout="constrained"
        )

        ax.imshow(square_matrix, cmap=cmap)
        if colorbar:
            fig.colorbar(ax.images[0], ax=ax, shrink=0.6, label="Connectivity (A.U.)")

        ax.set_title(
            f"{con_type} {con_method} ({type_n_cons} connection{_pl(type_n_cons)} from "
            f"{type_n_nodes} node{_pl(type_n_nodes)})"
        )
        ax.set_xlabel("Targets")
        ax.set_ylabel("Seeds")
        ax.set_xticks(np.arange(type_n_nodes))
        ax.set_yticks(np.arange(type_n_nodes))
        ax.set_xticklabels(type_node_names, rotation=90)
        ax.set_yticklabels(type_node_names)

        figs.append(fig)

    if show:
        plt.show()

    return figs
