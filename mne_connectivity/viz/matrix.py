import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mne._fiff.pick import pick_info
from mne.utils.check import _check_option, _validate_type
from mne.utils.misc import _pl

from .helpers import (
    _add_comps_as_connections,
    _check_data_is_real,
    _check_info,
    _get_con_info,
    _get_node_names_and_indices,
    _handle_data_and_indices,
    _handle_picks,
)


# TODO: Add masking support
# TODO: Could there be interactivity to show con names on click?
def plot_connectivity(
    con,
    picks=None,
    exclude="bads",
    info=None,
    node_aliases=None,
    vmin=None,
    vmax=None,
    cnorm=None,
    cmap="viridis",
    colorbar=True,
    node_labels="ticks",
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
    vmin : float | None (default None)
        Minimum value for the colormap. If ``None``, it is determined automatically.
    vmax : float | None (default None)
        Maximum value for the colormap. If ``None``, it is determined automatically.
    cnorm : matplotlib.colors.Normalize | None
        How to normalize the colormap. If ``None``, standard linear normalization is
        performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored. See
        :ref:`Matplotlib docs <matplotlib:colormapnorms>` for more details on colormap
        normalization.
    colorbar : bool (default True)
        Whether to display a colorbar for each figure.
    cmap : str | instance of matplotlib.colors.Colormap (default "viridis")
        The colormap to use for coloring the connectivity values.
    node_labels : ``'names'`` | ``'ticks'`` | None (default ``'ticks'``)
        How to label the nodes in the matrix along the x- and y-axes. If ``'names'``,
        each node's name is shown. Note that for many nodes, this can lead to
        overlapping labels. If ``'ticks'``, the indices of the nodes are shown at evenly
        spaced intervals. Note that for many nodes, not all may have labels. If
        ``None``, no labels are shown.
    show : bool (default True)
        Whether to show the figure(s).

    Notes
    -----
    Plotting for multivariate connectivity is handled by treating each component of the
    multivariate connections as a separate connection. The names of the nodes are
    differentiated by the addition of the component number to the node name, e.g.,
    ``'node 0 (component 0)', 'node 0 (component 1)', ...``.
    """
    from mne_connectivity import Connectivity

    _validate_type(con, Connectivity, "con", "Connectivity")

    _check_data_is_real(con.get_data())

    _check_option("con.shape", len(con.shape), [1, 2], " length")

    _validate_type(info, (mne.Info, None), "`info`", "mne.Info or None")

    _check_option("node_labels", node_labels, ["names", "ticks", None])

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
        if vmin is None:
            vmin = np.nanmin(square_matrix)
        if vmax is None:
            vmax = np.nanmax(square_matrix)
        if cnorm is None:
            cnorm = Normalize(vmin=vmin, vmax=vmax)

        # Create figure and axis
        fig, ax = plt.subplots(
            1, 1, figsize=(6, 6), facecolor="w", layout="constrained"
        )

        ax.imshow(square_matrix, cmap=cmap, norm=cnorm)
        if colorbar:
            fig.colorbar(ax.images[0], ax=ax, shrink=0.6, label="Connectivity (A.U.)")

        ax.set_title(
            f"{con_type} {con_method} ({type_n_cons} connection{_pl(type_n_cons)} from "
            f"{type_n_nodes} node{_pl(type_n_nodes)})"
        )
        ax.set_xlabel("Targets")
        ax.set_ylabel("Seeds")
        if node_labels == "names":
            ax.set_xticks(np.arange(type_n_nodes))
            ax.set_yticks(np.arange(type_n_nodes))
            ax.set_xticklabels(type_node_names, rotation=90)
            ax.set_yticklabels(type_node_names)
        elif node_labels is None:
            ax.set_xticks([])
            ax.set_yticks([])
        # Don't need to do anything for "ticks" option, just use mpl defaults

        figs.append(fig)

    if show:
        plt.show()

    return figs
