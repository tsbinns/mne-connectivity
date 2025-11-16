from functools import partial

import mne
import numpy as np
from matplotlib import pyplot as plt
from mne._fiff.pick import _picks_to_idx, pick_info
from mne.defaults import DEFAULTS
from mne.utils.check import (
    _check_if_nan,
    _check_option,
    _validate_type,
)
from mne.utils.misc import _pl
from mne.viz.circle import _plot_connectivity_circle
from mne.viz.evoked import (
    _butterfly_on_button_press,
    _butterfly_onpick,
)


def plot_spectral_connectivity(
    con,
    picks=None,
    info=None,
    node_aliases=None,
    interactive=True,
    show=True,
):
    """Plot spectral connectivity as line plots with circle plot overviews."""
    from mne_connectivity import SpectralConnectivity

    _validate_type(con, SpectralConnectivity, "con", "SpectralConnectivity")

    return _plot_spectral_or_temporal_connectivity(
        con,
        picks,
        info,
        node_aliases,
        interactive,
        show,
        xvar=con.freqs,
        xlabel="Frequency (Hz)",
    )


def plot_temporal_connectivity(
    con,
    picks=None,
    info=None,
    node_aliases=None,
    interactive=True,
    show=True,
):
    """Plot temporal connectivity as line plots with circle plot overviews."""
    from mne_connectivity import TemporalConnectivity

    _validate_type(con, TemporalConnectivity, "con", "TemporalConnectivity")

    return _plot_spectral_or_temporal_connectivity(
        con,
        picks,
        info,
        node_aliases,
        interactive,
        show,
        xvar=con.times,
        xlabel="Times (s)",
    )


def _plot_spectral_or_temporal_connectivity(
    con,
    picks,
    info,
    node_aliases,
    interactive,
    show,
    xvar,
    xlabel,
):
    _check_option("con.shape", len(con.shape), [2, 3], " length")

    _validate_type(info, (mne.Info, None), "info", "mne.Info or None")

    _validate_type(node_aliases, (dict, None), "`node_aliases`", "dict or None")

    ch_names = con.names
    ch_info = _check_info(info, ch_names)
    data, indices, is_multivar = _handle_data_and_indices(con, ch_info)

    # Get info about nodes and connections
    node_names, node_indices = _get_node_names_and_indices(
        ch_names, node_aliases, indices, is_multivar
    )
    con_info = _get_con_info(ch_info, node_names, indices, is_multivar)

    # Get requested connections
    picks = _picks_to_idx(info=con_info, picks=picks, none="all")
    data = data[picks]
    indices = (indices[0][picks], indices[1][picks])
    node_indices = (node_indices[0][picks], node_indices[1][picks])
    con_info = pick_info(con_info, picks)
    con_info["temp"]["con_types"] = con_info["temp"]["con_types"][picks]

    # Add multivariate components as additional connections
    if data.ndim == 3:
        data, con_info = _add_comps_as_connections(data, con_info, comps_axis=1)

    con_types = con_info["temp"]["con_types"]
    figs = []
    for con_type in np.unique(con_types):
        # Prepare connectivity info for plotting
        type_mask = con_types == con_type
        type_data = data[type_mask]
        type_con_names = np.array(con_info["ch_names"])[type_mask]
        type_node_indices = tuple(idcs[type_mask] for idcs in node_indices)

        # Prepare circle plot values
        circle_names, circle_indices, is_all_to_all = _get_circle_names_and_indices(
            node_names, type_node_indices
        )
        n_circle_nodes = len(circle_names)
        # If plot is interactive and connectivity data is all-to-all, colouring works
        # best if connections are duplicated such that all nodes are seeds and targets
        duplicate_cons = is_all_to_all and interactive
        if duplicate_cons:
            circle_indices = (
                np.concatenate([circle_indices[0], circle_indices[1]]),
                np.concatenate([circle_indices[1], circle_indices[0]]),
            )
        circle_con, circle_con_order = _get_circle_con(
            circle_indices, n_circle_nodes, duplicate_cons
        )

        # Create figure and axes
        fig = plt.figure(figsize=(15, 5), facecolor="w", layout="constrained")
        line_ax = fig.add_subplot(1, 3, (1, 2))
        circle_ax = fig.add_subplot(1, 3, 3, polar=True)

        # Plot connectivity as circle
        fig, circle_ax = _plot_connectivity_circle(
            con=circle_con,
            node_names=circle_names,
            indices=circle_indices,
            node_colors="k",
            facecolor="w",
            textcolor="k",
            node_edgecolor="w",
            colormap="turbo",
            colorbar=False,
            ax=circle_ax,
            interactive=False,  # use our modified callback
            show=show,
        )

        # Plot connectivity as lines
        fig, line_ax = _plot_lines_connectivity(
            data=type_data,
            con_colors=_get_con_colors(circle_ax, circle_con_order),
            con_names=type_con_names,
            n_nodes=n_circle_nodes,
            duplicate_cons=duplicate_cons,
            fig=fig,
            ax=line_ax,
            xlim=(xvar[0], xvar[-1]),
            ylim=None,
            xvar=xvar,
            xlabel=xlabel,
            title=con_type,
            interactive=interactive,
            line_alpha=0.75,
            highlight=None,
        )

        # Add connectivity selection callback
        if interactive:
            callback = partial(
                _plot_connectivity_circle_onpick,
                fig=fig,
                circle_ax=circle_ax,
                line_ax=line_ax,
                indices=circle_indices,
                node_angles=np.linspace(0, 2 * np.pi, n_circle_nodes, endpoint=False),
                circle_con_order=circle_con_order,
                duplicate_cons=duplicate_cons,
            )
            fig.canvas.mpl_connect("button_press_event", callback)

        # Hide duplicate connections initially
        if duplicate_cons:
            _hide_duplicate_cons(
                fig, circle_ax, line_ax, len(type_data), circle_con_order
            )

        figs.append(fig)

    return figs


def _handle_data_and_indices(con, ch_info):
    """Extract data and indices from connectivity object."""
    indices = con.indices
    is_multivar = False

    data = con.get_data("raveled")
    if isinstance(indices, tuple):  # Explicit indices provided
        if not np.all(
            [np.issubdtype(type(ind), int) for ind in indices[0]]
        ) and not np.all([np.issubdtype(type(ind), int) for ind in indices[1]]):
            is_multivar = True

    elif indices is None or indices == "all":  # All-to-all connectivity
        # Construct explicit indices
        n_cons = con.shape[0]
        n_chans = con.n_nodes
        if n_cons == 1 and n_chans > 2:  # multivariate connectivity
            indices = (np.arange(n_chans)[None, :], np.arange(n_chans)[None, :])
            is_multivar = True
        else:  # bivariate connectivity
            indices = np.tril_indices(n_chans, -1)
            data = data.reshape(n_chans, n_chans, -1)[indices]

        # Drop entries for bad channels from all-to-all data/indices
        bad_idcs = []
        if ch_info is not None:
            bad_idcs = [con.names.index(bad) for bad in ch_info["bads"]]
        if len(bad_idcs) > 0 and not is_multivar:
            good_con_mask = np.ones(data.shape[0], dtype=bool)
            for con_idx, (seed, target) in enumerate(zip(*indices)):
                if seed in bad_idcs or target in bad_idcs:
                    good_con_mask[con_idx] = False
            data = data[good_con_mask]
            indices = (indices[0][good_con_mask], indices[1][good_con_mask])
        elif len(bad_idcs) > 0 and is_multivar:
            indices = (np.delete(indices[0], bad_idcs), np.delete(indices[1], bad_idcs))

    else:
        assert indices == "symmetric"
        raise NotImplementedError("check how to handle symm indices")

    _check_if_nan(data)

    return data, indices, is_multivar


def _check_info(info, ch_names):
    """Check (or create) info object and ensure all channels are present."""
    if info is None:
        info = mne.create_info(ch_names=ch_names, sfreq=1.0, ch_types="misc")

    # Make sure all channel names from con object found in info
    missing_channels = [name for name in ch_names if name not in info["ch_names"]]
    if len(missing_channels) != 0:
        raise ValueError(
            "Not all channel names from `con.names` found in `info`. Missing channels: "
            f"{missing_channels}"
        )

    return info


def _get_node_names_and_indices(ch_names, node_aliases, indices, is_multivar):
    """Get/create names of seeds/targets in connections and their indices."""
    if node_aliases is None:
        node_aliases = dict()
    if any(idx not in (*indices[0], *indices[1]) for idx in node_aliases.keys()):
        raise ValueError("All keys in `node_aliases` must be present in `con.indices`.")

    # Get names of nodes (via aliases, directly, or create for multivar connections)
    unique_nodes = np.unique([*indices[0], *indices[1]]).tolist()
    node_names = [None] * len(ch_names)
    for node_idx in unique_nodes:
        if node_idx in node_aliases.keys():
            node_names[node_idx] = node_aliases[node_idx]
        elif not is_multivar:
            node_names[node_idx] = ch_names[node_idx]
        else:
            node_names[node_idx] = ",".join([ch_names[ch_idx] for ch_idx in node_idx])

    # Get indices in terms of node_names entries
    if not is_multivar:  # just use original indices
        node_indices = (indices[0].copy(), indices[1].copy())
    else:  # get indices in terms of unique nodes
        node_indices = ([], [])
        for seed, target in zip(*indices):
            node_indices[0].append(unique_nodes.index(seed))
            node_indices[1].append(unique_nodes.index(target))

    return node_names, node_indices


def _get_con_info(ch_info, node_names, indices, is_multivar):
    """Create info object for connectivity data."""
    con_names = []
    for seed, target in zip(*indices):
        con_names.append(f"{node_names[seed]} → {node_names[target]}")

    ch_types = ch_info.get_channel_types()
    con_types = []
    for seed, target in zip(*indices):
        if not is_multivar:
            seed_type = DEFAULTS["titles"][ch_types[seed]]
            target_type = DEFAULTS["titles"][ch_types[target]]
            con_types.append(f"{seed_type} → {target_type}")
        else:
            seed_types = np.unique(
                [DEFAULTS["titles"][ch_types[ch_idx]] for ch_idx in seed]
            )
            target_types = np.unique(
                [DEFAULTS["titles"][ch_types[ch_idx]] for ch_idx in target]
            )
            con_types.append(f"{', '.join(seed_types)} → {', '.join(target_types)}")

    con_info = mne.create_info(
        ch_names=con_names,
        sfreq=1.0,
        ch_types="eeg",  # use a valid data type instead of misc
    )
    # Can't store connectivity types in ch_types as they are not recognised
    con_info["temp"] = dict()
    con_info["temp"]["con_types"] = np.array(con_types)

    return con_info


def _add_comps_as_connections(data, con_info, comps_axis):
    """Add multivariate components as additional connections."""
    n_comps = data.shape[comps_axis]
    data = np.reshape(data, data.shape[0] * n_comps, -1)

    new_con_names = []
    for con_name in con_info["ch_names"]:
        new_con_names.extend(
            [f"{con_name} (component {comp + 1})" for comp in range(n_comps)]
        )
    new_con_types = np.repeat(con_info["temp"]["con_types"], n_comps)

    with con_info._unlock():
        con_info["ch_names"] = new_con_names
    con_info["temp"]["con_types"] = new_con_types

    return data, con_info


def _get_circle_names_and_indices(node_names, node_indices):
    """Get names of nodes and indices of connections between them for circle plot."""
    unique_nodes = np.unique(np.r_[node_indices[0], node_indices[1]])
    circle_names = [node_names[idx] for idx in unique_nodes]

    circle_indices = [np.searchsorted(unique_nodes, ind) for ind in node_indices]

    is_all_to_all = np.all(
        np.all(ind == all_to_all_ind)
        for ind, all_to_all_ind in zip(
            circle_indices, np.tril_indices(len(circle_names), -1)
        )
    )  # check if all-to-all connectivity

    return circle_names, circle_indices, is_all_to_all


def _get_circle_con(circle_indices, n_nodes, is_all_to_all):
    """Get connectivity values for circle plot (determines colour)."""
    if is_all_to_all:  # values span colourbar per node
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        circle_con = np.zeros(len(circle_indices[0]))
        for con_idx, (seed, target) in enumerate(zip(*circle_indices)):
            node_diff = node_angles[seed] - node_angles[target]
            if node_diff > 0:
                node_diff -= 2 * np.pi
            circle_con[con_idx] = np.abs(node_diff)
    else:  # values span colourbar over all connections
        circle_con = circle_indices[0] + circle_indices[1]

    # mne.viz.circle._plot_connectivity_circle default behaviour is to sort connections
    # by strength (valid as of MNE v1.11)
    circle_con_order = np.argsort(circle_con)  # to map cons in circle plot to indices

    return circle_con, circle_con_order


def _get_con_colors(circle_ax, circle_con_order):
    """Get colors of connections from circle plot."""
    con_colors = [None] * len(circle_con_order)
    for patch_idx, con_idx in enumerate(circle_con_order):
        patch = circle_ax.patches[patch_idx]
        con_colors[con_idx] = patch.get_edgecolor()

    return con_colors


def _plot_connectivity_circle_onpick(
    event,
    fig,
    circle_ax,
    line_ax,
    indices,
    node_angles,
    duplicate_cons,
    circle_con_order,
    ylim=(9, 10),
):
    """Isolate connections for a single node and reflect this in the line plot.

    On left click, shows only connections related to the clicked node.
    On right click, resets all connections.

    `y_lim` radius is default in circle plot (valid in MNE v1.11).
    """
    if event.inaxes != circle_ax:
        return

    patches = circle_ax.patches
    lines = line_ax.lines
    if event.button == 1:  # left click
        if not ylim[0] <= event.ydata <= ylim[1]:
            return  # ignore click if not near nodes

        # all angles in range [0, 2*pi]
        node_angles = node_angles % (np.pi * 2)
        node = np.argmin(np.abs(event.xdata - node_angles))

        for circle_idx, line_idx in enumerate(circle_con_order):
            seed, target = indices[0][line_idx], indices[1][line_idx]
            visible = node == seed if duplicate_cons else node in (seed, target)
            patches[circle_idx].set_visible(visible)
            lines[line_idx].set_visible(visible)
            lines[line_idx].set_picker(0 if not visible else True)
        fig.canvas.draw()

    elif event.button == 3:  # right click
        n_cons = len(indices[0]) if not duplicate_cons else len(indices[0]) // 2
        for circle_idx, line_idx in enumerate(circle_con_order):
            if line_idx < n_cons:  # make original connections visible
                patches[circle_idx].set_visible(True)
                lines[line_idx].set_visible(True)
                lines[line_idx].set_picker(True)
            else:  # hide duplicated connections
                patches[circle_idx].set_visible(False)
                lines[line_idx].set_visible(False)
                lines[line_idx].set_picker(False)
        for text in line_ax.texts:
            text.set_alpha(0)  # hide any connection labels
        fig.canvas.draw()


def _hide_duplicate_cons(fig, circle_ax, line_ax, n_cons, circle_con_order):
    """Hide duplicated connections in circle and line plots."""
    for circle_idx, line_idx in enumerate(circle_con_order):
        if line_idx >= n_cons:
            circle_ax.patches[circle_idx].set_visible(False)
            line_ax.lines[line_idx].set_visible(False)
            line_ax.lines[line_idx].set_picker(False)
    fig.canvas.draw()


def _plot_lines_connectivity(
    data,
    con_colors,
    con_names,
    n_nodes,
    duplicate_cons,
    fig,
    ax,
    xlim,
    ylim,
    xvar,
    xlabel,
    title,
    interactive,
    line_alpha,
    highlight,
):
    """Plot data as butterfly plot."""
    texts = list()
    n_cons = data.shape[0]
    idxs = np.arange(n_cons)
    if duplicate_cons:
        idxs = np.concatenate([idxs, idxs + n_cons])
    lines = list()

    if interactive:
        # Parameters for butterfly interactive plots
        if duplicate_cons:
            con_names = np.concatenate([con_names, con_names])
        params = dict(
            axes=[ax],
            texts=texts,
            lines=[lines],
            ch_names=con_names,
            idxs=[idxs],
            need_draw=False,
            path_effects=None,
        )
        fig.canvas.mpl_connect("pick_event", partial(_butterfly_onpick, params=params))
        fig.canvas.mpl_connect(
            "button_press_event", partial(_butterfly_on_button_press, params=params)
        )

    # Map cons with least activity in front of the more active ones
    z_ord = data.std(axis=1).argsort()

    # plot connections
    for con_idx, z in enumerate(z_ord):
        lines.append(
            ax.plot(
                xvar,
                data[con_idx],
                picker=True,
                zorder=z + 1,
                color=con_colors[con_idx],
                alpha=line_alpha,
                linewidth=0.5,
            )[0]
        )
        lines[-1].set_pickradius(3.0)
    if duplicate_cons:
        for con_idx, z in enumerate(z_ord):
            lines.append(
                ax.plot(
                    xvar,
                    data[con_idx],
                    picker=True,
                    zorder=z + 1,
                    color=con_colors[con_idx + n_cons],
                    alpha=line_alpha,
                    linewidth=0.5,
                )[0]
            )
            lines[-1].set_pickradius(3.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Connectivity (A.U.)")
    texts.append(
        ax.text(
            0,
            0,
            "",
            zorder=3,
            verticalalignment="baseline",
            horizontalalignment="left",
            fontweight="bold",
            alpha=0,
            clip_on=True,
        )
    )

    if xlim is not None:
        if xlim == "tight":
            xlim = (xvar[0], xvar[-1])
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set(
        title=(
            f"{title} ({n_cons} connection{_pl(n_cons)} "
            f"between {n_nodes} node{_pl(n_nodes)})"
        )
    )

    # Plot highlights
    if highlight is not None:
        this_ylim = ax.get_ylim() if (ylim is None) else ylim
        for this_highlight in highlight:
            ax.fill_betweenx(
                this_ylim,
                this_highlight[0],
                this_highlight[1],
                facecolor="orange",
                alpha=0.15,
                zorder=99,
            )
        # Put back the y limits as fill_betweenx messes them up
        ax.set_ylim(this_ylim)

    return fig, ax
