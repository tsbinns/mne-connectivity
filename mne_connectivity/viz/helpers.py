import mne
import numpy as np
from mne._fiff.pick import _picks_to_idx
from mne.defaults import DEFAULTS
from mne.utils.check import _check_if_nan


def _check_data_is_real(data):
    """Check that data is real-valued."""
    if np.iscomplexobj(data):
        raise ValueError(
            "Plotting for complex-valued connectivity data is not supported. Consider "
            "plotting the absolute values, or the real and imaginary parts separately."
        )


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
        if not is_multivar:
            indices = (np.array(indices[0]), np.array(indices[1]))

    elif indices is None or indices == "all":  # All-to-all connectivity
        # Construct explicit indices
        n_cons = con.shape[0]
        n_chans = con.n_nodes
        if n_cons == 1 and n_chans > 2:  # multivariate connectivity
            indices = (np.arange(n_chans)[None, :], np.arange(n_chans)[None, :])
            is_multivar = True
        else:  # bivariate connectivity
            indices = np.tril_indices(n_chans, -1)
            square_shape = (n_chans, n_chans)
            if data.ndim > 1:
                square_shape += (-1,)
            data = data.reshape(*square_shape)[indices]

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
            indices = (
                np.delete(indices[0][0], bad_idcs),
                np.delete(indices[1][0], bad_idcs),
            )

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
    if any(
        idx not in np.array((*indices[0], *indices[1])) for idx in node_aliases.keys()
    ):
        raise ValueError("All keys in `node_aliases` must be present in `con.indices`.")

    # Get names of nodes (via aliases, directly, or create for multivar connections)
    if not is_multivar:
        unique_nodes = np.unique([*indices[0], *indices[1]]).tolist()
        node_names = ch_names
        for node_ind in unique_nodes:
            if node_ind in node_aliases.keys():
                node_names[node_ind] = node_aliases[node_ind]
    else:
        unique_nodes = list(set([tuple(ind) for ind in (*indices[0], *indices[1])]))
        node_names = [f"node {node_idx}" for node_idx in range(len(unique_nodes))]
        for node_idx, node_ind in enumerate(unique_nodes):
            if node_ind in node_aliases.keys():
                node_names[node_idx] = node_aliases[node_ind]

    # Get indices in terms of node_names entries
    if not is_multivar:  # just use original indices
        node_indices = (indices[0].copy(), indices[1].copy())
    else:  # get indices in terms of unique nodes
        node_indices = ([], [])
        for seed, target in zip(*indices):
            node_indices[0].append(np.where((unique_nodes == seed).all(axis=1))[0][0])
            node_indices[1].append(np.where((unique_nodes == target).all(axis=1))[0][0])
        node_indices = (np.array(node_indices[0]), np.array(node_indices[1]))

    return node_names, node_indices


def _get_con_info(ch_info, node_names, indices, node_indices, is_multivar):
    """Create info object for connectivity data."""
    con_names = []
    for seed, target in zip(*node_indices):
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

    con_info = mne.create_info(ch_names=con_names, sfreq=1.0, ch_types="misc")
    # Can't store connectivity types in ch_types as they are not recognised
    con_info["temp"] = dict()
    con_info["temp"]["con_types"] = np.array(con_types)

    return con_info


def _handle_picks(picks, exclude, ch_info, indices, is_multivar):
    """Handle picks for connectivity data."""
    ch_picks = _picks_to_idx(info=ch_info, picks=picks, none="all", exclude=exclude)
    con_picks = []
    for con_idx, (seed, target) in enumerate(zip(*indices)):
        if not is_multivar:
            seed, target = [seed], [target]
        if np.any([ch in ch_picks for ch in seed]) or np.any(
            [ch in ch_picks for ch in target]
        ):
            con_picks.append(con_idx)

    return con_picks


def _add_comps_as_connections(data, con_info, node_indices, comps_axis):
    """Add multivariate components as additional connections."""
    n_comps = data.shape[comps_axis]
    new_shape = (data.shape[0] * n_comps,)
    if comps_axis + 1 < data.ndim:
        new_shape += (-1,)
    data = np.reshape(data, new_shape)
    node_indices = (
        np.repeat(node_indices[0], n_comps),
        np.repeat(node_indices[1], n_comps),
    )

    new_con_names = []
    for con_name in con_info["ch_names"]:
        new_con_names.extend(
            [f"{con_name} (component {comp})" for comp in range(n_comps)]
        )
    new_con_types = np.repeat(con_info["temp"]["con_types"], n_comps)

    with con_info._unlock():
        con_info["ch_names"] = new_con_names
    con_info["temp"]["con_types"] = new_con_types

    return data, con_info, node_indices
