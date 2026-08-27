import numpy as np

KNOWN_METHODS = [
    # spec_conn_epochs/time
    "coh",
    "cohy",
    "imcoh",
    "cacoh",
    "mic",
    "mim",
    "plv",
    "ciplv",
    "ppc",
    "pli2_unbiased",
    "dpli",
    "wpli",
    "wpli2_debiased",
    "gc",
    "gc_tr",
    # Mutual info
    "SMI",
    "wSMI",
    # VAR models
    "VAR(1)",
    "VAR(p)",
    "Time-varying VAR(1)",
    "Time-varying VAR(p)",
    # Other
    "envelope correlation",
    "phase-slope-index",
]

CAN_SYMMETRISE = {
    "coh": _symmetrise_coh,
    "cohy": _symmetrise_cohy,
    "imcoh": _symmetrise_imcoh,
    "cacoh": _symmetrise_coh,
    "mic": _symmetrise_cohy,
    "mim": _symmetrise_coh,
    "SMI": _symmetrise_smi,
    "wSMI": _symmetrise_smi,
    "envelope correlation": _symmetrise_envelope_correlation,
    "phase-slope-index": _symmetrise_phase_slope_index,
    "plv": _symmetrise_plv,
    "ciplv": _symmetrise_ciplv,
    "ppc": _symmetrise_ppc,
    "pli2_unbiased": _symmetrise_pli2_unbiased,
    "dpli": _symmetrise_dpli,
    "wpli": _symmetrise_wpli,
    "wpli2_debiased": _symmetrise_wpli2_debiased,
}


def _symmetrise_connectivity(data, indices, n_nodes, method):
    """"""
    # Check if data is already full (no need to symmetrise)
    if indices == "all":
        return data.reshape((n_nodes, n_nodes, *data.shape[1:]))
    if isinstance(indices, tuple):
        full_indices = np.indices((n_nodes, n_nodes))
        sorted_seeds = np.argsort(indices[0])
        if not np.array_equal(
            indices[0][sorted_seeds], full_indices[0].ravel()
        ) or not np.array_equal(indices[1][sorted_seeds], full_indices[1].ravel()):
            raise ValueError(
                "Cannot symmetrise connectivity data when indices are specified as a "
                "tuple that does not represent the full connectivity matrix."
            )
        return data.reshape((n_nodes, n_nodes, *data.shape[1:]))

    # Check if we can symmetrise the connectivity data for the given method
    if method not in CAN_SYMMETRISE:
        raise ValueError(
            f"Cannot symmetrise connectivity data for the method {method}."
        )
    data = _make_square(data, indices, n_nodes)
    return CAN_SYMMETRISE[method](data, indices)


def _make_square(data, indices, n_nodes):
    """Make the connectivity data square [n_nodes, n_nodes(, ...)]."""
    fill = 0.0
    if np.iscomplexobj(data):
        fill = fill + 1j * fill
    square_matrix = np.full(
        (n_nodes, n_nodes, *data.shape[1:]), fill_value=fill, dtype=data.dtype
    )

    if indices == "lower":
        square_matrix[np.tril_indices(n_nodes, k=-1)] = data
    else:  # "upper"
        square_matrix[np.triu_indices(n_nodes, k=1)] = data

    return square_matrix


def _symmetrise_coh(data, indices):
    """Symmetrise coherence data."""
    data = data + data.transpose(1, 0, *range(2, data.ndim))
    data[np.diag_indices(data.shape[0])] = 1.0
    return data


def _symmetrise_cohy(data, indices):
    """Symmetrise coherency data."""
    data = data + data.transpose(1, 0, *range(2, data.ndim))
    data[np.diag_indices(data.shape[0])] = 1.0 + 1.0j
    tril = np.tril_indices(data.shape[0], k=-1)
    triu = np.triu_indices(data.shape[0], k=1)
    if indices == "lower":
        data[triu] = np.conj(data[tril])
    else:  # "upper"
        data[tril] = np.conj(data[triu])
    return data


def _symmetrise_imcoh(data, indices):
    """Symmetrise imaginary part of coherency data."""
    data = data + data.transpose(1, 0, *range(2, data.ndim))
    data[np.diag_indices(data.shape[0])] = 0.0
    tril = np.tril_indices(data.shape[0], k=-1)
    triu = np.triu_indices(data.shape[0], k=1)
    if indices == "lower":
        data[triu] *= -1.0
    else:  # "upper"
        data[tril] *= -1.0
    return data


def _symmetrise_smi(data, indices):
    """Symmetrise symbolic mutual information data."""
    data = data + data.transpose(1, 0, *range(2, data.ndim))
    data[np.diag_indices(data.shape[0])] = 0.0
    return data


def _symmetrise_envelope_correlation(data, indices):
    """Symmetrise envelope correlation data."""
    data = data + data.transpose(1, 0, *range(2, data.ndim))
    data[np.diag_indices(data.shape[0])] = 1.0
    return data


CAN_SYMMETRISE = {
    "coh": _symmetrise_coh,
    "cohy": _symmetrise_cohy,
    "imcoh": _symmetrise_imcoh,
    "plv": _symmetrise_plv,
    "ciplv": _symmetrise_ciplv,
    "ppc": _symmetrise_ppc,
    "pli2_unbiased": _symmetrise_pli2_unbiased,
    "dpli": _symmetrise_dpli,
    "wpli": _symmetrise_wpli,
    "wpli2_debiased": _symmetrise_wpli2_debiased,
    "SMI": _symmetrise_smi,
    "wSMI": _symmetrise_smi,
    "envelope correlation": _symmetrise_envelope_correlation,
}
