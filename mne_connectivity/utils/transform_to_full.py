import numpy as np


def _symmetrise_connectivity(data, indices, n_nodes, method):
    """Lorem ipsum."""
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


def _make_symmetric(data, indices, diag, transpose_extra=None):
    """Make the connectivity data symmetric."""
    # Set the diagonal
    data[np.diag_indices(data.shape[0])] = diag
    # Always transpose
    data = data + data.transpose(1, 0, *range(2, data.ndim))
    # Perform something on top of the transpose if needed
    if transpose_extra is not None:
        if indices == "lower":
            triu = np.triu_indices(data.shape[0], k=1)
            data[triu] = transpose_extra(data[triu])
        else:  # "upper"
            tril = np.tril_indices(data.shape[0], k=-1)
            data[tril] = transpose_extra(data[tril])
    return data


def _symmetrise_coh(data, indices):
    """Symmetrise coherence data."""
    return _make_symmetric(data, indices, diag=1.0)


def _symmetrise_cohy(data, indices):
    """Symmetrise coherency data."""
    return _make_symmetric(data, indices, diag=1.0 + 0.0j, transpose_extra=np.conj)


def _symmetrise_imcoh(data, indices):
    """Symmetrise imaginary part of coherency data."""
    return _make_symmetric(data, indices, diag=0.0, transpose_extra=lambda x: -x)


def _symmetrise_plv(data, indices):
    """Symmetrise phase-locking value data."""
    return _make_symmetric(data, indices, diag=1.0)


def _symmetrise_ciplv(data, indices):
    """Symmetrise corrected imaginary part of phase-locking value data."""
    return _make_symmetric(data, indices, diag=0.0)


def _symmetrise_ppc(data, indices):
    """Symmetrise pairwise phase consistency data."""
    return _make_symmetric(data, indices, diag=1.0)


def _symmetrise_pli(data, indices):
    """Symmetrise phase lag index data."""
    return _make_symmetric(data, indices, diag=0.0)


def _symmetrise_dpli(data, indices):
    """Symmetrise directed phase lag index data."""
    return _make_symmetric(data, indices, diag=0.5, transpose_extra=lambda x: 1.0 - x)


def _symmetrise_wpli(data, indices):
    """Symmetrise weighted phase lag index data."""
    return _make_symmetric(data, indices, diag=0.0)


def _symmetrise_smi(data, indices):
    """Symmetrise symbolic mutual information data."""
    return _make_symmetric(data, indices, diag=0.0)


def _symmetrise_envelope_correlation(data, indices):
    """Symmetrise envelope correlation data."""
    return _make_symmetric(data, indices, diag=1.0)


CAN_SYMMETRISE = {
    "coh": _symmetrise_coh,
    "cohy": _symmetrise_cohy,
    "imcoh": _symmetrise_imcoh,
    "plv": _symmetrise_plv,
    "ciplv": _symmetrise_ciplv,
    "ppc": _symmetrise_ppc,
    "pli": _symmetrise_pli,
    "pli2_unbiased": _symmetrise_pli,
    "dpli": _symmetrise_dpli,
    "wpli": _symmetrise_wpli,
    "wpli2_debiased": _symmetrise_wpli,
    "SMI": _symmetrise_smi,
    "wSMI": _symmetrise_smi,
    "envelope correlation": _symmetrise_envelope_correlation,
}
