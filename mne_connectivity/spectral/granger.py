from functools import partial

import numpy as np
from mne.parallel import parallel_func
from mne.time_frequency import (
    EpochsSpectrum,
    EpochsSpectrumArray,
    EpochsTFR,
    EpochsTFRArray,
)
from mne.utils import ProgressBar, _check_option, _validate_type, verbose

from ..utils import fill_doc
from .epochs import _compute_freq_mask
from .epochs_multivariate import _check_rank_input


@verbose
@fill_doc
def spectral_granger_causality_epochs(
    data,
    indices,
    method="trgc",
    *,
    return_constituents=False,
    fmin=None,
    fmax=None,
    fdecim=1,
    tmin=None,
    tmax=None,
    tdecim=1,
    n_lags=40,
    rank=None,
    n_jobs=1,
    verbose=None,
):
    r"""Compute frequency- and time-frequency-domain state-space Granger causality.

    Parameters
    ----------
    data : ~mne.time_frequency.EpochsSpectrum | ~mne.time_frequency.EpochsTFR
        The data to compute Granger causality on. Must be complex-valued coefficients.
    ...
    method : 'gc' | 'net_gc' | 'trgc'
        The Granger causality measure to compute:
        - ``'gc'`` : Granger causality. Represents the directed information flow from
          seeds to targets.
        - ``'net_gc'`` : net Granger causality. Represents the net interaction between
          seeds and targets, without correcting for spurious connectivity estimates.
        - ``'trgc'`` : time-reversed Granger causality. Like ``'net_gc'``, but with
          correction for spurious connectivity estimates (default).

        See notes for details.
    return_constituents : bool
        Whether or not to additionally return the constituent components of the Granger
        causality measure requested by ``method`` (defalt False). See notes for details.
    ...
    n_lags : int
        Number of lags to use for the vector autoregressive model. Higher value
        increase computational cost, but reduce the degree of spectral smoothing in the
        results. Default 40.

    Returns
    -------
    con : instance of SpectralConnectivity or SpectroTemporalConnectivity
        The computed Granger causality measure. A :class:`SpectralConnectivity` object
        is returned if ``data`` is an instance of
        :class:`~mne.time_frequency.EpochsSpectrum`, and a
        :class:`SpectroTemporalConnectivity` object is returned if ``data`` is an
        instance of :class:`~mne.time_frequency.EpochsTFR`.
    constituents : dict
        A dictionary containing the constituent components of the Granger causality
        measure as a mapping of method names to connectivity containers, matching the
        type of ``con``. Only returned if ``return_constituents`` is True.

    Notes
    -----
    Spectral state-space Granger causality (``method='gc'``) is calculated as

        :math:`F_{A\to B} = ln\Large{(\frac{\lvert\boldsymbol{S}_{BB}\rvert}{\lvert
        \boldsymbol{S}_{BB}-\boldsymbol{H}_{BA}\boldsymbol{\Sigma}_{AA
        \lvert B}\boldsymbol{H}_{BA}^*\rvert}})`

    where: :math:`A` and :math:`B` represent the seeds and targets, respectively;
    :math:`\boldsymbol{H}` is the spectral transfer function;
    :math:`\boldsymbol{\Sigma}` is the residuals matrix of the autoregressive model;
    and :math:`\boldsymbol{S}` is :math:`\boldsymbol{\Sigma}` transformed by
    :math:`\boldsymbol{H}` :footcite:`BarnettSeth2015`.

    Net Granger causality (``method='net_gc'``) is the difference of Granger scores
    between directions, :math:`F^{net}_{A\to B} = F_{A\to B} - F_{B\to A}`.

    Time-reversed Granger causality (``method='trgc'``) is a form of net Granger
    causality corrected for spurious estimates of connectivity arising from correlated
    noise between signals, e.g., due to source mixing :footcite:`WinklerEtAl2016`. It is
    defined as :math:`\tilde{D}_{A\to B} = F^{net}_{A\to B} -
    \tilde{F}^{net}_{\tilde{A}\to \tilde{B}}`, where
    :math:`\tilde{F}^{net}_{\tilde{A}\to \tilde{B}}` is the net Granger causality of the
    time-reversed signals. In practice, time-reversal of the signals is achieved by
    transposing the autocovariance sequence produced from the autoregressive model
    :footcite:`HaufeEtAl2012`. **Because of its robustness to spurious connectivity
    estimates, time-reversed Granger causality is the recommended approach.**

    ..versionadded:: 1.0
    """
    _check_option("method", method, ["gc", "net_gc", "trgc"])

    _validate_type(data, [EpochsSpectrum, EpochsTFR], "data")
    if not np.iscomplexobj(data.get_data()):
        raise TypeError(
            "`data` must contain complex-valued Fourier coefficients, such as that "
            "returned from `Epochs.compute_psd/tfr(..., output='complex')`."
        )
    if isinstance(data, EpochsSpectrum):
        if isinstance(data, EpochsSpectrumArray):  # infer mode from dimensions
            mode = "multitaper" if "taper" in data._dims else "welch"
        else:  # read mode from object
            mode = data.method
        if mode == "welch" and "segment" not in data._dims:
            raise ValueError(
                "`data` for the Welch method must contain unaggregated segments, such "
                "as that returned from "
                "`Epochs.compute_psd(method='welch', output='complex', average=None)`."
            )
        is_tfr_con = False
    else:
        if isinstance(data, EpochsTFRArray):  # infer mode from dimensions
            mode = "multitaper" if "taper" in data._dims else "morlet"
        else:  # read mode from object
            mode = data.method
        is_tfr_con = True
    # Extract weights from the EpochsSpectrum/TFR object
    if not hasattr(data, "weights") or (data.weights is None and mode == "multitaper"):
        # TODO Version: Only mention re-computing saved objects when mne<1.10 dropped
        raise AttributeError(
            "Weights are required for multitaper coefficients stored in "
            "`EpochsSpectrum` (requires MNE >= 1.8) and `EpochsTFR` (requires MNE >= "
            "1.10) objects. Objects saved from older versions of MNE will need to be "
            "recomputed."
        )
    if not np.allclose(np.diff(freqs), np.diff(freqs)[0]):
        raise ValueError("`data` must contain evenly spaced frequencies bins.")

    if fmin is None:
        fmin = -np.inf
    if fmax is None:
        fmax = np.inf
    _validate_type(fmin, "numeric", "fmin")
    _validate_type(fmax, "numeric", "fmax")
    _validate_type(fdecim, "int", "fdecim")
    if fdecim < 1:
        raise ValueError("`fdecim` must be > 0.")

    if tmin is None:
        tmin = -np.inf
    if tmax is None:
        tmax = np.inf
    _validate_type(tmin, "numeric", "tmin")
    _validate_type(tmax, "numeric", "tmax")
    _validate_type(tdecim, "int", "tdecim")
    if tdecim < 1:
        raise ValueError("`tdecim` must be > 0.")

    _validate_type(n_lags, "int", "n_lags")
    if n_lags < 1:
        raise ValueError("`n_lags` must be > 0.")

    rank = _check_rank_input(rank, data, indices)

    data_arr = data.get_data()
    freqs, times, weights = (data.freqs, data.get("times"), data.weights)

    freq_mask = _compute_freq_mask(freqs, fmin, fmax, fdecim)
    if not np.any(freq_mask):
        raise ValueError(
            f"No frequency bins exist between {fmin} and {fmax} Hz. Increase the "
            "`fmin`-`fmax` range or increase the frequency resolution of the data."
        )
    data_arr = data_arr.take(freq_mask, axis=data._dims.index("freq"))
    freqs = freqs[freq_mask]

    csd_padding, freqs_padded = _pad_missing_freqs(freqs, data.sfreq)

    time_mask = None
    if is_tfr_con:
        time_mask = _compute_freq_mask(times, tmin, tmax, tdecim)
        if not np.any(time_mask):
            raise ValueError(
                f"No time bins exist between {tmin} and {tmax} s. Increase the "
                "`tmin`-`tmax` range or increase the time resolution of the data."
            )
        data_arr = data_arr.take(time_mask, axis=data._dims.index("time"))
        times = times[time_mask]

    parallel, parallel_gc_single_epoch = parallel_func(
        _gc_single_epoch, n_jobs, verbose=verbose
    )


@verbose
@fill_doc
def spectral_granger_causality_time():
    """"""


def _pad_missing_freqs(freqs, sfreq):
    """Pad missing CSD frequency bins with zeros to fill the 0-Nyquist range."""
    fstep = np.diff(freqs).mean()
    nyquist = sfreq / 2

    fmin = freqs[0]
    fmax = freqs[-1]

    n_missing_start = int(np.round(fmin / fstep))
    n_missing_end = int(np.round((nyquist - fmax) / fstep))

    csd_padding = partial(
        np.pad,
        pad_width=((0, 0), (n_missing_start, n_missing_end), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    freqs_padded = np.pad(
        freqs, (n_missing_start, n_missing_end), mode="constant", constant_values=np.nan
    )

    return csd_padding, freqs_padded


def _gc_single_epoch(data, mode, weights):
    """Compute Granger causality for a single epoch."""
    # Compute CSD
    if mode == "welch":
        data = np.moveaxis(data, 2, 1)  # move segments to taper position
        weights = np.ones(data.ndim)  # create dummy weights
    if mode in ["welch", "multitaper"]:
        weights_sq = (weights * weights.conj()).real
        data *= weights * np.sqrt(2 / weights_sq.sum(axis=1, keepdims=True))
    csd = (data * data.conj()).sum(axis=1)  # sum over tapers/segments

    # Compute Granger causality
    con = _spectral_granger_causality()


def _spectral_granger_causality(csd, indices, ranks, freqs, n_lags, sfreq, n_jobs):
    """"""
    csd_padded, freqs_padded = _pad_missing_csd_freqs(csd, freqs, sfreq)

    time_idcs = np.arange(csd_padded.shape[0])
    freq_idcs = np.arange(csd_padded.shape[1])

    con_i = 0
    for seed_idcs, target_idcs, seed_rank, target_rank in zip(
        indices[0], indices[1], ranks[0], ranks[1]
    ):
        _log_connection_number(con_i)

        seed_idcs = seed_idcs.compressed()
        target_idcs = target_idcs.compressed()
        con_idcs = [*seed_idcs, *target_idcs]

        C = csd_padded[np.ix_(time_idcs, freq_idcs, con_idcs, con_idcs)]

        C_bar = _project_csd_to_subspace(C, seed_idcs, seed_rank, target_rank)
        con_seeds = np.arange(seed_rank)
        con_targets = np.arange(target_rank) + seed_rank

        autocov = _compute_autocov(C_bar, n_lags)
        if name == "GC time-reversed":
            autocov = autocov.transpose(0, 1, 3, 2)

        A_f, V = _autocov_to_full_var(autocov)
        A, K = _full_var_to_iss(A_f)

        con_scores[con_i] = _iss_to_ugc(
            A, A_f, K, V, con_seeds, con_targets, freqs_padded, n_steps, n_jobs
        )

        con_i += 1


def _project_csd_to_subspace(csd, seed_idcs, seed_rank, target_rank):
    """Project CSD to rank subspace using SVD applied to covariance matrix."""
    # sum over times and epochs to get cov. from CSD
    cov = csd.sum(axis=(0, 1))

    n_seeds = len(seed_idcs)
    n_targets = csd.shape[3] - n_seeds

    cov_aa = cov[:n_seeds, :n_seeds]
    cov_bb = cov[n_seeds:, n_seeds:]

    if seed_rank != n_seeds:
        U_aa = np.linalg.svd(np.real(cov_aa), full_matrices=False)[0]
        U_bar_aa = U_aa[:, :seed_rank]
    else:
        U_bar_aa = np.identity(n_seeds)

    if target_rank != n_targets:
        U_bb = np.linalg.svd(np.real(cov_bb), full_matrices=False)[0]
        U_bar_bb = U_bb[:, :target_rank]
    else:
        U_bar_bb = np.identity(n_targets)

    C_aa = csd[..., :n_seeds, :n_seeds]
    C_ab = csd[..., :n_seeds, n_seeds:]
    C_bb = csd[..., n_seeds:, n_seeds:]
    C_ba = csd[..., n_seeds:, :n_seeds]

    C_bar_aa = U_bar_aa.transpose(1, 0) @ (C_aa @ U_bar_aa)
    C_bar_ab = U_bar_aa.transpose(1, 0) @ (C_ab @ U_bar_bb)
    C_bar_bb = U_bar_bb.transpose(1, 0) @ (C_bb @ U_bar_bb)
    C_bar_ba = U_bar_bb.transpose(1, 0) @ (C_ba @ U_bar_aa)
    C_bar = np.append(
        np.append(C_bar_aa, C_bar_ab, axis=3),
        np.append(C_bar_ba, C_bar_bb, axis=3),
        axis=2,
    )

    return C_bar


def _compute_autocov(csd, n_lags, freq_res):
    """Compute autocovariance from CSD."""
    n_times = csd.shape[0]
    n_signals = csd.shape[2]

    csd = np.concatenate(
        [np.flip(np.conj(csd[:, 1:]), axis=1), csd[:, :-1]], axis=1
    )  # circular shifting

    csd_3d = np.reshape(csd, csd.shape[:2] + (-1,), order="F")
    ifft_csd = np.fft.ifft(csd_3d, n=freq_res, axis=1)
    ifft_csd = np.reshape(ifft_csd, csd.shape, order="F")

    lags_ifft_csd = np.reshape(
        ifft_csd[:, : n_lags + 1],
        (n_times, n_lags + 1, n_signals**2),
        order="F",
    )

    signs = 1 - 2 * (np.arange(n_lags + 1) % 2)

    return np.real(
        np.reshape(
            lags_ifft_csd * signs[np.newaxis, :, np.newaxis],
            (n_times, n_lags + 1, n_signals, n_signals),
            order="F",
        )
    )


def _autocov_to_full_var(autocov):
    """Convert autocovariance to full VAR model using Whittle's LWR recursion."""
    if np.any(np.linalg.det(autocov) == 0):
        raise RuntimeError(
            "The autocovariance matrix is singular. Check if your data is "
            "rank-deficient and specify an appropriate rank argument ≤ the rank of the "
            "seeds and targets"
        )

    A_f, V = _whittle_lwr_recursion(autocov)

    if not np.isfinite(A_f).all():
        raise RuntimeError(
            "At least one VAR model coefficient is infinite or NaN. Check whether "
            "your data contains infinite or NaN values."
        )

    try:
        np.linalg.cholesky(V)
    except np.linalg.LinAlgError as np_error:
        raise RuntimeError(
            "The covariance matrix of the residuals is not positive-definite. "
            "Check the singular values of your data and specify an appropriate "
            "rank argument ≤ the rank of the seeds and targets"
        ) from np_error

    A_f = np.reshape(A_f, autocov.shape[:2] + (-1,), order="F")

    return A_f, V


def _whittle_lwr_recursion(G):
    """Solve Yule-Walker equations for full VAR parameters with LWR recursion.

    See: Whittle P., 1963. Biometrika, DOI: 10.1093/biomet/50.1-2.129
    """
    # Initialise recursion
    n = G.shape[2]  # number of signals
    q = G.shape[1] - 1  # number of lags
    t = G.shape[0]  # number of times
    qn = n * q

    cov = G[:, 0, :, :]  # covariance
    G_f = np.reshape(
        G[:, 1:, :, :].transpose(0, 3, 1, 2), (t, qn, n), order="F"
    )  # forward autocov
    G_b = np.reshape(
        np.flip(G[:, 1:, :, :], 1).transpose(0, 3, 2, 1), (t, n, qn), order="F"
    ).transpose(0, 2, 1)  # backward autocov

    A_f = np.zeros((t, n, qn))  # forward coefficients
    A_b = np.zeros((t, n, qn))  # backward coefficients

    k = 1  # model order
    r = q - k
    k_f = np.arange(k * n)  # forward indices
    k_b = np.arange(r * n, qn)  # backward indices

    try:
        A_f[:, :, k_f] = np.linalg.solve(
            cov, G_b[:, k_b, :].transpose(0, 2, 1)
        ).transpose(0, 2, 1)
        A_b[:, :, k_b] = np.linalg.solve(
            cov, G_f[:, k_f, :].transpose(0, 2, 1)
        ).transpose(0, 2, 1)

        # Perform recursion
        for k in np.arange(2, q + 1):
            var_A = G_b[:, (r - 1) * n : r * n, :] - (A_f[:, :, k_f] @ G_b[:, k_b, :])
            var_B = cov - (A_b[:, :, k_b] @ G_b[:, k_b, :])
            AA_f = np.linalg.solve(var_B, var_A.transpose(0, 2, 1)).transpose(0, 2, 1)

            var_A = G_f[:, (k - 1) * n : k * n, :] - (A_b[:, :, k_b] @ G_f[:, k_f, :])
            var_B = cov - (A_f[:, :, k_f] @ G_f[:, k_f, :])
            AA_b = np.linalg.solve(var_B, var_A.transpose(0, 2, 1)).transpose(0, 2, 1)

            A_f_previous = A_f[:, :, k_f]
            A_b_previous = A_b[:, :, k_b]

            r = q - k
            k_f = np.arange(k * n)
            k_b = np.arange(r * n, qn)

            A_f[:, :, k_f] = np.dstack((A_f_previous - (AA_f @ A_b_previous), AA_f))
            A_b[:, :, k_b] = np.dstack((AA_b, A_b_previous - (AA_b @ A_f_previous)))
    except np.linalg.LinAlgError as np_error:
        raise RuntimeError(
            "The autocovariance matrix is singular. Check if your data is"
            "rank-deficient and specify an appropriate rank argument ≤ the rank of the "
            "seeds and targets"
        ) from np_error

    V = cov - (A_f @ G_f)
    A_f = np.reshape(A_f, (t, n, n, q), order="F")

    return A_f, V


def _full_var_to_iss(A_f):
    """Compute innovations-form parameters for a state-space model from full VAR model.

    Parameters computed from a full VAR model using Aoki's method. For a
    non-moving-average full VAR model, the state-space parameter C (observation
    matrix) is identical to AF of the VAR model.

    See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
    10.1103/PhysRevE.91.040101.
    """
    t = A_f.shape[0]
    m = A_f.shape[1]  # number of signals
    p = A_f.shape[2] // m  # number of autoregressive lags

    I_p = np.dstack(t * [np.eye(m * p)]).transpose(2, 0, 1)
    A = np.hstack((A_f, I_p[:, : (m * p - m), :]))  # state transition matrix
    K = np.hstack(
        (
            np.dstack(t * [np.eye(m)]).transpose(2, 0, 1),
            np.zeros((t, (m * (p - 1)), m)),
        )
    )  # Kalman gain matrix

    return A, K


def _iss_to_ugc(A, C, K, V, seeds, targets, freqs, n_steps, n_jobs):
    """Compute unconditional GC from innovations-form state-space paramseters.

    See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
    10.1103/PhysRevE.91.040101.
    """
    time_inds = np.arange(A.shape[0])
    freq_inds = np.arange(len(freqs))

    # points on a unit circle in the complex plane, one for each frequency
    z = np.exp(-1j * np.pi * np.linspace(0, 1, len(freqs)))[~np.isnan(freqs)]

    H = _iss_to_tf(A, C, K, z, n_steps, n_jobs)  # spectral transfer function
    V_22_1 = np.linalg.cholesky(_partial_covar(V, seeds, targets))
    HV = H @ np.linalg.cholesky(V)
    S = HV @ HV.conj().transpose(0, 1, 3, 2)  # Eq. 6
    S_11 = S[np.ix_(freq_inds, time_inds, targets, targets)]
    HV_12 = H[np.ix_(freq_inds, time_inds, targets, seeds)] @ V_22_1
    HVH = HV_12 @ HV_12.conj().transpose(0, 1, 3, 2)

    # Eq. 11
    return np.real(np.log(np.linalg.det(S_11)) - np.log(np.linalg.det(S_11 - HVH)))


def _iss_to_tf(A, C, K, z, n_steps, n_jobs):
    """Compute transfer function for innovations-form state-space params.

    In the frequency domain, the back-shift operator, z, is a vector of points on a
    unit circle in the complex plane. z = e^-iw, where -pi < w <= pi.

    A note on efficiency: solving over the 4D time-freq. tensor is slower than
    looping over times and freqs when n_times and n_freqs is high, and when n_times and
    n_freqs is low, looping over times and freqs very fast anyway (plus tensor solving
    doesn't allow for parallelisation).

    See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
    10.1103/PhysRevE.91.040101.
    """
    t = A.shape[0]
    h = len(z)
    n = C.shape[1]
    m = A.shape[1]
    I_n = np.eye(n)
    I_m = np.eye(m)
    H = np.zeros((h, t, n, n), dtype=np.complex128)

    parallel, parallel_compute_transfer_func, _ = parallel_func(
        _compute_transfer_func, n_jobs, verbose=False
    )
    H = np.zeros((h, t, n, n), dtype=np.complex128)
    for block_i in ProgressBar(range(n_steps), mesg="frequency blocks"):
        freqs = _get_block_indices(block_i, h, n_jobs)
        H[freqs] = parallel(
            parallel_compute_transfer_func(A, C, K, z[k], I_n, I_m) for k in freqs
        )

    return H


def _compute_transfer_func(A, C, K, z_k, I_n, I_m):
    """Compute transfer function for innovations-form state-space params.

    See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
    10.1103/PhysRevE.91.040101, Eq. 4.
    """
    from scipy import linalg  # XXX: is this necessary???

    H = np.zeros((A.shape[0], C.shape[1], C.shape[1]), dtype=np.complex128)
    for t in range(A.shape[0]):
        H[t] = I_n + (C[t] @ linalg.lu_solve(linalg.lu_factor(z_k * I_m - A[t]), K[t]))

    return H


def _get_block_indices(block_i, limit, n_jobs):
    """Get indices for a computation block capped by a limit."""
    indices = np.arange(block_i * n_jobs, (block_i + 1) * n_jobs)

    return indices[np.nonzero(indices < limit)]


def _partial_covar(V, seeds, targets):
    """Compute partial covariance of a matrix.

    Given a covariance matrix V, the partial covariance matrix of V between indices
    i and j, given k (V_ij|k), is equivalent to V_ij - V_ik * V_kk^-1 * V_kj. In
    this case, i and j are seeds, and k are targets.

    See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
    10.1103/PhysRevE.91.040101.
    """
    times = np.arange(V.shape[0])
    W = np.linalg.solve(
        np.linalg.cholesky(V[np.ix_(times, targets, targets)]),
        V[np.ix_(times, targets, seeds)],
    )
    W = W.transpose(0, 2, 1) @ W

    return V[np.ix_(times, seeds, seeds)] - W
