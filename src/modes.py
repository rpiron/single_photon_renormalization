"""Box momentum grids and normalized incident wave packets."""

import numpy as np

from .xp_config import normalize_cutoffs


def momentum_grid(lengths, cutoffs):
    """Return integer indices, momenta and energies, sorted by energy.

    Both energy boundaries are included. A floating-point tolerance prevents
    rounding from discarding modes mathematically on those boundaries.
    """
    lengths = np.atleast_1d(np.asarray(lengths, dtype=float))
    if lengths.ndim != 1 or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("Box lengths must be positive and finite.")
    bounds = normalize_cutoffs(cutoffs)
    boundary_tolerance = 8 * np.finfo(float).eps * max(1., bounds["Omega_max"])
    delta_p = 2 * np.pi / lengths
    n_max = np.floor((bounds["Omega_max"] + boundary_tolerance) / delta_p).astype(int)
    grids = np.meshgrid(*(np.arange(-n, n + 1) for n in n_max), indexing="ij")
    indices = np.stack([grid.ravel() for grid in grids], axis=1)
    momenta = indices * delta_p[None, :]
    omega = np.linalg.norm(momenta, axis=1)
    mask = (omega >= bounds["Omega_min"] - boundary_tolerance) & (omega <= bounds["Omega_max"] + boundary_tolerance)
    order = np.argsort(omega[mask], kind="mergesort")
    if not np.any(mask):
        raise ValueError("The frequency window retains no modes in this box.")
    return indices[mask][order], momenta[mask][order], omega[mask][order]


def photon_parameters(photon, dimension):
    values = {}
    for name in ("p_init", "x_init", "sigma_p"):
        value = np.atleast_1d(np.asarray(photon[name], dtype=float))
        if value.shape != (dimension,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain {dimension} finite components.")
        values[name] = value
    if np.any(values["sigma_p"] <= 0):
        raise ValueError("sigma_p must be positive.")
    return values


def gaussian_packet(p_n, photon, mask=None):
    params = photon_parameters(photon, p_n.shape[1])
    reduced = (p_n - params["p_init"]) / params["sigma_p"]
    packet = np.exp(-np.sum(reduced**2, axis=1) / 4) * np.exp(-1j * p_n @ params["x_init"])
    if mask is not None:
        packet *= mask
    norm = np.linalg.norm(packet)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("The incident packet has no numerical support in the retained modes.")
    return packet / norm
