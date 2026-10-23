"""Calibrate the 2D model by fitting its spectral kernel F_jk(omega)."""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import j0
from tqdm.auto import tqdm

from .database import load_target_profile


def generate_target_profile(omega_values, atom_positions, Omega_j, d_j,
                            reference_cutoffs, id_geom=None, epsabs=1e-8,
                            epsrel=1e-8, limit=200, cache_decimals=12, progress=True):
    """Tabulate F_jk = d_j G_jk d_k with uniform frequency weights."""
    atom_positions = np.asarray(atom_positions, dtype=float).reshape(-1, 2)
    n_atoms = len(atom_positions)
    Omega_j = np.broadcast_to(np.asarray(Omega_j, dtype=float), (n_atoms,))
    d_j = np.broadcast_to(np.asarray(d_j, dtype=float), (n_atoms,))
    omega_values = np.asarray(omega_values, dtype=float)
    j_indices, k_indices = np.indices((n_atoms, n_atoms))
    blocks = []

    for omega in tqdm(omega_values, disable=not progress):
        kernel = spectral_kernel(
            omega, atom_positions, Omega_j, d_j, reference_cutoffs,
            epsabs=epsabs, epsrel=epsrel, limit=limit, cache_decimals=cache_decimals,
        )
        blocks.append(pd.DataFrame({
            "omega": omega,
            "j": j_indices.ravel(),
            "k": k_indices.ravel(),
            "W": (1 / len(omega_values)) / kernel.size,
            "F_real": kernel.real.ravel(),
            "F_imag": kernel.imag.ravel(),
        }))

    profile = pd.concat(blocks, ignore_index=True)
    profile.attrs["metadata"] = {
        "id_geom": id_geom,
        "nb_omega_values": len(omega_values),
        "Omega_A": Omega_j.copy(),
        "d_A": d_j.copy(),
        "reference_omega_min": reference_cutoffs["Omega_min"],
        "reference_omega_max": reference_cutoffs["Omega_max"],
    }
    return profile


def get_2d_bare_parameters(target_profile, atom_positions, cutoffs):
    """Fit Omega_0 and d_0 to the target kernel on the retained window."""
    profile = load_target_profile(target_profile, cutoffs)
    atom_positions = np.asarray(atom_positions, dtype=float).reshape(-1, 2)
    n_atoms = len(atom_positions)
    Omega_A = profile.attrs["metadata"]["Omega_A"]
    d_A = profile.attrs["metadata"]["d_A"]
    target_F = profile["F_real"].to_numpy() + 1j * profile["F_imag"].to_numpy()
    weights = np.sqrt(profile["W"].to_numpy())

    def residual(parameters):
        Omega_0, d_0 = parameters[:n_atoms], parameters[n_atoms:]
        trial_F = np.empty(len(profile), dtype=complex)
        for omega, rows in profile.groupby("omega", sort=False):
            kernel = spectral_kernel(omega, atom_positions, Omega_0, d_0, cutoffs)
            trial_F[rows.index] = kernel[rows["j"].to_numpy(dtype=int), rows["k"].to_numpy(dtype=int)]
        error = trial_F - target_F
        return np.concatenate((weights * error.real, weights * error.imag))

    result = least_squares(
        residual,
        np.concatenate((np.clip(Omega_A, cutoffs["Omega_min"], cutoffs["Omega_max"]), d_A)),
        bounds=(
            np.r_[np.full(n_atoms, cutoffs["Omega_min"]), np.full(n_atoms, -np.inf)],
            np.r_[np.full(n_atoms, cutoffs["Omega_max"]), np.full(n_atoms, np.inf)],
        ),
        max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )
    return result.x[:n_atoms], result.x[n_atoms:]


def spectral_kernel(omega, atom_positions, Omega_j, d_j, cutoffs,
                    epsabs=1e-8, epsrel=1e-8, limit=200, cache_decimals=12):
    """Compute the radial integrals, self-energy Sigma, then F = d G d.

    G = [omega I - Omega - Sigma]^{-1}. Repeated interatomic distances share
    one quadrature; cache_decimals controls their grouping as in the notebooks.
    """
    atom_positions = np.asarray(atom_positions, dtype=float).reshape(-1, 2)
    n_atoms = len(atom_positions)
    Omega_j = np.broadcast_to(np.asarray(Omega_j, dtype=float), (n_atoms,))
    d_j = np.broadcast_to(np.asarray(d_j, dtype=float), (n_atoms,))
    low, high = cutoffs["Omega_min"], cutoffs["Omega_max"]
    radii = np.linalg.norm(atom_positions[:, None, :] - atom_positions[None, :, :], axis=2)
    radius_keys = np.round(radii.ravel(), cache_decimals)

    # Principal-value radial integral for each distinct distance.
    pv_by_radius = {}
    for key in np.unique(radius_keys):
        radius = float(radii.ravel()[np.flatnonzero(radius_keys == key)[0]])
        if low < omega < high:
            value, _ = quad(
                lambda w: w * j0(w * radius), low, high,
                weight="cauchy", wvar=omega, epsabs=epsabs, epsrel=epsrel, limit=limit,
            )
            pv_by_radius[key] = -value
        else:
            value, _ = quad(
                lambda w: w * j0(w * radius) / (omega - w), low, high,
                epsabs=epsabs, epsrel=epsrel, limit=limit,
            )
            pv_by_radius[key] = value

    pv = np.asarray([pv_by_radius[key] for key in radius_keys]).reshape(radii.shape)
    d_outer = d_j[:, None] * d_j[None, :]
    sigma = d_outer * pv / (2 * np.pi) - 1j * omega * d_outer * j0(omega * radii) / 2
    green_inverse = omega * np.eye(n_atoms, dtype=complex) - np.diag(Omega_j) - sigma
    green = np.linalg.solve(green_inverse, np.eye(n_atoms, dtype=complex))
    return d_j[:, None] * green * d_j[None, :]
