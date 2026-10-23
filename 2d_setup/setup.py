"""Prepare a 2D experiment from its geometry and atomic parameters."""

import numpy as np

from src.modes import gaussian_packet, momentum_grid, photon_parameters
from src.xp_config import ExperimentConfig
from .database import find_bare_param_path, load_bare_parameters_csv


def make_2d_config(param_photon, param_atom, param_time_evol, cutoffs,
                   correction=False, *, store_state=True):
    """Assign atomic parameters, build the modes and couplings, and configure RK4.

    param_atom contains geometry, Omega_A and d_A. With correction=True, load
    the calibrated Omega_0 and d_0 for this geometry and frequency window.
    """
    geometry = param_atom["geometry"]
    lengths = np.asarray(geometry["lengths"], dtype=float)
    positions = np.asarray(geometry["atom_positions"], dtype=float)
    n_atoms = len(positions)

    Omega_0 = np.broadcast_to(np.asarray(param_atom["Omega_A"], dtype=float), (n_atoms,)).copy()
    d_0 = np.broadcast_to(np.asarray(param_atom["d_A"], dtype=float), (n_atoms,)).copy()
    if correction:
        path = find_bare_param_path(
            Omega_0, d_0, cutoffs, id_geom=geometry.get("id_geom"),
            id_target_profile=param_atom.get("id_target_profile"),
        )
        Omega_0, d_0 = load_bare_parameters_csv(path)

    mode_indices, p_n, omega_n = momentum_grid(lengths, cutoffs)
    photon = photon_parameters(param_photon, 2)
    momentum_norm = np.linalg.norm(photon["p_init"])
    if momentum_norm == 0:
        raise ValueError("p_init must define a nonzero incident direction.")
    projection = p_n @ (photon["p_init"] / momentum_norm)
    g_0 = 1j * d_0[None, :] * np.exp(-1j * p_n @ positions.T) / np.sqrt(np.prod(lengths))

    return ExperimentConfig(
        omega_n=omega_n,
        Omega_0=Omega_0,
        g_0=g_0,
        c_init=gaussian_packet(p_n, photon),
        b_init=np.zeros(n_atoms, complex),
        time=param_time_evol,
        p_n=p_n,
        mode_indices=mode_indices,
        forward_mask=projection > 1e-12,
        backward_mask=projection < -1e-12,
        store_state=store_state,
        metadata={"model": "fiber2d", "lengths": lengths.copy(),
                  "x_j": positions.copy(), "d_0": d_0.copy()},
    )
