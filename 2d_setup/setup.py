"""Convert 2D geometry and numerical parameters to the common Hamiltonian."""

import numpy as np

from src.modes import gaussian_packet, momentum_grid, photon_parameters
from src.xp_config import ExperimentConfig


def make_array_config(*, lengths, x_j, Omega_0, d_0, photon, time, cutoffs,
                      store_state=True, integrator_func=None):
    """Pure preparation from already calibrated parameters; no disk access."""
    lengths = np.asarray(lengths, dtype=float)
    positions = np.asarray(x_j, dtype=float)
    if lengths.shape != (2,) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Require lengths.shape=(2,) and x_j.shape=(N_A,2).")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Atomic positions must be finite.")
    n_atoms = len(positions)
    Omega_0 = np.broadcast_to(np.asarray(Omega_0, dtype=float), (n_atoms,)).copy()
    d_0 = np.broadcast_to(np.asarray(d_0, dtype=complex), (n_atoms,)).copy()
    mode_indices, p_n, omega_n = momentum_grid(lengths, cutoffs)
    params = photon_parameters(photon, 2)
    momentum_norm = np.linalg.norm(params["p_init"])
    if momentum_norm == 0:
        raise ValueError("p_init must define a nonzero incident direction.")
    projection = p_n @ (params["p_init"] / momentum_norm)
    g_0 = 1j * d_0[None, :] * np.exp(-1j * p_n @ positions.T) / np.sqrt(np.prod(lengths))
    return ExperimentConfig(
        omega_n=omega_n, Omega_0=Omega_0, g_0=g_0,
        c_init=gaussian_packet(p_n, params), b_init=np.zeros(n_atoms, complex),
        time=time, p_n=p_n, mode_indices=mode_indices,
        forward_mask=projection > 1e-12, backward_mask=projection < -1e-12,
        store_state=store_state, integrator_func=integrator_func,
        metadata={"model": "fiber2d", "lengths": lengths.copy(),
                  "x_j": positions.copy(), "d_0": d_0.copy()},
    )


def make_fiber_config(param_photon, param_atom, param_time_evol, cutoffs,
                      correction=False, *, store_state=True, database_dir=None):
    """Convenience preparation using an existing geometry and optional CSV fit.

    Calibration itself remains an explicit, separate operation. database_dir
    points to database_fiber; omitted, use the repository's bundled database.
    """
    from pathlib import Path
    from .database import BARE_PARAM_DIR, find_bare_param_path, load_bare_parameters_csv

    geometry = param_atom["geometry"]
    n_atoms = len(geometry["atom_positions"])
    Omega_A = np.broadcast_to(np.asarray(param_atom["Omega_A"], dtype=float), (n_atoms,))
    d_A = np.broadcast_to(np.asarray(param_atom["d_A"], dtype=float), (n_atoms,))
    Omega_0, d_0 = Omega_A, d_A
    if correction:
        directory = BARE_PARAM_DIR if database_dir is None else Path(database_dir) / "bare_param"
        path = find_bare_param_path(Omega_A, d_A, cutoffs,
                                   id_geom=geometry.get("id_geom"),
                                   id_target_profile=param_atom.get("id_target_profile"),
                                   directory=directory)
        Omega_0, d_0 = load_bare_parameters_csv(path)
    return make_array_config(lengths=geometry["lengths"], x_j=geometry["atom_positions"],
                             Omega_0=Omega_0, d_0=d_0, photon=param_photon,
                             time=param_time_evol, cutoffs=cutoffs, store_state=store_state)
