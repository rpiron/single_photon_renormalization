"""Prepare the guide without geometry files or a geometry database."""

import numpy as np

from src.modes import gaussian_packet, momentum_grid, photon_parameters
from src.xp_config import ExperimentConfig, normalize_cutoffs


def make_waveguide_config(*, L, Omega_0, gamma_0, photon, time, cutoffs,
                          store_state=True, integrator_func=None):
    
    """Prepare N_A=1, x_A=0 and g_{0,1n}=i sqrt(gamma_0/(2L))."""

    if not np.isfinite(L) or L <= 0 or not np.isfinite(gamma_0) or gamma_0 < 0:
        raise ValueError("Require L > 0 and gamma_0 >= 0, both finite.")
    
    bounds = normalize_cutoffs(cutoffs)
    params = photon_parameters(photon, 1)

    if params["p_init"][0] <= 0:
        raise ValueError("The waveguide benchmark uses a right-moving packet: p_init > 0.")
    
    mode_indices, p_n, omega_n = momentum_grid([L], bounds)
    forward, backward = p_n[:, 0] > 0, p_n[:, 0] < 0
    c_init = gaussian_packet(p_n, params, forward)
    
    return ExperimentConfig(
        omega_n=omega_n, Omega_0=np.array([Omega_0]),
        g_0=np.full((len(omega_n), 1), 1j * np.sqrt(gamma_0 / (2 * L))),
        c_init=c_init, b_init=np.zeros(1, complex), time=time,
        p_n=p_n, mode_indices=mode_indices, forward_mask=forward,
        backward_mask=backward, store_state=store_state,
        integrator_func=integrator_func,
        metadata={"model": "waveguide", "lengths": np.array([L]), "cutoffs": bounds},
    )
