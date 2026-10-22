import numpy as np

from src.xp_config import normalize_cutoffs

pi = np.pi


def get_bare_parameters(Omega_A, gamma_A, cutoffs, *, correction=True, precision=1e-4):
    """Return (Omega_0, gamma_0) using the paper's target/numerical notation."""
    bounds = normalize_cutoffs(cutoffs)
    if not correction:
        return Omega_A, gamma_A
    return _invert_parameters(Omega_A, gamma_A, bounds["Omega_min"], bounds["Omega_max"], precision)


def _invert_parameters(Omega_A, gamma_A, ir, uv, precision=1e-4):
    """
    Computes the bare parameters by inverting the one-photon renormalization
    relations.

    Parameters:
    Omega_A : physical transition frequency of the TLS
    gamma_A : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : convergence tolerance for the bisection search

    Returns:
    Omega_0 : bare frequency to parameterize the Hamiltonian
    gamma_0 : bare decay rate to parameterize the Hamiltonian
    """

    w0_inf = 1.01 * ir
    w0_sup = 0.99 * uv
    residual_diff = np.inf
    N_iter = 0

    while np.abs(residual_diff) > precision and N_iter < 1e4:
        w0_guess = 0.5 * (w0_inf + w0_sup)
        residual_diff = Omega_A - w0_guess + gamma_A / (2 * pi) * np.log(
            (uv - w0_guess) / (w0_guess - ir)
        )

        if residual_diff > 0:
            w0_inf = w0_guess
        else:
            w0_sup = w0_guess
        N_iter += 1

    gamma_0 = gamma_A / (
        1 - gamma_A / (2 * pi) * (1 / (ir - w0_guess) - 1 / (uv - w0_guess))
    )

    if N_iter <= 1e4:
        return w0_guess, gamma_0
    return np.nan, np.nan

