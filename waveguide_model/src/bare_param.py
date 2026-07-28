import numpy as np

pi = np.pi


def get_bare_param_n1(omega_A, gamma_A, ir, uv, precision=1e-4):
    """
    Computes the bare parameters by inverting the one-photon renormalization
    relations.

    Parameters:
    omega_A : physical transition frequency of the TLS
    gamma_A : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : convergence tolerance for the bisection search

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma_0 : bare decay rate to parameterize the Hamiltonian
    """

    w0_inf = 1.01 * ir
    w0_sup = 0.99 * uv
    residual_diff = np.inf
    N_iter = 0

    while np.abs(residual_diff) > precision and N_iter < 1e4:
        w0_guess = 0.5 * (w0_inf + w0_sup)
        residual_diff = omega_A - w0_guess + gamma_A / (2 * pi) * np.log(
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


def get_bare_param_n(omega_A, gamma_A, ir, uv, n=1):
    """
    Computes the bare parameters for the benchmark model.

    n=-1 is the baseline with no correction to the bare parameters.
    n=1 applies the one-photon cutoff correction.
    """

    if n == -1:
        omega_0 = omega_A
        gamma_0 = gamma_A
    else:
        omega_0, gamma_0 = get_bare_param_n1(omega_A, gamma_A, ir, uv)

    return omega_0, gamma_0
