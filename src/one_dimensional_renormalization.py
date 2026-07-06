import numpy as np


def get_1d_bare_parameters(
    Omega_A: float,
    g_A: float,
    omega_min: float,
    omega_max: float,
    precision: float = 1e-4,
    apply_corrections: bool = False,
) -> tuple[float, float]:
    """
    Invert the benchmark-model cutoff corrections for Omega and g.

    The renormalization relation is written for g^2. The returned parameters
    are the bare Omega and positive coupling amplitude g used by the simulator.
    """

    if not apply_corrections:
        return Omega_A, g_A

    g_A_squared = float(g_A) ** 2

    Omega_inf = 1.01 * omega_min
    Omega_sup = 0.99 * omega_max
    residual_diff = np.inf
    n_iter = 0

    while abs(residual_diff) > precision and n_iter < 1e4:
        Omega_guess = 0.5 * (Omega_inf + Omega_sup)
        residual_diff = (
            Omega_A
            - Omega_guess
            + g_A_squared
            / np.pi
            * np.log((omega_max - Omega_guess) / (Omega_guess - omega_min))
        )

        if residual_diff > 0:
            Omega_inf = Omega_guess
        else:
            Omega_sup = Omega_guess
        n_iter += 1

    g_squared = g_A_squared / (
        1
        - g_A_squared
        / np.pi
        * (1 / (omega_min - Omega_guess) - 1 / (omega_max - Omega_guess))
    )

    if n_iter < 1e4:
        return Omega_guess, np.sqrt(g_squared)
    return np.nan, np.nan


get_bare_parameters = get_1d_bare_parameters
