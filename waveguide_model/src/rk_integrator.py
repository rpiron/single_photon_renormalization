import numpy as np
from tqdm import tqdm


def increment_state(c, b, omega_tab, param_cavity, param_time_evol, t):
    """
    Computes the infinitesimal increment (c_new, b_new) = -i * dt * V_I(t) (c,b).
    """

    omega_0 = param_cavity["omega_0"]
    gamma_0 = param_cavity["gamma_0"]
    L = param_cavity["L"]

    dt = param_time_evol["dt"]

    V_matrix = -1j * np.sqrt(gamma_0 / (2 * L)) * np.exp(
        1j * (omega_tab - omega_0) * t
    )

    c_new = b * V_matrix
    b_new = c @ np.conjugate(V_matrix)

    return -1j * dt * c_new, -1j * dt * b_new


def rk_propagator(
    c_init,
    b_init,
    omega_tab,
    param_cavity,
    param_time_evol,
    progress: bool = False,
    store_state: bool = True,
):
    """
    Propagates the state of the system using the RK4 scheme.
    """

    dt = param_time_evol["dt"]
    n_time_step = int(param_time_evol["T"] / dt)

    n_modes = len(c_init) // 2

    c_current = c_init
    b_current = b_init

    if store_state:
        c_array = np.zeros((n_time_step, 2 * n_modes), dtype=complex)
        b_array = np.zeros(n_time_step, dtype=complex)
        c_array[0] = c_init
        b_array[0] = b_init

    for i in tqdm(range(1, n_time_step), disable=not progress):
        t = i * dt

        c_n1, b_n1 = increment_state(
            c_current, b_current, omega_tab, param_cavity, param_time_evol, t
        )
        c_n2, b_n2 = increment_state(
            c_current + c_n1 / 2,
            b_current + b_n1 / 2,
            omega_tab,
            param_cavity,
            param_time_evol,
            t + dt / 2,
        )
        c_n3, b_n3 = increment_state(
            c_current + c_n2 / 2,
            b_current + b_n2 / 2,
            omega_tab,
            param_cavity,
            param_time_evol,
            t + dt / 2,
        )
        c_n4, b_n4 = increment_state(
            c_current + c_n3,
            b_current + b_n3,
            omega_tab,
            param_cavity,
            param_time_evol,
            t + dt,
        )

        c_new = c_current + (c_n1 + 2 * c_n2 + 2 * c_n3 + c_n4) / 6
        b_new = b_current + (b_n1 + 2 * b_n2 + 2 * b_n3 + b_n4) / 6

        if store_state:
            c_array[i] = c_new
            b_array[i] = b_new

        c_current = c_new
        b_current = b_new

    if store_state:
        return c_array, b_array
    return c_current, b_current
