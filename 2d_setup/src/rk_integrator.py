import numpy as np
from tqdm import tqdm


def increment_state(
    c,
    b,
    mode_omegas,
    momenta,
    atom_positions,
    atom_frequencies,
    d_j,
    volume,
    dt,
    t,
):
    # Continuous-to-discrete coupling from the paper:
    # g_{j,n}^{(0)} = i d_j exp(-i p_n.x_j) / sqrt(L1 L2).
    V_matrix = (
        1j
        * d_j[None, :]
        * np.exp(-1j * momenta @ atom_positions.T)
        / np.sqrt(volume)
    )
    detunings = mode_omegas[:, None] - atom_frequencies[None, :]
    interaction = V_matrix * np.exp(-1j * detunings * t)
    c_increment = interaction.conj() @ b
    b_increment = c @ interaction
    return -1j * dt * c_increment, -1j * dt * b_increment


def rk_propagator(
    c_init,
    b_init,
    mode_omegas,
    momenta,
    atom_positions,
    atom_frequencies,
    d_j,
    volume,
    time,
    progress=False,
):
    dt = float(time["dt"])
    n_steps = int(round(float(time["T"]) / dt)) + 1

    c_array = np.zeros((n_steps, len(c_init)), dtype=complex)
    b_array = np.zeros((n_steps, len(b_init)), dtype=complex)
    c_array[0] = c_init
    b_array[0] = b_init

    for i in tqdm(range(1, n_steps), disable=not progress):
        t = (i - 1) * dt
        c = c_array[i - 1]
        b = b_array[i - 1]

        c_1, b_1 = increment_state(
            c,
            b,
            mode_omegas,
            momenta,
            atom_positions,
            atom_frequencies,
            d_j,
            volume,
            dt,
            t,
        )
        c_2, b_2 = increment_state(
            c + c_1 / 2,
            b + b_1 / 2,
            mode_omegas,
            momenta,
            atom_positions,
            atom_frequencies,
            d_j,
            volume,
            dt,
            t + dt / 2,
        )
        c_3, b_3 = increment_state(
            c + c_2 / 2,
            b + b_2 / 2,
            mode_omegas,
            momenta,
            atom_positions,
            atom_frequencies,
            d_j,
            volume,
            dt,
            t + dt / 2,
        )
        c_4, b_4 = increment_state(
            c + c_3,
            b + b_3,
            mode_omegas,
            momenta,
            atom_positions,
            atom_frequencies,
            d_j,
            volume,
            dt,
            t + dt,
        )

        c_array[i] = c + (c_1 + 2 * c_2 + 2 * c_3 + c_4) / 6
        b_array[i] = b + (b_1 + 2 * b_2 + 2 * b_3 + b_4) / 6

    return c_array, b_array
