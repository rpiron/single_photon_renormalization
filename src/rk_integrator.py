"""Vectorized RK4 in the interaction picture (paper, propagation appendix)."""

import numpy as np
from tqdm.auto import tqdm

from .xp_config import time_grid


def state_derivative(c, b, g_0, detunings, t):
    interaction = g_0 * np.exp(-1j * detunings * t)
    return -1j * (interaction.conj() @ b), -1j * (c @ interaction)


def rk_propagator(config, progress=False):
    """Return (c, b), with a time axis only when store_state=True.

    No spatial phases or geometry are evaluated inside the RK loop.
    """
    times = time_grid(**config.time)
    detunings = config.omega_n[:, None] - config.Omega_0[None, :]
    c, b = config.c_init.copy(), config.b_init.copy()
    if config.store_state:
        c_array = np.empty((len(times), c.size), dtype=complex)
        b_array = np.empty((len(times), b.size), dtype=complex)
        c_array[0], b_array[0] = c, b
        
    for i in tqdm(range(1, len(times)), disable=not progress):
        t, h = times[i - 1], times[i] - times[i - 1]

        def increment(c_stage, b_stage, time):
            dc, db = state_derivative(c_stage, b_stage, config.g_0, detunings, time)
            return h * dc, h * db
        
        c1, b1 = increment(c, b, t)
        c2, b2 = increment(c + c1 / 2, b + b1 / 2, t + h / 2)
        c3, b3 = increment(c + c2 / 2, b + b2 / 2, t + h / 2)
        c4, b4 = increment(c + c3, b + b3, t + h)
        c = c + (c1 + 2 * c2 + 2 * c3 + c4) / 6
        b = b + (b1 + 2 * b2 + 2 * b3 + b4) / 6
        if config.store_state:
            c_array[i], b_array[i] = c, b
    return (c_array, b_array) if config.store_state else (c, b)
