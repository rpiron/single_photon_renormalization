"""Numerical inputs in the notation of theory_tex/main_tqe.tex."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def normalize_cutoffs(cutoffs):
    """Validate the retained energy interval [Omega_min, Omega_max]."""
    low, high = float(cutoffs["Omega_min"]), float(cutoffs["Omega_max"])
    if not np.all(np.isfinite([low, high])) or not 0 <= low < high:
        raise ValueError("Require finite 0 <= Omega_min < Omega_max.")
    return {"Omega_min": low, "Omega_max": high}


def time_grid(T, dt):
    """Include 0 and T, shortening the last RK step when necessary."""
    T, dt = float(T), float(dt)
    if not np.isfinite(T) or not np.isfinite(dt) or T < 0 or dt <= 0:
        raise ValueError("Require finite T >= 0 and dt > 0.")
    if T == 0:
        return np.array([0.0])
    count = int(np.floor(T / dt))
    times = dt * np.arange(count + 1)
    if count and np.isclose(times[-1], T, rtol=1e-12, atol=0):
        times[-1] = T
    else:
        times = np.append(times, T)
    return times


@dataclass
class ExperimentConfig:
    """Prepared discrete Hamiltonian, independent of geometry storage/calibration.

    g_0[n,j] is the discrete g_{0,jn}, including 1/sqrt(V).
    All atomic states are vectors, including the one-atom case.
    """

    omega_n: np.ndarray
    Omega_0: np.ndarray
    g_0: np.ndarray
    c_init: np.ndarray
    b_init: np.ndarray
    time: dict
    p_n: np.ndarray
    mode_indices: np.ndarray
    forward_mask: np.ndarray
    backward_mask: np.ndarray
    store_state: bool = True
    integrator_func: Callable | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, dtype in [("omega_n", float), ("Omega_0", float),
                            ("g_0", complex), ("c_init", complex),
                            ("b_init", complex), ("p_n", float),
                            ("mode_indices", int), ("forward_mask", bool),
                            ("backward_mask", bool)]:
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values.")
            setattr(self, name, value)
        m, a = self.omega_n.size, self.Omega_0.size
        expected = {"omega_n": (m,), "Omega_0": (a,), "g_0": (m, a),
                    "c_init": (m,), "b_init": (a,), "forward_mask": (m,),
                    "backward_mask": (m,)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
        if m == 0 or a == 0:
            raise ValueError("At least one retained mode and one atom are required.")
        if self.p_n.ndim != 2 or self.p_n.shape[0] != m:
            raise ValueError("p_n must have shape (N_modes, dimension).")
        if self.mode_indices.shape != self.p_n.shape:
            raise ValueError("mode_indices and p_n must have matching shapes.")
        if np.any(self.forward_mask & self.backward_mask):
            raise ValueError("Forward and backward mode masks must be disjoint.")
        norm = np.vdot(self.c_init, self.c_init).real + np.vdot(self.b_init, self.b_init).real
        if not np.isclose(norm, 1, atol=1e-10, rtol=1e-10):
            raise ValueError("Initial total probability must equal one.")
        self.time = {"T": float(self.time["T"]), "dt": float(self.time["dt"])}
        time_grid(**self.time)
