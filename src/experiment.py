"""One shared experiment runner for already prepared single-photon models."""

import numpy as np

from .observables import compute_observables
from .rk_integrator import rk_propagator
from .xp_config import time_grid


class Experiment:
    def __init__(self, config):
        self.config = config
        self.omega_n = config.omega_n
        self.p_n = config.p_n
        self.mode_indices = config.mode_indices
        self.Omega_0 = config.Omega_0
        self.g_0 = config.g_0
        self.n_modes, self.n_atoms = config.g_0.shape
        self.incident_packet = config.c_init
        grid = time_grid(**config.time)
        self.times = grid if config.store_state else grid[-1:]
        self.messages = {}
        self.c_array = self.b_array = None
        self.observables = None
        if "lengths" in config.metadata:
            self.lengths = config.metadata["lengths"]

    def propagate_state(self, progress=False):
        integrator = self.config.integrator_func or rk_propagator
        self.c_array, self.b_array = integrator(self.config, progress=progress)
        self.observables = None
        self.messages.clear()
        return self.c_array, self.b_array

    def compute_observables(self, progress=False):
        if self.c_array is None:
            raise RuntimeError("Call propagate_state() before compute_observables().")
        obs = self.observables = compute_observables(self.c_array, self.b_array, self.config)
        if not np.all(np.isfinite(obs.norm)) or np.max(np.abs(obs.norm - 1)) > 1e-3:
            self.messages["probability_conservation"] = "Probability drift exceeds 1e-3; reduce dt."
        if obs.atomic_population[-1] > 1e-2:
            self.messages["final_excited_state"] = "Final atomic excitation exceeds 1e-2."
        return obs

    def get_messages(self):
        return self.messages.copy()
