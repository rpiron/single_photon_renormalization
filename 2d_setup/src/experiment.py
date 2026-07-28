import numpy as np

from .xp_config import ExperimentConfig


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.integrator_func = config.integrator_func
        self.lengths = config.lengths
        self.volume = config.volume
        self.atom_positions = config.atom_positions
        self.atom_frequencies = config.atom_frequencies
        self.atom_couplings = config.atom_couplings
        self.Omega_j = config.Omega_j
        self.d_j = config.d_j
        self.photon = config.photon
        self.time = config.time
        self.cutoffs = config.cutoffs
        self.messages = {}

        delta_p = 2 * np.pi / self.lengths
        n_max = np.floor(self.cutoffs["uv_cutoff"] / delta_p).astype(int)
        grids = np.meshgrid(
            np.arange(-n_max[0], n_max[0] + 1),
            np.arange(-n_max[1], n_max[1] + 1),
            indexing="ij",
        )
        mode_indices = np.stack([grid.reshape(-1) for grid in grids], axis=1)
        momenta = mode_indices * delta_p[None, :]
        omega_tab = np.linalg.norm(momenta, axis=1)
        mask = (omega_tab >= self.cutoffs["ir_cutoff"]) & (
            omega_tab <= self.cutoffs["uv_cutoff"]
        )
        order = np.argsort(omega_tab[mask], kind="mergesort")

        self.mode_indices = mode_indices[mask][order]
        self.momenta = momenta[mask][order]
        self.omega_tab = omega_tab[mask][order]
        self.n_modes = len(self.omega_tab)
        self.n_atoms = len(self.atom_frequencies)

        reduced_momenta = (self.momenta - self.photon["p_0"]) / self.photon["sigma_p"]
        incident_packet = (
            np.exp(-np.sum(reduced_momenta**2, axis=1) / 4)
            * np.exp(-1j * self.momenta @ self.photon["x_0"])
        )
        self.incident_packet = incident_packet / np.linalg.norm(incident_packet)

        dt = self.time["dt"]
        n_steps = int(round(self.time["T"] / dt)) + 1
        self.times = dt * np.arange(n_steps)

        self.c_array = np.zeros((len(self.times), self.n_modes), dtype=complex)
        self.b_array = np.zeros((len(self.times), self.n_atoms), dtype=complex)
        self.atom_populations_array = np.zeros((len(self.times), self.n_atoms))
        self.An_array = np.zeros(len(self.times))
        self.photon_number_array = np.zeros(len(self.times))
        self.Tn_array = np.zeros(len(self.times))
        self.Rn_array = np.zeros(len(self.times))
        self.forward_array = np.zeros(len(self.times))
        self.backward_array = np.zeros(len(self.times))
        self.side_array = np.zeros(len(self.times))

    def propagate_state(self, progress=False):
        b_init = np.zeros(self.n_atoms, dtype=complex)
        self.c_array, self.b_array = self.integrator_func(
            self.incident_packet,
            b_init,
            self.omega_tab,
            self.momenta,
            self.atom_positions,
            self.atom_frequencies,
            self.d_j,
            self.volume,
            self.time,
            progress=progress,
        )
        return self.c_array, self.b_array

    def compute_observables(self):
        self.atom_populations_array = np.abs(self.b_array) ** 2
        self.An_array = np.sum(self.atom_populations_array, axis=1)
        photon_density = np.abs(self.c_array) ** 2
        self.photon_number_array = np.sum(photon_density, axis=1)

        self.Tn_array = np.abs(self.c_array @ np.conjugate(self.incident_packet)) ** 2
        self.Rn_array = self.photon_number_array - self.Tn_array

        direction = self.photon["p_0"] / np.linalg.norm(self.photon["p_0"])
        projections = self.momenta @ direction
        self.forward_array = np.sum(photon_density[:, projections > 1e-12], axis=1)
        self.backward_array = np.sum(photon_density[:, projections < -1e-12], axis=1)
        self.side_array = (
            self.photon_number_array - self.forward_array - self.backward_array
        )
        return self.An_array, self.Tn_array, self.Rn_array

    def get_messages(self):
        return self.messages
