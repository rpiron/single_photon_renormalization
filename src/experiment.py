import numpy as np

from src.xp_config import ExperimentConfig


class Experiment:
    """
    One-photon scattering experiment for one or several atoms in 1D or 2D.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.integrator_func = config.integrator_func

        self.dimension = config.dimension
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

        self.messages: dict[str, str] = {}

        self.mode_indices, self.momenta, self.omega_tab = self._build_modes()
        self.n_modes = len(self.omega_tab)
        self.n_atoms = len(self.atom_frequencies)
        self.mode_couplings = self._build_mode_couplings()
        self.times = self._build_times()

        self.c_array = np.zeros((len(self.times), self.n_modes), dtype=complex)
        self.b_array = np.zeros((len(self.times), self.n_atoms), dtype=complex)

        self.atom_populations_array = np.zeros((len(self.times), self.n_atoms))
        self.An_array = np.zeros(len(self.times))
        self.photon_number_array = np.zeros(len(self.times))
        self.Tn_array = np.zeros(len(self.times))
        self.Rn_array = np.zeros(len(self.times))
        self.side_array = np.zeros(len(self.times))

    def propagate_state(self, progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
        c_init = self._initial_photon_state()
        b_init = self._initial_atom_state()

        c_array, b_array = self.integrator_func(
            c_init,
            b_init,
            self.omega_tab,
            self.atom_frequencies,
            self.mode_couplings,
            self.time,
            progress=progress,
        )

        self.c_array = c_array
        self.b_array = b_array

        return c_array, b_array

    def compute_observables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.atom_populations_array = np.abs(self.b_array) ** 2
        self.An_array = np.sum(self.atom_populations_array, axis=1)
        photon_density = np.abs(self.c_array) ** 2
        self.photon_number_array = np.sum(photon_density, axis=1)

        direction = self._incident_direction()
        projections = self.momenta @ direction
        tolerance = 1e-12
        forward_mask = projections > tolerance
        backward_mask = projections < -tolerance

        self.Tn_array = np.sum(photon_density[:, forward_mask], axis=1)
        self.Rn_array = np.sum(photon_density[:, backward_mask], axis=1)
        self.side_array = self.photon_number_array - self.Tn_array - self.Rn_array

        total_probability = self.photon_number_array[-1] + self.An_array[-1]
        if not np.isclose(total_probability, 1.0, atol=1e-3):
            self.messages["probability_conservation"] = (
                "Error: probability not conserved at final time: "
                f"P = {total_probability:.6f} != 1."
            )

        if self.An_array[-1] > 1e-2:
            self.messages["final_excited_state"] = (
                "Warning: the total excited-state population at final time is "
                f"{self.An_array[-1]:.2e}, which is significant."
            )

        return self.An_array, self.Tn_array, self.Rn_array

    def get_messages(self) -> dict[str, str]:
        if self.messages:
            print("Current messages:")
            for key, message in self.messages.items():
                print("-", key)
                print(message)

        return self.messages

    def _build_modes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta_p = 2 * np.pi / self.lengths
        n_max = np.floor(self.cutoffs["omega_max"] / delta_p).astype(int)

        axes = [np.arange(-limit, limit + 1) for limit in n_max]
        grids = np.meshgrid(*axes, indexing="ij")
        mode_indices = np.stack([grid.reshape(-1) for grid in grids], axis=1)
        momenta = mode_indices * delta_p[None, :]
        omegas = np.linalg.norm(momenta, axis=1)

        mask = (omegas >= self.cutoffs["omega_min"]) & (
            omegas <= self.cutoffs["omega_max"]
        )
        mode_indices = mode_indices[mask]
        momenta = momenta[mask]
        omegas = omegas[mask]

        order = np.argsort(omegas, kind="mergesort")
        return mode_indices[order], momenta[order], omegas[order]

    def _build_mode_couplings(self) -> np.ndarray:
        phase = np.exp(-1j * self.momenta @ self.atom_positions.T)
        return phase * self.d_j[None, :] / np.sqrt(self.volume)

    def _build_times(self) -> np.ndarray:
        dt = float(self.time["dt"])
        n_steps = int(round(float(self.time["T"]) / dt)) + 1
        return dt * np.arange(n_steps)

    def _initial_photon_state(self) -> np.ndarray:
        p0 = self.photon["p0"]
        Deltap0 = self.photon["Deltap0"]
        x0 = self.photon["x0"]

        reduced_momenta = (self.momenta - p0[None, :]) / Deltap0[None, :]
        envelope = np.exp(-np.sum(reduced_momenta**2, axis=1) / 4)
        phase = np.exp(-1j * self.momenta @ x0)
        return self._normalize_state(envelope * phase)

    def _initial_atom_state(self) -> np.ndarray:
        return np.zeros(self.n_atoms, dtype=complex)

    def _central_momentum(self) -> np.ndarray:
        return self.photon["p0"]

    def _incident_direction(self) -> np.ndarray:
        p0 = self._central_momentum()
        return p0 / np.linalg.norm(p0)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("Initial photon state has zero norm on retained modes.")
        return state / norm
