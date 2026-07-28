import numpy as np
from tqdm import tqdm

from src.xp_config import ExperimentConfig


class Experiment:
    """Represent and propagate a one-photon waveguide scattering experiment."""

    def __init__(self, config: ExperimentConfig):
        self.integrator_func = config.integrator_func
        self.store_state = config.store_state

        self.param_photon = config.param_photon
        self.param_cavity = config.param_cavity
        self.param_time_evol = config.param_time_evol

        self.ir_cutoff = config.cutoffs["ir_cutoff"]
        self.uv_cutoff = config.cutoffs["uv_cutoff"]

        self.messages = {}

        omega_tab_half = np.array(
            [
                2 * np.pi * n / self.param_cavity["L"]
                for n in range(-1000000, 1000000)
                if 2 * np.pi * n / self.param_cavity["L"] <= self.uv_cutoff
                and 2 * np.pi * n / self.param_cavity["L"] >= self.ir_cutoff
            ]
        )

        self.omega_tab = np.concatenate((omega_tab_half, omega_tab_half))
        self.n_modes = len(omega_tab_half)

        n_time_step = int(self.param_time_evol["T"] / self.param_time_evol["dt"])

        if self.store_state:
            self.c_array = np.zeros((n_time_step, 2 * self.n_modes), dtype=complex)
            self.b_array = np.zeros(n_time_step, dtype=complex)

            self.An_array = np.zeros(n_time_step, dtype=float)
            self.P1n_array = np.zeros(n_time_step, dtype=float)
            self.P2n_array = np.zeros(n_time_step, dtype=float)
        else:
            self.c_array = np.zeros(2 * self.n_modes, dtype=complex)
            self.b_array = 0j

            self.An_array = np.zeros(1, dtype=float)
            self.P1n_array = np.zeros(1, dtype=float)
            self.P2n_array = np.zeros(1, dtype=float)

        if self.param_photon["sigma_p"] / (self.param_cavity["gamma_0"] / 2) > 0.1:
            self.messages["monochromatic_limit"] = (
                "Warning: Photon's wavepacket is not in the monochromatic limit."
            )

    def check_parameters(self):
        """Validate the experiment configuration."""
        return True

    def propagate_state(self, progress: bool = False):
        """Initialize and propagate the one-photon state."""

        b_init = 0j

        c_init = (
            np.exp(
                -((self.omega_tab - self.param_photon["p_0"]) ** 2)
                / (4 * self.param_photon["sigma_p"] ** 2)
            )
            * np.exp(-1j * self.omega_tab * self.param_photon["x_0"])
            * np.concatenate((np.ones(self.n_modes), np.zeros(self.n_modes)))
        )
        c_init = c_init / np.linalg.norm(c_init)

        c_array, b_array = self.integrator_func(
            c_init,
            b_init,
            self.omega_tab,
            self.param_cavity,
            self.param_time_evol,
            progress=progress,
            store_state=self.store_state,
        )

        self.c_array = c_array
        self.b_array = b_array

        return c_array, b_array

    def compute_observables(self, progress: bool = False):
        """Compute atomic and photon-sector populations from the propagated state."""

        if self.store_state:
            for i in tqdm(range(len(self.c_array)), disable=not progress):
                self.An_array[i] = np.abs(self.b_array[i]) ** 2
                self.P1n_array[i] = np.sum(np.abs(self.c_array[i, : self.n_modes]) ** 2)
                self.P2n_array[i] = np.sum(np.abs(self.c_array[i, self.n_modes :]) ** 2)
        else:
            self.An_array[-1] = np.abs(self.b_array) ** 2
            self.P1n_array[-1] = np.sum(np.abs(self.c_array[: self.n_modes]) ** 2)
            self.P2n_array[-1] = np.sum(np.abs(self.c_array[self.n_modes :]) ** 2)

        if not np.isclose(
            self.P1n_array[-1] + self.P2n_array[-1] + self.An_array[-1],
            1.0,
            atol=1e-3,
        ):
            self.messages["probability_conservation"] = (
                "Error: Probability not conserved at final time: "
                f"P1 + P2 + A = "
                f"{self.P1n_array[-1] + self.P2n_array[-1] + self.An_array[-1]:.6f} != 1."
            )

        if self.An_array[-1] > 1e-2:
            self.messages["final_excited_state"] = (
                "Warning: The population of the excited state at final time is "
                f"{self.An_array[-1]:.2e}, which is significant."
            )

        return self.An_array, self.P1n_array, self.P2n_array

    def get_messages(self):
        """Print and return messages generated during the experiment."""
        if self.messages:
            print("Current messsages:")
            for msg in self.messages:
                print("-", msg)
                print(self.messages[msg])

        return self.messages
