from dataclasses import dataclass

import numpy as np

from .rk_integrator import rk_propagator


@dataclass
class ExperimentConfig:
    lengths: np.ndarray
    atom_positions: np.ndarray
    atom_frequencies: np.ndarray
    atom_couplings: np.ndarray
    photon: dict
    time: dict
    cutoffs: dict
    integrator_func: object = rk_propagator

    def __post_init__(self):
        self.dimension = 2
        self.lengths = np.asarray(self.lengths, dtype=float).reshape(2)
        self.atom_positions = np.asarray(self.atom_positions, dtype=float).reshape(-1, 2)
        self.atom_frequencies = np.asarray(self.atom_frequencies, dtype=float).reshape(-1)
        self.atom_couplings = np.asarray(self.atom_couplings, dtype=complex).reshape(-1)
        self.photon = {
            "p_0": np.asarray(self.photon["p_0"], dtype=float).reshape(2),
            "sigma_p": np.asarray(self.photon["sigma_p"], dtype=float).reshape(2),
            "x_0": np.asarray(self.photon["x_0"], dtype=float).reshape(2),
        }
        self.time = {"T": float(self.time["T"]), "dt": float(self.time["dt"])}
        self.cutoffs = normalize_cutoffs(self.cutoffs)

    @property
    def volume(self):
        return float(np.prod(self.lengths))

    @property
    def n_atoms(self):
        return len(self.atom_positions)

    @property
    def Omega_j(self):
        return self.atom_frequencies

    @property
    def d_j(self):
        return self.atom_couplings


def normalize_cutoffs(cutoffs):
    if "ir_cutoff" in cutoffs:
        ir_cutoff = cutoffs["ir_cutoff"]
        uv_cutoff = cutoffs["uv_cutoff"]
    else:
        ir_cutoff = cutoffs["omega_min"]
        uv_cutoff = cutoffs["omega_max"]
    return {"ir_cutoff": float(ir_cutoff), "uv_cutoff": float(uv_cutoff)}
