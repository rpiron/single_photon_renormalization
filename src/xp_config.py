from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.rk_integrator import rk_propagator


def _as_1d_array(value: Any, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


@dataclass
class ExperimentConfig:
    """
    Minimal configuration for a one-photon scattering experiment.

    The photon dictionary has a single accepted shape:
        {"p0": ..., "Deltap0": ..., "x0": ...}
    where each entry is a d-dimensional vector. In 1D, scalars are accepted and
    converted to one-component vectors.

    Cutoffs are always named omega_min and omega_max.
    """

    dimension: int
    lengths: Sequence[float]
    atom_positions: Sequence[Sequence[float]] | Sequence[float] | float
    atom_frequencies: Sequence[float] | float
    atom_couplings: Sequence[complex] | complex
    photon: Mapping[str, Any]
    time: Mapping[str, float]
    cutoffs: Mapping[str, float]
    integrator_func: Callable[..., tuple[np.ndarray, np.ndarray]] = rk_propagator

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        if self.dimension not in (1, 2):
            raise ValueError("Only dimension=1 or dimension=2 are supported.")

        self.lengths = _as_1d_array(self.lengths, float)
        if len(self.lengths) != self.dimension:
            raise ValueError("lengths must contain one value per dimension.")
        if np.any(self.lengths <= 0):
            raise ValueError("All box lengths must be positive.")

        self.atom_frequencies = _as_1d_array(self.atom_frequencies, float)
        self.atom_couplings = _as_1d_array(self.atom_couplings, complex)
        if len(self.atom_frequencies) != len(self.atom_couplings):
            raise ValueError("atom_frequencies and atom_couplings must have same length.")

        self.atom_positions = self._normalize_positions(self.atom_positions)
        if len(self.atom_positions) != len(self.atom_frequencies):
            raise ValueError("atom_positions must contain one position per atom.")

        self.photon = self._normalize_photon(dict(self.photon))
        self.time = dict(self.time)
        self.cutoffs = self._normalize_cutoffs(dict(self.cutoffs))

        self._validate_time()

    @property
    def n_atoms(self) -> int:
        return len(self.atom_frequencies)

    @property
    def Omega_j(self) -> np.ndarray:
        return self.atom_frequencies

    @property
    def d_j(self) -> np.ndarray:
        return self.atom_couplings

    @property
    def volume(self) -> float:
        return float(np.prod(self.lengths))

    @property
    def omega_min(self) -> float:
        return self.cutoffs["omega_min"]

    @property
    def omega_max(self) -> float:
        return self.cutoffs["omega_max"]

    def _normalize_positions(self, value: Any) -> np.ndarray:
        positions = np.asarray(value, dtype=float)

        if positions.ndim == 0:
            positions = positions.reshape(1, 1)
        elif self.dimension == 1 and positions.ndim == 1:
            positions = positions.reshape(-1, 1)
        elif self.dimension == 2 and positions.ndim == 1:
            positions = positions.reshape(1, 2)

        if positions.ndim != 2 or positions.shape[1] != self.dimension:
            raise ValueError("atom_positions must have shape (n_atoms, dimension).")

        return positions

    def _normalize_cutoffs(self, cutoffs: dict[str, float]) -> dict[str, float]:
        expected = {"omega_min", "omega_max"}
        if set(cutoffs) != expected:
            raise ValueError("cutoffs must contain exactly omega_min and omega_max.")

        omega_min = float(cutoffs["omega_min"])
        omega_max = float(cutoffs["omega_max"])
        if omega_min < 0:
            raise ValueError("omega_min must be non-negative.")
        if omega_max <= omega_min:
            raise ValueError("omega_max must be larger than omega_min.")
        return {"omega_min": omega_min, "omega_max": omega_max}

    def _normalize_photon(self, photon: dict[str, Any]) -> dict[str, np.ndarray]:
        expected = ("p0", "Deltap0", "x0")
        if set(photon) != set(expected):
            raise ValueError("photon must contain exactly p0, Deltap0 and x0.")

        normalized = {
            key: self._as_vector(photon[key], f"photon['{key}']") for key in expected
        }
        if np.linalg.norm(normalized["p0"]) == 0:
            raise ValueError("photon['p0'] must be non-zero.")
        if np.any(normalized["Deltap0"] <= 0):
            raise ValueError("photon['Deltap0'] entries must be positive.")
        return normalized

    def _as_vector(self, value: Any, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.ndim == 0 and self.dimension == 1:
            vector = vector.reshape(1)
        if vector.shape != (self.dimension,):
            raise ValueError(f"{name} must have shape ({self.dimension},).")
        return vector

    def _validate_time(self) -> None:
        if "T" not in self.time or "dt" not in self.time:
            raise ValueError("time must define T and dt.")
        if float(self.time["T"]) <= 0 or float(self.time["dt"]) <= 0:
            raise ValueError("time['T'] and time['dt'] must be positive.")
