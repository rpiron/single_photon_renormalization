import numpy as np

from src.one_dimensional_renormalization import get_1d_bare_parameters
from src.xp_config import ExperimentConfig


def make_benchmark_config(
    param_photon: dict[str, object],
    param_atom: dict[str, object],
    param_time_evol: dict[str, float],
    cutoffs: dict[str, float],
    correction: bool = False,
) -> ExperimentConfig:
    required = {"Omega_A", "g_A", "lengths", "positions"}
    missing = required.difference(param_atom)
    if missing:
        raise ValueError(f"param_atom is missing keys: {sorted(missing)}.")

    lengths = _as_1d_array(param_atom["lengths"], float)
    if len(lengths) != 1:
        raise ValueError("benchmark_model only builds 1D configs; lengths must have one entry.")

    positions = _as_1d_array(param_atom["positions"], float)
    n_atoms = len(positions)
    Omega_A = _atom_parameter_array(param_atom["Omega_A"], n_atoms, float, "Omega_A")
    g_A = _atom_parameter_array(param_atom["g_A"], n_atoms, float, "g_A")

    bare_parameters = [
        get_1d_bare_parameters(
            Omega_A[j],
            g_A[j],
            cutoffs["omega_min"],
            cutoffs["omega_max"],
            apply_corrections=correction,
        )
        for j in range(n_atoms)
    ]
    Omega_j = [parameters[0] for parameters in bare_parameters]
    g_j = [parameters[1] for parameters in bare_parameters]

    return ExperimentConfig(
        dimension=1,
        lengths=lengths,
        atom_positions=positions,
        atom_frequencies=Omega_j,
        atom_couplings=[1j * g for g in g_j],
        photon=param_photon,
        time=param_time_evol,
        cutoffs=cutoffs,
    )


def _as_1d_array(value, dtype):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _atom_parameter_array(value, n_atoms: int, dtype, name: str):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(n_atoms, array.item(), dtype=dtype)

    array = array.reshape(-1)
    if len(array) != n_atoms:
        raise ValueError(f"param_atom['{name}'] must contain one value per atom.")
    return array
