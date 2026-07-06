import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.experiment import Experiment
from src.fiber_geometry import make_fiber_config


def _geometry_suffix(geometry: dict) -> str:
    return geometry["label"]


def result_filename(geometry: dict, index_experiment: int) -> str:
    return f"reflection_vs_frequency_{_geometry_suffix(geometry)}_xp{index_experiment}.csv"


def run_reflection_vs_frequency(
    param_photon_bis,
    param_atom,
    param_time_evol,
    frequency_values,
    cutoffs,
    geometry,
    index_experiment=0,
    correction_scheme=None,
    store_results: bool = True,
    progress: bool = True,
):
    """
    Runs the 2D reflection/transmission experiment for different photon energies.

    Parameters:
    param_photon_bis (Dict): Dictionary containing {'Deltap0', 'x0'}.
    param_atom (Dict): Dictionary containing {'Omega_A', 'g_A', 'geometry'}.
    param_time_evol (Dict): Dictionary containing {'T', 'dt'}.
    frequency_values (np.array): Incoming photon energies omega_init.
    cutoffs (Dict): Dictionary containing {'omega_min', 'omega_max'}.
    geometry (Dict): Loaded geometry with lengths, matter dimensions, N_A, label, atom_positions.
    correction_scheme (str | None): None, '1d_scheme', or '2d_scheme'.
    """

    final_reflection_tab = np.zeros(len(frequency_values))
    final_transmission_tab = np.zeros(len(frequency_values))
    final_vertical_tab = np.zeros(len(frequency_values))
    final_atom_excitation_tab = np.zeros(len(frequency_values))
    final_total_probability_tab = np.zeros(len(frequency_values))
    n_modes_tab = np.zeros(len(frequency_values), dtype=int)

    for i in tqdm(range(len(frequency_values)), disable=not progress):
        omega_init = frequency_values[i]

        param_photon_current = param_photon_bis.copy()
        param_photon_current["p0"] = [omega_init, 0.0]

        config = make_fiber_config(
            param_photon_current,
            param_atom,
            param_time_evol,
            cutoffs,
            correction_scheme=correction_scheme,
        )

        experiment = Experiment(config)
        n_modes_tab[i] = experiment.n_modes
        experiment.propagate_state(progress=False)
        An_array, _, _ = experiment.compute_observables()

        photon_density_final = np.abs(experiment.c_array[-1]) ** 2
        px = experiment.momenta[:, 0]

        final_transmission_tab[i] = np.sum(photon_density_final[px > 0])
        final_reflection_tab[i] = np.sum(photon_density_final[px < 0])
        final_vertical_tab[i] = np.sum(photon_density_final[np.isclose(px, 0.0)])
        final_atom_excitation_tab[i] = An_array[-1]
        final_total_probability_tab[i] = (
            final_reflection_tab[i]
            + final_transmission_tab[i]
            + final_vertical_tab[i]
            + final_atom_excitation_tab[i]
        )

        del experiment

    if store_results:
        data_to_save = {
            "photon_frequency_tab": frequency_values,
            "omega_min": cutoffs["omega_min"] * np.ones(len(frequency_values)),
            "omega_max": cutoffs["omega_max"] * np.ones(len(frequency_values)),
            "n_modes_tab": n_modes_tab,
            "final_reflection_tab": final_reflection_tab,
            "final_transmission_tab": final_transmission_tab,
            "final_vertical_tab": final_vertical_tab,
            "final_atom_excitation_tab": final_atom_excitation_tab,
            "final_total_probability_tab": final_total_probability_tab,
        }
        df = pd.DataFrame(data_to_save)
        output_dir = (
            project_root
            / "2d_setup"
            / "results"
            / "csv_files"
            / "reflection_vs_frequency"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / result_filename(geometry, index_experiment), index=False)

    return frequency_values, final_reflection_tab, final_transmission_tab
