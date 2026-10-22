import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

benchmark_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(benchmark_root.parent))

from waveguide_model.setup import make_waveguide_config
from src import Experiment


def run_reflection_vs_frequency(param_photon_bis, param_cavity, param_time_evol, p0_values, cutoffs,
                                index_experiment=0, correction:bool=False, store_results:bool=True, progress:bool=True, output_dir=None):

    """
    Runs the reflection experiment for different central momenta.
    
    param_photon_bis (Dict) : Dictionary containing only {'sigma_p', 'x_init'}
    param_cavity (Dict) : Dictionary containing {'Omega_0', 'gamma_0', 'L'}
    param_time_evol (Dict) : Dictionary containing {'T', 'dt'}
    cutoffs (Dict) : Dictionary containing {'Omega_min', 'Omega_max'}
    p0_values (np.array) : Array of photon central momentum values
    index_experiment (int) : Index of the experiment if multiple are run in sequence.
    store_results (bool) : Whether to store the results in a CSV file.
    progress (bool) : Whether to display a progress bar.

    Returns:
    p0_values (np.array) : Array of photon central momentum values used.
    final_reflection_tab (np.array) : Array of final reflection values corresponding to each momentum.

    """

    #prepare the array to store the results
    final_reflection_tab = np.zeros(len(p0_values))

    for i in tqdm(range(len(p0_values)), disable=not progress):
        p_0 = p0_values[i]
        param_photon_current = param_photon_bis.copy()
        param_photon_current["p_init"] = p_0

        config = make_waveguide_config(photon=param_photon_current,
                                  L=param_cavity["L"], Omega_0=param_cavity["Omega_0"], gamma_0=param_cavity["gamma_0"],
                                  time=param_time_evol,
                                  cutoffs=cutoffs,
                                  store_state=False)
        
        experiment = Experiment(config)
        experiment.propagate_state(progress=False)
        P2n_array = experiment.compute_observables().backward
        final_reflection_tab[i] = P2n_array[-1]
    
    if store_results:
        data_to_save = {'photon_momentum_tab': p0_values, 'final_reflection_tab': final_reflection_tab}
        df = pd.DataFrame(data_to_save)
        output_dir = Path(output_dir) if output_dir is not None else benchmark_root / 'results' / 'csv_files' / 'reflection_vs_frequency'
        output_dir.mkdir(parents=True, exist_ok=True)
        if correction:
            df.to_csv(output_dir / f'reflection_vs_frequency_{index_experiment}_corrected.csv', index=False)
        else:
            df.to_csv(output_dir / f'reflection_vs_frequency_{index_experiment}_uncorrected.csv', index=False)

    return p0_values, final_reflection_tab
