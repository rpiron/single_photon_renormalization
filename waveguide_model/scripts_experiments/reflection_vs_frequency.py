import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

benchmark_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(benchmark_root))

from src.xp_config import ExperimentConfig
from src.experiment import Experiment


def run_reflection_vs_frequency(param_photon_bis, param_cavity, param_time_evol, p0_values, cutoffs,
                                index_experiment=0, correction:bool=False, store_results:bool=True, progress:bool=True):

    """
    Runs the reflection experiment for different central momenta.
    
    param_photon_bis (Dict) : Dictionary containing only {'sigma_p', 'x_0'}
    param_cavity (Dict) : Dictionary containing {'omega_0', 'gamma_0', 'L'}
    param_time_evol (Dict) : Dictionary containing {'T', 'dt'}
    cutoffs (Dict) : Dictionary containing {'ir_cutoff', 'uv_cutoff'}
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
        param_photon_current["p_0"] = p_0

        config = ExperimentConfig(param_photon=param_photon_current,
                                  param_cavity=param_cavity,
                                  param_time_evol=param_time_evol,
                                  cutoffs=cutoffs,
                                  store_state=False)
        
        experiment = Experiment(config)
        experiment.propagate_state(progress=False)
        _, _, P2n_array = experiment.compute_observables()
        final_reflection_tab[i] = P2n_array[-1]
    
    if store_results:
        data_to_save = {'photon_momentum_tab': p0_values, 'final_reflection_tab': final_reflection_tab}
        df = pd.DataFrame(data_to_save)
        output_dir = benchmark_root / 'results' / 'csv_files' / 'reflection_vs_frequency'
        output_dir.mkdir(parents=True, exist_ok=True)
        if correction:
            df.to_csv(output_dir / f'reflection_vs_frequency_{index_experiment}_corrected.csv', index=False)
        else:
            df.to_csv(output_dir / f'reflection_vs_frequency_{index_experiment}_uncorrected.csv', index=False)

    return p0_values, final_reflection_tab
