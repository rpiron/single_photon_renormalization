import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.benchmark_config import make_benchmark_config
from src.experiment import Experiment

pi = np.pi

def run_reflection_vs_bandwidth(param_photon, param_atom, param_time_evol, omega_min_tab, omega_max_tab,
                                index_p0 = 0, index_experiment = 0, correction:bool=False, store_results:bool=True, progress:bool=True):
    """
    Run the benchmark reflection experiment while varying the retained bandwidth.

    param_atom must contain {'Omega_A', 'g_A', 'lengths', 'positions'}.
    """

    #Prepare to store the output
    reflection_tab = np.zeros(len(omega_min_tab))

    for i in tqdm(range(len(omega_min_tab)), disable=not progress):

        #Frequency window
        cutoffs = {"omega_min": omega_min_tab[i], "omega_max": omega_max_tab[i]}

        #Sanity check
        omega0 = np.linalg.norm(param_photon["p0"])
        if omega0 < cutoffs["omega_min"] or omega0 > cutoffs["omega_max"]:
            print("WARNING : The photon frequency is not included in the frequency window. Returning NaN")
            reflection_tab[i] = np.nan
        
        else:
            config = make_benchmark_config(
                param_photon,
                param_atom,
                param_time_evol,
                cutoffs,
                correction=correction,
            )
                
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)
            _, _, Rn_array = experiment.compute_observables()            
            reflection_tab[i] = Rn_array[-1]

            del experiment

    if store_results:
        data_to_save = {
            "omega_min_tab": omega_min_tab,
            "omega_max_tab": omega_max_tab,
            "reflection_tab": reflection_tab,
        }
        df = pd.DataFrame(data_to_save)
        output_dir = project_root / 'benchmark_model' / 'results' / 'csv_files' / 'reflection_vs_bandwidth'
        output_dir.mkdir(parents=True, exist_ok=True)
        if correction:
            df.to_csv(output_dir / f'reflection_vs_bandwidth_p0_{index_p0}_xp{index_experiment}_corrected.csv', index=False)
        else:
            df.to_csv(output_dir / f'reflection_vs_bandwidth_p0_{index_p0}_xp{index_experiment}_uncorrected.csv', index=False)
        
    return omega_min_tab, omega_max_tab, reflection_tab
