import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

benchmark_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(benchmark_root.parent))

from waveguide_model.setup import make_waveguide_config
from src import Experiment


def run_reflection_vs_RKstep(param_photon, param_cavity, T, dt_tab, cutoffs,
                             index_experiment=0, index_p0 = 0, store_results:bool=True, progress:bool=True, output_dir=None):
    
    """
    Run the benchmark reflection experiment while varying the RK timestep.

    param_cavity must contain {'Omega_0', 'gamma_0', 'L'}.
    param_photon must contain {'p_init', 'sigma_p', 'x_init'}.
    cutoffs must contain {'Omega_min', 'Omega_max'}.
    """

    #prepare the array to store the results
    data_rows = []

    for i in tqdm(range(len(dt_tab)), disable=not progress):
        dt = dt_tab[i]
        param_time_evol = {'T': T, 'dt': dt}
        
        config = make_waveguide_config(photon=param_photon,
                                  L=param_cavity["L"], Omega_0=param_cavity["Omega_0"], gamma_0=param_cavity["gamma_0"],
                                  time=param_time_evol,
                                  cutoffs=cutoffs,
                                  store_state=False)
            
        experiment = Experiment(config)
        experiment.propagate_state(progress=False)
        P2n_array = experiment.compute_observables().backward
            
        data_rows.append({'dt': dt, 'final_reflection': P2n_array[-1]})
    
    if store_results:

        df = pd.DataFrame(data_rows)
        filename = (f'reflection_vs_RKstep_omegap_{index_p0}_xp{index_experiment}.csv' if index_experiment else 'reflection_vs_RKstep.csv')
        output_dir = Path(output_dir) if output_dir is not None else benchmark_root / 'results' / 'csv_files' / 'reflection_vs_RKstep'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / filename, index=False)

    return data_rows
