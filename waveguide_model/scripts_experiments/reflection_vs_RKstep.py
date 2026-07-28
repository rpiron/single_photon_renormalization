import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

benchmark_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(benchmark_root))

from src.xp_config import ExperimentConfig
from src.experiment import Experiment


def run_reflection_vs_RKstep(param_photon, param_cavity, T, dt_tab, cutoffs,
                             index_experiment=0, index_p0 = 0, store_results:bool=True, progress:bool=True):
    
    """
    Run the benchmark reflection experiment while varying the RK timestep.

    param_cavity must contain {'omega_0', 'gamma_0', 'L'}.
    param_photon must contain {'p_0', 'sigma_p', 'x_0'}.
    cutoffs must contain {'ir_cutoff', 'uv_cutoff'}.
    """

    #prepare the array to store the results
    data_rows = []

    for i in tqdm(range(len(dt_tab)), disable=not progress):
        dt = dt_tab[i]
        param_time_evol = {'T': T, 'dt': dt}
        
        config = ExperimentConfig(param_photon=param_photon,
                                  param_cavity=param_cavity,
                                  param_time_evol=param_time_evol,
                                  cutoffs=cutoffs,
                                  store_state=False)
            
        experiment = Experiment(config)
        experiment.propagate_state(progress=False)
        _, _, P2n_array = experiment.compute_observables()
            
        data_rows.append({'dt': dt, 'final_reflection': P2n_array[-1]})
    
    if store_results:

        df = pd.DataFrame(data_rows)
        filename = (f'reflection_vs_RKstep_p0_{index_p0}_xp{index_experiment}.csv' if index_experiment else 'reflection_vs_RKstep.csv')
        output_dir = benchmark_root / 'results' / 'csv_files' / 'reflection_vs_RKstep'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / filename, index=False)

    return data_rows
