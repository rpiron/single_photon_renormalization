import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.benchmark_config import make_benchmark_config
from src.experiment import Experiment

def run_reflection_vs_RKstep(param_photon, param_atom, T, dt_tab, cutoffs,
                             index_experiment=0, index_p0 = 0, store_results:bool=True, progress:bool=True):
    
    """
    Run the benchmark reflection experiment while varying the RK timestep.

    param_atom must contain {'Omega_A', 'g_A', 'lengths', 'positions'}.
    param_photon must contain {'p0', 'Deltap0', 'x0'}.
    cutoffs must contain {'omega_min', 'omega_max'}.
    """

    #prepare the array to store the results
    data_rows = []

    for i in tqdm(range(len(dt_tab)), disable=not progress):
        dt = dt_tab[i]
        param_time_evol = {'T': T, 'dt': dt}
        
        config = make_benchmark_config(
            param_photon,
            param_atom,
            param_time_evol,
            cutoffs,
            correction=True,
        )
            
        experiment = Experiment(config)
        experiment.propagate_state(progress=False)
        _, _, Rn_array = experiment.compute_observables()
            
        data_rows.append({'dt': dt, 'final_reflection': Rn_array[-1]})
    
    if store_results:

        df = pd.DataFrame(data_rows)
        filename = (f'reflection_vs_RKstep_p0_{index_p0}_xp{index_experiment}.csv' if index_experiment else 'reflection_vs_RKstep.csv')
        output_dir = project_root / 'benchmark_model' / 'results' / 'csv_files' / 'reflection_vs_RKstep'
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / filename, index=False)

    return data_rows
