import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

benchmark_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(benchmark_root))

from src.xp_config import ExperimentConfig
from src.experiment import Experiment
from src.bare_param import get_bare_param_n

pi = np.pi


def run_reflection_vs_bandwidth(param_photon, param_cavity_physical, param_time_evol, ir_tab, uv_tab,
                                index_p0 = 0, index_experiment = 0, n=1, store_results:bool=True, progress:bool=True):
    """
    Run the benchmark reflection experiment while varying the retained bandwidth.

    param_cavity_physical must contain {'omega_A', 'gamma_A', 'L'}.
    """

    #Prepare to store the output
    reflection_tab = np.zeros(len(ir_tab))

    for i in tqdm(range(len(ir_tab)), disable=not progress):

        #Frequency window
        cutoffs = {"ir_cutoff": ir_tab[i], "uv_cutoff": uv_tab[i]}

        #Sanity check
        p_0 = param_photon["p_0"]
        if p_0 < cutoffs["ir_cutoff"] or p_0 > cutoffs["uv_cutoff"]:
            print("WARNING : The photon momentum is not included in the cutoff window. Returning NaN")
            reflection_tab[i] = np.nan
        
        else:
            omega_0, gamma_0 = get_bare_param_n(param_cavity_physical["omega_A"],
                                                param_cavity_physical["gamma_A"],
                                                cutoffs["ir_cutoff"],
                                                cutoffs["uv_cutoff"],
                                                n=n)
            param_cavity = {"omega_0": omega_0,
                            "gamma_0": gamma_0,
                            "L": param_cavity_physical["L"]}

            config = ExperimentConfig(param_photon=param_photon,
                                      param_cavity=param_cavity,
                                      param_time_evol=param_time_evol,
                                      cutoffs=cutoffs,
                                      store_state=False)
                
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)
            _, _, P2n_array = experiment.compute_observables()
            reflection_tab[i] = P2n_array[-1]

            del experiment

    if store_results:
        data_to_save = {
            "ir_tab": ir_tab,
            "uv_tab": uv_tab,
            "reflection_tab": reflection_tab,
        }
        df = pd.DataFrame(data_to_save)
        output_dir = benchmark_root / 'results' / 'csv_files' / 'reflection_vs_bandwidth'
        output_dir.mkdir(parents=True, exist_ok=True)
        if n != -1:
            df.to_csv(output_dir / f'reflection_vs_bandwidth_p0_{index_p0}_xp{index_experiment}_corrected.csv', index=False)
        else:
            df.to_csv(output_dir / f'reflection_vs_bandwidth_p0_{index_p0}_xp{index_experiment}_uncorrected.csv', index=False)
        
    return ir_tab, uv_tab, reflection_tab
