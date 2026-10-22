import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

benchmark_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(benchmark_root.parent))

from waveguide_model.setup import make_waveguide_config
from src import Experiment
from waveguide_model.bare_param import get_bare_parameters

pi = np.pi


def run_reflection_vs_bandwidth(param_photon, param_cavity_physical, param_time_evol, ir_tab, uv_tab,
                                index_p0 = 0, index_experiment = 0, correction=True, store_results:bool=True, progress:bool=True, output_dir=None):
    """
    Run the benchmark reflection experiment while varying the retained bandwidth.

    param_cavity_physical must contain {'Omega_A', 'gamma_A', 'L'}.
    """

    #Prepare to store the output
    reflection_tab = np.zeros(len(ir_tab))

    for i in tqdm(range(len(ir_tab)), disable=not progress):

        #Frequency window
        cutoffs = {"Omega_min": ir_tab[i], "Omega_max": uv_tab[i]}

        #Sanity check
        p_0 = param_photon["p_init"]
        if p_0 < cutoffs["Omega_min"] or p_0 > cutoffs["Omega_max"]:
            print("WARNING : The photon momentum is not included in the cutoff window. Returning NaN")
            reflection_tab[i] = np.nan
        
        else:
            omega_0, gamma_0 = get_bare_parameters(param_cavity_physical["Omega_A"],
                                                param_cavity_physical["gamma_A"],
                                                cutoffs, correction=correction)
            param_cavity = {"Omega_0": omega_0,
                            "gamma_0": gamma_0,
                            "L": param_cavity_physical["L"]}

            config = make_waveguide_config(photon=param_photon,
                                      L=param_cavity["L"], Omega_0=param_cavity["Omega_0"], gamma_0=param_cavity["gamma_0"],
                                      time=param_time_evol,
                                      cutoffs=cutoffs,
                                      store_state=False)
                
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)
            P2n_array = experiment.compute_observables().backward
            reflection_tab[i] = P2n_array[-1]

            del experiment

    if store_results:
        data_to_save = {
            "ir_tab": ir_tab,
            "uv_tab": uv_tab,
            "reflection_tab": reflection_tab,
        }
        df = pd.DataFrame(data_to_save)
        output_dir = Path(output_dir) if output_dir is not None else benchmark_root / 'results' / 'csv_files' / 'reflection_vs_bandwidth'
        output_dir.mkdir(parents=True, exist_ok=True)
        if correction:
            df.to_csv(output_dir / f'reflection_vs_bandwidth_omega{index_p0}_xp{index_experiment}_corrected.csv', index=False)
        else:
            df.to_csv(output_dir / f'reflection_vs_bandwidth_omega{index_p0}_xp{index_experiment}_uncorrected.csv', index=False)
        
    return ir_tab, uv_tab, reflection_tab
