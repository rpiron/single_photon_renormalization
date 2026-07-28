import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


setup_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(setup_root))

from src.bare_param import make_fiber_config
from src.experiment import Experiment


def run_transmittivity_vs_freqwindow(
    param_photon_bis,
    param_atom,
    param_time_evol,
    omega_init_values,
    cutoffs_list,
    apply_correction=False,
    store_results=True,
    progress=True,
):
    """
    Run the 2D transmittivity experiment for several incoming momenta and windows.

    param_photon_bis contains only sigma_p and x_0. For each run, p_0 is set to
    [omega_init, 0]. cutoffs_list is the list of frequency windows indexed by
    index_xp. If apply_correction=True, make_fiber_config replaces Omega_A,d_A by
    the fitted Omega_0,d_0 associated with that window.
    """

    output_dir = setup_root / "results" / "csv_files" / "transmittivity_vs_freqwindow"
    if store_results:
        output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "corrected" if apply_correction else "uncorrected"
    paths = []
    final_transmittivity = np.zeros((len(omega_init_values), len(cutoffs_list)))

    iterator = tqdm(
        total=len(omega_init_values) * len(cutoffs_list),
        disable=not progress,
        desc=f"transmittivity_vs_freqwindow [{suffix}]",
    )

    for index_omegainit, omega_init in enumerate(omega_init_values, start=1):
        for index_xp, cutoffs in enumerate(cutoffs_list, start=1):
            iterator.set_postfix(
                omega_init=index_omegainit,
                xp=index_xp,
                refresh=progress,
            )
            param_photon = dict(param_photon_bis)
            param_photon["p_0"] = [float(omega_init), 0.0]

            config = make_fiber_config(
                param_photon,
                param_atom,
                param_time_evol,
                cutoffs,
                correction=apply_correction,
            )
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)
            _, Tn_array, _ = experiment.compute_observables()

            final_transmittivity[index_omegainit - 1, index_xp - 1] = Tn_array[-1]

            if store_results:
                path = (
                    output_dir
                    / f"transmittivity_vs_freqwindow_omegainit_{index_omegainit}"
                    f"_xp_{index_xp}_{suffix}.csv"
                )
                _save_transmittivity_csv(
                    path,
                    experiment.times,
                    Tn_array,
                    omega_init,
                    cutoffs,
                    index_omegainit,
                    index_xp,
                    apply_correction,
                    param_atom,
                )
                paths.append(path)

            iterator.update(1)

    iterator.close()
    return final_transmittivity, paths


def _save_transmittivity_csv(
    path,
    time,
    Tn_array,
    omega_init,
    cutoffs,
    index_omegainit,
    index_xp,
    apply_correction,
    param_atom,
):
    data = pd.DataFrame({"time": time, "T_n0": Tn_array})
    with path.open("w", newline="") as stream:
        stream.write(f"# index_omegainit: {index_omegainit}\n")
        stream.write(f"# index_xp: {index_xp}\n")
        stream.write(f"# omega_init: {float(omega_init):.16g}\n")
        stream.write(f"# omega_init_pi: {float(omega_init) / np.pi:.16g}\n")
        stream.write(f"# ir_cutoff_pi: {float(cutoffs['ir_cutoff']) / np.pi:.16g}\n")
        stream.write(f"# uv_cutoff_pi: {float(cutoffs['uv_cutoff']) / np.pi:.16g}\n")
        stream.write(f"# correction: {bool(apply_correction)}\n")
        if param_atom["geometry"].get("id_geom") is not None:
            stream.write(f"# id_geom: {int(param_atom['geometry']['id_geom'])}\n")
        if param_atom.get("id_target_profile") is not None:
            stream.write(f"# id_target_profile: {int(param_atom['id_target_profile'])}\n")
        data.to_csv(stream, index=False)
