from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.rk_integrator import rk_propagator


@dataclass
class ExperimentConfig:
    """
    Configuration class for the waveguide benchmark model.
    """

    param_photon: Dict[str, np.ndarray]
    param_cavity: Dict[str, float]
    param_time_evol: Dict[str, float]
    cutoffs: Dict[str, float]
    integrator_func: Optional[Any] = rk_propagator
    store_state: bool = True
