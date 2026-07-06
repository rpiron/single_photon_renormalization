# Waveguide QED Simulator

## Overview

This repository contains a minimal momentum-space simulator for one-photon
light-matter scattering. The core code in `src/` now supports:

- one or two spatial dimensions;
- arbitrary periodic box lengths `L_1, ..., L_d`;
- one or several atoms with positions `r_j`, bare frequencies `Omega_j`, and
  continuum couplings `g_j`;
- vacuum dispersion `omega = ||p||`;
- RK4 propagation in the single-excitation sector.

The `benchmark_model/` folder keeps the one-dimensional single-atom benchmark
used in `main_tqe.pdf`, including its cutoff-renormalization helper and the
existing script/notebook/results workflow.

## Project Structure

```text
.
├── src/
│   ├── experiment.py       # Experiment class: modes, states, observables
│   ├── rk_integrator.py    # RK4 propagation in the one-photon sector
│   └── xp_config.py        # ExperimentConfig dataclass
├── benchmark_model/
│   ├── scripts_experiments/
│   ├── notebooks/
│   └── results/
├── 2d_setup/               # 2D fiber geometry experiments
│   ├── geometries/          # Saved atomic-interface CSV geometries
│   ├── notebooks/
│   └── results/
├── timedomain_cross_validation/
├── main_tqe.pdf
└── environment.yml
```

## Minimal Usage

```python
import numpy as np

from src.xp_config import ExperimentConfig
from src.experiment import Experiment

Omega = 10 * np.pi
g = np.sqrt(np.pi / 2)

config = ExperimentConfig(
    dimension=1,
    lengths=[50],
    atom_positions=[0.0],
    atom_frequencies=[Omega],
    atom_couplings=[1j * g],
    photon={"p0": [Omega], "Deltap0": [0.05 * np.pi], "x0": [-12.5]},
    time={"T": 25, "dt": 0.01},
    cutoffs={"omega_min": 7 * np.pi, "omega_max": 13 * np.pi},
)

experiment = Experiment(config)
experiment.propagate_state(progress=True)
atom_pop, transmitted, reflected = experiment.compute_observables()
```

For 2D, use `dimension=2`, two box lengths, 2D atom positions, and 2D photon
vectors:

```python
photon = {
    "p0": [Omega, 0.0],
    "Deltap0": [0.2, 0.2],
    "x0": [-10, 0],
}
```

## Benchmark Workflow

The benchmark scripts keep their historical public signatures and can still be
driven from notebooks:

```python
from benchmark_model.scripts_experiments.reflection_vs_frequency import (
    run_reflection_vs_frequency,
)
```

Generated CSV files are stored under `benchmark_model/results/csv_files/`.

## 2D Fiber Setup

Create and reload 2D geometries from the single source in `src.fiber_geometry`:

```python
from src.fiber_geometry import create_geometry, load_geometry, plot_geometry

kind = "slab"  # or "disk"
lengths = [50.0, 1.0]
matter_width = 1.0
matter_height = 1.0
N_A = 100
N_layer = 1  # ignored for disk geometries

geometry = create_geometry(kind, lengths, matter_width, matter_height, N_A, N_layer)
geometry = load_geometry(kind, lengths, matter_width, matter_height, N_A, N_layer)
```

`lengths` defines the simulation box. `kind="slab"` distributes `N_A` atoms
regularly across `N_layer` vertical layers inside the matter rectangle.
`kind="disk"` uses the largest circle or ellipse that fits inside
`matter_width` by `matter_height`.

## Environment

The simulations were developed for the Conda environment described in
`environment.yml`.
