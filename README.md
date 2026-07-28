# Waveguide QED Simulator

## Overview

This repository contains two independent minimal momentum-space simulators for
one-photon light-matter scattering:

- `waveguide_model/`: the one-dimensional single-atom waveguide benchmark used
  in `main_tqe.pdf`;
- `2d_setup/`: the two-dimensional fiber setup with its own geometry database,
  target profiles, correction workflow, and local source code.

Each model folder has its own local `src/` directory and should be treated
independently.

## Project Structure

```text
.
├── waveguide_model/
│   ├── src/
│   ├── scripts_experiments/
│   ├── notebooks/
│   └── results/
├── 2d_setup/               # 2D fiber geometry experiments
│   ├── src/
│   ├── database_fiber/
│   ├── notebooks/
│   └── results/
├── timedomain_cross_validation/
├── main_tqe.pdf
└── environment.yml
```

## Waveguide Model

Run the one-dimensional benchmark scripts from `waveguide_model/`:

```python
from waveguide_model.scripts_experiments.reflection_vs_frequency import (
    run_reflection_vs_frequency,
)
```

Generated CSV files are stored under `waveguide_model/results/csv_files/`.

## 2D Fiber Setup

Create and reload 2D geometries from `2d_setup/src/generate_fiber_geom.py`.
Geometry CSV files are named by id and stored in `2d_setup/database_fiber/`:

```python
from src.generate_fiber_geom import create_geometry, load_geometry_by_id

kind = "slab"  # or "disk"
lengths = [50.0, 1.0]
matter_width = 1.0
matter_height = 1.0
N_A = 100
N_layer = 1  # ignored for disk geometries
rotation = 0.2  # radians, global rotation around the origin

geometry = create_geometry(
    kind,
    lengths,
    matter_width,
    matter_height,
    N_A,
    N_layer,
    rotation=rotation,
)
geometry_by_id = load_geometry_by_id(1)
```

Build a 2D experiment config with the local momentum notation:

```python
from src.bare_param import make_fiber_config

param_photon = {
    "p_0": [10 * np.pi, 0.0],
    "sigma_p": [np.pi, np.pi],
    "x_0": [-12.5, 0.0],
}
cutoffs = {"ir_cutoff": 8 * np.pi, "uv_cutoff": 12 * np.pi}
param_atom = {"Omega_A": Omega_A, "d_A": d_A, "geometry": geometry}
config = make_fiber_config(param_photon, param_atom, param_time_evol, cutoffs, correction=False)
```

`lengths` defines the simulation box. `kind="slab"` distributes `N_A` atoms
regularly across `N_layer` vertical layers inside the matter rectangle.
`kind="disk"` uses the largest circle or ellipse that fits inside
`matter_width` by `matter_height`.

## Environment

The simulations were developed for the Conda environment described in
`environment.yml`.
