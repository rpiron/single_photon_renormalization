# Single-photon scattering

Single-photon scattering simulations with a shared propagation engine for a bidirectional waveguide with one atom and two-dimensional atomic arrays. The repository includes model preparation, bare-parameter calibration, example notebooks, parameter scans, and an independent Julia time-domain comparison.

## Getting started

From the repository root, create and activate the Python environment, then start JupyterLab:

```bash
conda env create -f environment.yml
conda activate single-photon-renorm
jupyter lab
```

Open the [waveguide example](waveguide_model/notebooks/experiment_class_example.ipynb) or the [2D example](2d_setup/notebooks/experiment_class_example_2d.ipynb), select the `single-photon-renorm` environment as the kernel, and run the cells from top to bottom. The plotting cells use LaTeX, which must be installed separately from the Python environment.

## Repository structure

The tree below describes the current working tree, excluding files and directories matched by `.gitignore`. Repeated result files are grouped by filename pattern; `.gitkeep` files are empty directory placeholders.

```text
single_photon_code/
├── .gitignore                          # Exclusions for local files and generated caches
├── README.md                           # Project overview and usage
├── environment.yml                     # Conda environment and Python dependencies
├── src/                                # Numerical engine shared by both Python models
│   ├── __init__.py                      # Shared package exports
│   ├── xp_config.py                     # ExperimentConfig, frequency cutoffs, and time grids
│   ├── modes.py                         # Momentum grids and Gaussian photon wave packets
│   ├── rk_integrator.py                 # Vectorized fourth-order Runge–Kutta propagation
│   ├── experiment.py                    # Experiment runner and probability diagnostics
│   └── observables.py                   # Atomic/photon populations, norm, and mode fidelity
├── waveguide_model/                    # One atom at the origin of a bidirectional waveguide
│   ├── __init__.py                      # Waveguide package exports
│   ├── setup.py                         # make_waveguide_config: prepare modes and couplings
│   ├── bare_param.py                    # Analytical calibration of bare atomic parameters
│   ├── notebooks/
│   │   ├── experiment_class_example.ipynb
│   │   │                               # First waveguide simulation and observables
│   │   ├── reflection_vs_RKstep_results.ipynb
│   │   │                               # Time-step convergence analysis
│   │   ├── reflection_vs_bandwidth_results.ipynb
│   │   │                               # Reflection versus retained frequency bandwidth
│   │   └── reflection_vs_frequency_results.ipynb
│   │                                   # Reflection spectra with/without calibration
│   ├── scripts_experiments/             # Parameter-scan functions used by the notebooks
│   │   ├── reflection_vs_RKstep.py
│   │   ├── reflection_vs_bandwidth.py
│   │   └── reflection_vs_frequency.py
│   └── results/
│       ├── csv_files/                   # Saved numerical data, grouped by scan
│       │   ├── reflection_vs_RKstep/    # reflection_vs_RKstep_*.csv
│       │   ├── reflection_vs_bandwidth/ # reflection_vs_bandwidth_*.csv
│       │   └── reflection_vs_frequency/ # reflection_vs_frequency_*.csv
│       └── fig/                         # Exported PDF plots, grouped by scan
│           ├── reflection_vs_RKstep/
│           ├── reflection_vs_bandwidth/
│           └── reflection_vs_frequency/
├── 2d_setup/                           # Two-dimensional atomic arrays
│   ├── __init__.py                      # 2D package exports
│   ├── setup.py                         # make_2d_config: prepare geometry, modes, and couplings
│   ├── geometry.py                      # Create, load, and plot atomic geometries
│   ├── bare_param.py                    # Reference spectral kernels and numerical calibration
│   ├── database.py                      # Locate, read, and write calibration CSV files
│   ├── database_geom/                   # Geometry and calibration inputs
│   │   ├── geometries/
│   │   │   └── geometry_id5.csv         # Bundled atomic geometry
│   │   ├── target_profiles/
│   │   │   ├── .gitkeep
│   │   │   └── target_profile_id1.csv   # Bundled reference spectral profile
│   │   └── bare_param/
│   │       └── bare_param_id*.csv       # Calibrated parameter sets (IDs 1–24)
│   ├── notebooks/
│   │   ├── experiment_class_example_2d.ipynb
│   │   │                               # First 2D simulation, fidelity, and density snapshots
│   │   ├── geometry_creation.ipynb      # Preview and save atomic geometries
│   │   ├── generate_target_profile.ipynb
│   │   │                               # Inspect or generate a reference spectral profile
│   │   ├── run_correction.ipynb         # Fit bare parameters for a reduced frequency window
│   │   └── mode_fidelity_vs_freqwindow.ipynb
│   │                                   # Compare calibrated and uncalibrated mode fidelity
│   ├── scripts_experiments/
│   │   └── mode_fidelity_vs_freqwindow.py
│   │                                   # Run mode-fidelity scans and save their time traces
│   └── results/
│       ├── csv_files/
│       │   ├── transmission_vs_frequency/
│       │   │                           # transmission_vs_frequency_*.csv: saved frequency scans
│       │   └── mode_fidelity_vs_freqwindow/
│       │                               # mode_fidelity_vs_freqwindow_*.csv: mode-fidelity traces
│       └── fig/
│           ├── .gitkeep
│           ├── bare_parameters_freqwindows_2d.pdf
│           ├── density_snapshots_2d.pdf
│           ├── final_mode_fidelity_vs_freqwindow_2d.pdf
│           └── timedep_mode_fidelity_vs_freqwindow_2d.pdf
└── timedomain_cross_validation/        # Independent Julia time-domain scattering calculation
    ├── Project.toml                    # Julia project dependencies
    ├── Manifest.toml                   # Resolved Julia dependency versions
    ├── jl_src/
    │   └── reflection.jl               # Reflection scan using WaveguideQED and QuantumOptics
    ├── notebook/
    │   └── reflection_profile_jl_results.ipynb
    │                                   # Plot the saved Julia reflection profiles in Python
    └── results/
        ├── csv/
        │   └── reflection_profile_jl_dt_*.csv
        │                               # Reflection profiles for different time steps
        └── fig/
            └── reflection_profile_jl_result.pdf
```

## How the components fit together

Both Python models prepare an `ExperimentConfig` and pass it to the shared `Experiment` runner in `src/`. The runner propagates the state with RK4, then computes observables. Model-specific geometry, couplings, and calibration live in `waveguide_model/` and `2d_setup/`.

The waveguide model uses analytical bare-parameter calibration and does not require a geometry database. The 2D model uses `database_geom/` to connect an atomic geometry, a reference spectral profile, and fitted bare parameters for a chosen frequency window. Its preparation workflow is:

1. Create or load a geometry with `geometry_creation.ipynb`.
2. Inspect or generate its reference profile with `generate_target_profile.ipynb`.
3. Inspect or fit calibrated parameters with `run_correction.ipynb`.
4. Run the example simulation or the frequency-window scan.

Geometry 5, target profile 1, and their calibrated parameter sets are provided for the example workflow.

The Julia project is separate from the Conda environment. Its `jl_src/reflection.jl` script generates the time-domain reference CSV files, and `notebook/reflection_profile_jl_results.ipynb` visualizes those results.
