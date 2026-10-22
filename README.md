# Single-photon scattering

Moteur de propagation commun aux modèles guide d’onde et réseau 2D du papier dans `theory_tex/`.

```bash
conda env create -f environment.yml
conda activate single-photon-renorm
jupyter lab
```

Commencer par [l’exemple guide d’onde](waveguide_model/notebooks/experiment_class_example.ipynb) ou [l’exemple 2D](2d_setup/notebooks/experiment_class_example_2d.ipynb), puis exécuter les cellules dans l’ordre.

Le [guide PDF](guide_utilisation.pdf) décrit l’API, les notations, la calibration et les vérifications des figures.

- `src/` : configuration, modes, RK4, expérience et observables communs.
- `waveguide_model/` : préparation du guide et calibration analytique ; notebooks et résultats.
- `2d_setup/` : préparation 2D, calibration numérique et base de géométries ; notebooks et résultats.
- `timedomain_cross_validation/` : comparaison indépendante en Julia.
- `theory_tex/` : manuscrit.

Les notebooks de résultats lisent les CSV fournis. `RUN_SIMULATIONS=True` les recalcule ; `SAVE_FIGURES=True` exporte les PDF. LaTeX n’est pas nécessaire (`USE_TEX=False`).
