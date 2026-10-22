"""Populations and mode fidelity are distinct observables in both models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Observables:
    atom_populations: np.ndarray
    atomic_population: np.ndarray
    photon_population: np.ndarray
    mode_fidelity: np.ndarray
    orthogonal_photon_population: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    side: np.ndarray
    norm: np.ndarray


def compute_observables(c, b, config):
    if np.any(config.b_init):
        raise ValueError("Mode fidelity requires initially unexcited atoms.")
    c, b = np.atleast_2d(c), np.atleast_2d(b)
    atoms = np.abs(b)**2
    density = np.abs(c)**2
    photons = np.sum(density, axis=1)
    # Overlap with the incident photon packet in the interaction picture.
    fidelity = np.abs(c @ config.c_init.conj())**2
    forward = density[:, config.forward_mask].sum(axis=1)
    backward = density[:, config.backward_mask].sum(axis=1)
    atomic = atoms.sum(axis=1)
    return Observables(atoms, atomic, photons, fidelity, photons - fidelity,
                       forward, backward, photons - forward - backward, photons + atomic)
