from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import j0
from tqdm.auto import tqdm


def _as_positions(atom_positions: Any) -> np.ndarray:
    positions = np.asarray(atom_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("atom_positions must have shape (n_atoms, 2).")
    return positions


def _atom_parameter_array(value: Any, n_atoms: int, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(n_atoms, array.item(), dtype=dtype)

    array = array.reshape(-1)
    if len(array) != n_atoms:
        raise ValueError(f"{name} must be scalar or contain one value per atom.")
    return array


def _cutoff_bounds(cutoffs: Mapping[str, float]) -> tuple[float, float]:
    if "omega_min" not in cutoffs or "omega_max" not in cutoffs:
        raise ValueError("cutoffs must define omega_min and omega_max.")

    omega_min = float(cutoffs["omega_min"])
    omega_max = float(cutoffs["omega_max"])
    if omega_min < 0:
        raise ValueError("omega_min must be non-negative.")
    if omega_max <= omega_min:
        raise ValueError("omega_max must be larger than omega_min.")
    return omega_min, omega_max


def direction_from_angle(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def on_shell_momentum(angle: float, omega: float) -> np.ndarray:
    return float(omega) * direction_from_angle(angle)


def coupling_vector(
    momentum: Sequence[float],
    atom_positions: np.ndarray,
    d_j: Sequence[float] | float,
) -> np.ndarray:
    positions = _as_positions(atom_positions)
    d_j = _atom_parameter_array(d_j, len(positions), float, "d_j")
    momentum = np.asarray(momentum, dtype=float)
    if momentum.shape != (2,):
        raise ValueError("momentum must have shape (2,).")

    return d_j * np.exp(-1j * (positions @ momentum))


def _pv_bessel_integral(
    omega: float,
    radius: float,
    omega_min: float,
    omega_max: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> float:
    if omega <= 0:
        raise ValueError("omega must be positive.")
    if omega == omega_min or omega == omega_max:
        raise ValueError("omega must not coincide with a cutoff.")

    def numerator(omega_prime: float) -> float:
        return omega_prime * j0(omega_prime * radius)

    if omega_min < omega < omega_max:
        value, _ = quad(
            numerator,
            omega_min,
            omega_max,
            weight="cauchy",
            wvar=omega,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        return -float(value)

    def integrand(omega_prime: float) -> float:
        return numerator(omega_prime) / (omega - omega_prime)

    value, _ = quad(
        integrand,
        omega_min,
        omega_max,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    return float(value)


def two_dimensional_self_energy(
    omega: float,
    atom_positions: np.ndarray,
    d_j: Sequence[float] | float,
    cutoffs: Mapping[str, float],
    *,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 200,
    cache_decimals: int = 12,
) -> np.ndarray:
    positions = _as_positions(atom_positions)
    n_atoms = len(positions)
    d_j = _atom_parameter_array(d_j, n_atoms, float, "d_j")
    omega_min, omega_max = _cutoff_bounds(cutoffs)

    differences = positions[:, None, :] - positions[None, :, :]
    radii = np.linalg.norm(differences, axis=2)
    radius_keys = np.round(radii.reshape(-1), cache_decimals)

    pv_by_radius: dict[float, float] = {}
    flat_radii = radii.reshape(-1)
    for key in np.unique(radius_keys):
        radius = float(flat_radii[np.flatnonzero(radius_keys == key)[0]])
        pv_by_radius[float(key)] = _pv_bessel_integral(
            float(omega),
            radius,
            omega_min,
            omega_max,
            epsabs,
            epsrel,
            limit,
        )

    pv_integrals = np.array(
        [pv_by_radius[float(key)] for key in radius_keys],
        dtype=float,
    ).reshape(radii.shape)

    d_outer = d_j[:, None] * d_j[None, :]
    real_part = d_outer * pv_integrals / (2 * np.pi)
    imaginary_part = -1j * float(omega) * d_outer * j0(float(omega) * radii) / 2
    return real_part + imaginary_part


def two_dimensional_amplitude(
    theta: float,
    phi: float,
    omega: float,
    atom_positions: np.ndarray,
    Omega_j: Sequence[float] | float,
    d_j: Sequence[float] | float,
    cutoffs: Mapping[str, float],
    *,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 200,
    cache_decimals: int = 12,
) -> complex:
    positions = _as_positions(atom_positions)
    n_atoms = len(positions)
    Omega_j = _atom_parameter_array(Omega_j, n_atoms, float, "Omega_j")
    d_j = _atom_parameter_array(d_j, n_atoms, float, "d_j")

    sigma_eff = two_dimensional_self_energy(
        omega,
        positions,
        d_j,
        cutoffs,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        cache_decimals=cache_decimals,
    )
    return _two_dimensional_amplitude_from_self_energy(
        theta,
        phi,
        omega,
        positions,
        Omega_j,
        d_j,
        sigma_eff,
    )


def _two_dimensional_amplitude_from_self_energy(
    theta: float,
    phi: float,
    omega: float,
    atom_positions: np.ndarray,
    Omega_j: np.ndarray,
    d_j: np.ndarray,
    sigma_eff: np.ndarray,
) -> complex:
    positions = _as_positions(atom_positions)
    n_atoms = len(positions)

    p_phi = on_shell_momentum(phi, omega)
    q_theta = on_shell_momentum(theta, omega)
    g_p = coupling_vector(p_phi, positions, d_j)
    g_q = coupling_vector(q_theta, positions, d_j)

    green_matrix_inverse = (
        float(omega) * np.eye(n_atoms, dtype=complex)
        - np.diag(Omega_j)
        - sigma_eff
    )
    green_times_g_p = np.linalg.solve(green_matrix_inverse, g_p)
    return -1j * np.vdot(g_q, green_times_g_p)


def make_sampling_grid(
    theta_values: Sequence[float],
    phi_values: Sequence[float],
    omega_values: Sequence[float],
) -> pd.DataFrame:
    rows = [
        {
            "theta": float(theta),
            "phi": float(phi),
            "omega": float(omega),
        }
        for theta in theta_values
        for phi in phi_values
        for omega in omega_values
    ]
    data = pd.DataFrame(rows)
    data["W"] = 1 / len(data)
    return data


def make_frequency_grid(omega_values: Sequence[float]) -> pd.DataFrame:
    data = pd.DataFrame({"omega": [float(omega) for omega in omega_values]})
    if len(data) == 0:
        raise ValueError("omega_values must contain at least one frequency.")

    data["W_omega"] = 1 / len(data)
    return data


def _uniform_reference_value(values: np.ndarray, name: str) -> float:
    if not np.allclose(values, values[0]):
        raise ValueError(f"{name} must be uniform to store it in the target profile.")
    return float(values[0])


def _samples_dataframe(samples: pd.DataFrame | Sequence[Sequence[float]]) -> pd.DataFrame:
    if isinstance(samples, pd.DataFrame):
        data = samples.copy()
    else:
        array = np.asarray(samples, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("samples must have columns theta, phi and omega.")
        data = pd.DataFrame(array, columns=["theta", "phi", "omega"])

    required = {"theta", "phi", "omega"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"samples is missing columns: {sorted(missing)}")

    if "W" not in data.columns:
        data["W"] = 1 / len(data)

    return data[["theta", "phi", "omega", "W"]]


def _frequency_samples_dataframe(
    samples: pd.DataFrame | Sequence[float] | Sequence[Sequence[float]],
) -> pd.DataFrame:
    if isinstance(samples, pd.DataFrame):
        data = samples.copy()
    else:
        array = np.asarray(samples, dtype=float)
        if array.ndim == 1:
            data = pd.DataFrame({"omega": array})
        elif array.ndim == 2 and array.shape[1] == 1:
            data = pd.DataFrame({"omega": array[:, 0]})
        else:
            raise ValueError("samples must be a DataFrame with omega or a 1D list of frequencies.")

    if len(data) == 0:
        raise ValueError("samples must contain at least one frequency.")
    if "omega" not in data.columns:
        raise ValueError("samples is missing column: omega")

    if "W_omega" not in data.columns:
        if "W" in data.columns:
            data["W_omega"] = data["W"]
        else:
            data["W_omega"] = 1 / len(data)

    data = data[["omega", "W_omega"]].astype(float)
    if np.any(~np.isfinite(data["omega"].to_numpy())):
        raise ValueError("omega samples must be finite.")
    if np.any(data["omega"].to_numpy() <= 0):
        raise ValueError("omega samples must be positive.")
    if np.any(data["W_omega"].to_numpy() < 0):
        raise ValueError("frequency weights must be non-negative.")
    return data


def _spectral_kernel_from_self_energy(
    omega: float,
    atom_positions: np.ndarray,
    Omega_j: np.ndarray,
    d_j: np.ndarray,
    sigma_eff: np.ndarray,
) -> np.ndarray:
    positions = _as_positions(atom_positions)
    n_atoms = len(positions)
    green_matrix_inverse = (
        float(omega) * np.eye(n_atoms, dtype=complex)
        - np.diag(Omega_j)
        - sigma_eff
    )
    green_matrix = np.linalg.solve(green_matrix_inverse, np.eye(n_atoms, dtype=complex))
    return d_j[:, None] * green_matrix * d_j[None, :]


def generate_target_profile(
    samples: pd.DataFrame | Sequence[float] | Sequence[Sequence[float]],
    atom_positions: np.ndarray,
    Omega_j: Sequence[float] | float,
    d_j: Sequence[float] | float,
    reference_cutoffs: Mapping[str, float],
    *,
    epsabs: float = 1e-8,
    epsrel: float = 1e-8,
    limit: int = 200,
    cache_decimals: int = 12,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Generate the spectral target profile F_{l,jk}=d_j G_{jk}(omega_l)d_k.

    The output contains one row per frequency and atomic matrix component.
    Since the code now assumes real d_j, the mathematical factor d_j^* is
    represented by the same real number d_j.
    """

    positions = _as_positions(atom_positions)
    n_atoms = len(positions)
    Omega_j = _atom_parameter_array(Omega_j, n_atoms, float, "Omega_j")
    d_j = _atom_parameter_array(d_j, n_atoms, float, "d_j")
    omega_min, omega_max = _cutoff_bounds(reference_cutoffs)

    data = _frequency_samples_dataframe(samples).reset_index(drop=True)
    profile_blocks = []
    sigma_by_omega: dict[float, np.ndarray] = {}
    j_indices, k_indices = np.indices((n_atoms, n_atoms))
    flat_j = j_indices.reshape(-1)
    flat_k = k_indices.reshape(-1)

    iterator = tqdm(
        data.itertuples(index=False),
        total=len(data),
        disable=not progress,
        desc="target spectral profile",
        unit="omega",
    )
    for row in iterator:
        omega = float(row.omega)
        if omega not in sigma_by_omega:
            sigma_by_omega[omega] = two_dimensional_self_energy(
                omega,
                positions,
                d_j,
                {"omega_min": omega_min, "omega_max": omega_max},
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
                cache_decimals=cache_decimals,
            )

        spectral_kernel = _spectral_kernel_from_self_energy(
            omega,
            positions,
            Omega_j,
            d_j,
            sigma_by_omega[omega],
        )
        component_weight = float(row.W_omega) / (n_atoms * n_atoms)
        profile_blocks.append(
            pd.DataFrame(
                {
                    "omega": omega,
                    "j": flat_j,
                    "k": flat_k,
                    "W": component_weight,
                    "F_real": spectral_kernel.real.reshape(-1),
                    "F_imag": spectral_kernel.imag.reshape(-1),
                }
            )
        )

    profile = pd.concat(profile_blocks, ignore_index=True)
    profile["reference_Omega"] = _uniform_reference_value(Omega_j, "Omega_j")
    profile["reference_d"] = _uniform_reference_value(d_j, "d_j")
    return profile[
        [
            "omega",
            "j",
            "k",
            "W",
            "F_real",
            "F_imag",
            "reference_Omega",
            "reference_d",
        ]
    ]


def save_target_profile_csv(profile: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path, index=False)
    return output_path
