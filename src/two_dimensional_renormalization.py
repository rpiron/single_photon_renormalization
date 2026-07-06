from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm.auto import tqdm

from src.target_profile import two_dimensional_self_energy


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


def _load_target_profile(target_profile: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(target_profile, pd.DataFrame):
        data = target_profile.copy()
    else:
        data = pd.read_csv(Path(target_profile))

    if "W" not in data.columns:
        data["W"] = 1 / len(data)

    required = {
        "omega",
        "j",
        "k",
        "F_real",
        "F_imag",
        "W",
        "reference_Omega",
        "reference_d",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"target_profile is missing columns: {sorted(missing)}")

    return data.reset_index(drop=True)


def _reference_parameters(
    profile: pd.DataFrame | None,
    n_atoms: int,
    reference_Omega_j: Sequence[float] | float | None,
    reference_d_j: Sequence[float] | float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if reference_Omega_j is None:
        if profile is None:
            raise ValueError(
                "reference_Omega_j must be provided when target_profile is None."
            )
        reference_Omega_j = float(profile["reference_Omega"].iloc[0])

    if reference_d_j is None:
        if profile is None:
            raise ValueError(
                "reference_d_j must be provided when target_profile is None."
            )
        reference_d_j = float(profile["reference_d"].iloc[0])

    Omega_j = _atom_parameter_array(reference_Omega_j, n_atoms, float, "reference_Omega_j")
    d_j = _atom_parameter_array(reference_d_j, n_atoms, float, "reference_d_j")
    return Omega_j, d_j


def _select_fit_profile(
    profile: pd.DataFrame,
    max_samples: int | None,
    random_state: int | None,
) -> pd.DataFrame:
    if max_samples is None or len(profile) <= max_samples:
        return profile.reset_index(drop=True)

    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    if random_state is None:
        indices = np.linspace(0, len(profile) - 1, max_samples, dtype=int)
    else:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(len(profile), size=max_samples, replace=False))
    return profile.iloc[np.unique(indices)].reset_index(drop=True)


def _spectral_residual_weights(
    profile: pd.DataFrame,
    target_values: np.ndarray,
    *,
    relative_loss: bool,
    relative_loss_epsilon: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    row_weights = np.maximum(profile["W"].to_numpy(dtype=float), 0.0)
    target_norm = float(np.sum(row_weights * np.abs(target_values) ** 2))

    if not relative_loss:
        weights = np.sqrt(row_weights)
        return weights, {
            "relative_loss": False,
            "target_weighted_norm": target_norm,
            "zero_model_l2": target_norm,
        }

    if relative_loss_epsilon < 0:
        raise ValueError("relative_loss_epsilon must be non-negative.")

    weights = np.empty(len(profile), dtype=float)
    frequency_denominators: dict[float, float] = {}
    frequency_weights: dict[float, float] = {}

    for omega, row_index in profile.groupby("omega", sort=False).groups.items():
        indices = np.asarray(row_index, dtype=int)
        frequency_weight = float(np.sum(row_weights[indices]))
        denominator = float(
            np.sum(np.abs(target_values[indices]) ** 2) + relative_loss_epsilon
        )
        weights[indices] = np.sqrt(frequency_weight / denominator)
        frequency_denominators[float(omega)] = denominator
        frequency_weights[float(omega)] = frequency_weight

    zero_model_l2 = float(np.sum((weights**2) * np.abs(target_values) ** 2))
    return weights, {
        "relative_loss": True,
        "target_weighted_norm": target_norm,
        "zero_model_l2": zero_model_l2,
        "relative_loss_epsilon": float(relative_loss_epsilon),
        "min_frequency_target_norm": float(min(frequency_denominators.values())),
        "max_frequency_target_norm": float(max(frequency_denominators.values())),
        "min_frequency_weight": float(min(frequency_weights.values())),
        "max_frequency_weight": float(max(frequency_weights.values())),
    }


def _profile_spectral_kernel(
    profile: pd.DataFrame,
    atom_positions: np.ndarray,
    Omega_j: np.ndarray,
    d_j: np.ndarray,
    cutoffs: Mapping[str, float],
    *,
    epsabs: float,
    epsrel: float,
    limit: int,
    cache_decimals: int,
) -> np.ndarray:
    values = np.empty(len(profile), dtype=complex)
    n_atoms = len(atom_positions)

    for omega, row_index in profile.groupby("omega", sort=False).groups.items():
        indices = np.asarray(row_index, dtype=int)
        rows = profile.iloc[indices]
        j_indices = rows["j"].to_numpy(dtype=int)
        k_indices = rows["k"].to_numpy(dtype=int)
        if (
            np.any(j_indices < 0)
            or np.any(j_indices >= n_atoms)
            or np.any(k_indices < 0)
            or np.any(k_indices >= n_atoms)
        ):
            raise ValueError("target_profile contains j or k outside the atomic range.")

        sigma_eff = two_dimensional_self_energy(
            float(omega),
            atom_positions,
            d_j,
            cutoffs,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            cache_decimals=cache_decimals,
        )
        green_matrix_inverse = (
            float(omega) * np.eye(n_atoms, dtype=complex)
            - np.diag(Omega_j)
            - sigma_eff
        )
        green_matrix = np.linalg.solve(
            green_matrix_inverse,
            np.eye(n_atoms, dtype=complex),
        )
        spectral_kernel = d_j[:, None] * green_matrix * d_j[None, :]
        values[indices] = spectral_kernel[j_indices, k_indices]

    return values


def get_2d_bare_parameters(
    target_profile: pd.DataFrame | str | Path | None,
    atom_positions: np.ndarray,
    cutoffs: Mapping[str, float],
    *,
    correction: bool = False,
    reference_Omega_j: Sequence[float] | float | None = None,
    reference_d_j: Sequence[float] | float | None = None,
    fit_mode: str = "uniform",
    max_samples: int | None = None,
    random_state: int | None = None,
    epsabs: float = 1e-7,
    epsrel: float = 1e-7,
    limit: int = 200,
    cache_decimals: int = 12,
    optimizer_options: Mapping[str, Any] | None = None,
    relative_loss: bool = False,
    relative_loss_epsilon: float = 1e-15,
    include_regul: bool = False,
    regul_weight: float = 1.0,
    regul_omega_weight: float | None = None,
    regul_d_weight: float | None = None,
    regul_omega_scale: float | None = None,
    regul_d_scale: float | None = None,
    d_bounds: tuple[float | None, float | None] | None = None,
    progress: bool = False,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Return 2D bare parameters Omega_j and real d_j for a desired frequency window.

    If correction is False, reference_Omega_j and reference_d_j are returned
    unchanged and target_profile can be None. If correction is True, a
    least-squares L2 fit minimizes the difference between the target spectral
    kernel F_{l,jk} and the analytical 2D kernel computed with the requested
    cutoffs, so target_profile is required. Set relative_loss to normalize the
    spectral residual frequency by frequency. Set include_regul to add a
    quadratic pull toward the reference Omega_j and d_j. Use d_bounds to
    constrain the real couplings, for example d_bounds=(0.0, None).
    """

    positions = _as_positions(atom_positions)
    n_atoms = len(positions)
    omega_min, omega_max = _cutoff_bounds(cutoffs)
    profile = None if target_profile is None else _load_target_profile(target_profile)

    if correction and profile is None:
        raise ValueError("target_profile is required when correction=True.")

    reference_Omega_j, reference_d_j = _reference_parameters(
        profile,
        n_atoms,
        reference_Omega_j,
        reference_d_j,
    )

    if not correction:
        diagnostics = {
            "correction": False,
            "fit_mode": fit_mode,
            "n_atoms": n_atoms,
            "n_profile_points": 0 if profile is None else len(profile),
            "n_fit_points": 0,
            "success": True,
            "message": "Corrections disabled; reference parameters returned.",
        }
        if return_diagnostics:
            return reference_Omega_j, reference_d_j, diagnostics
        return reference_Omega_j, reference_d_j

    if fit_mode not in {"uniform", "per_atom"}:
        raise ValueError("fit_mode must be 'uniform' or 'per_atom'.")

    fit_profile = _select_fit_profile(profile, max_samples, random_state)
    target_values = (
        fit_profile["F_real"].to_numpy(dtype=float)
        + 1j * fit_profile["F_imag"].to_numpy(dtype=float)
    )
    weights, loss_metadata = _spectral_residual_weights(
        fit_profile,
        target_values,
        relative_loss=relative_loss,
        relative_loss_epsilon=relative_loss_epsilon,
    )

    if regul_weight < 0:
        raise ValueError("regul_weight must be non-negative.")
    if regul_omega_weight is None:
        regul_omega_weight = regul_weight
    if regul_d_weight is None:
        regul_d_weight = regul_weight
    if regul_omega_weight < 0 or regul_d_weight < 0:
        raise ValueError("regularization weights must be non-negative.")
    if regul_omega_scale is None:
        regul_omega_scale = max(float(np.mean(np.abs(reference_Omega_j))), 1.0)
    if regul_d_scale is None:
        regul_d_scale = max(float(np.mean(np.abs(reference_d_j))), 1.0)
    if regul_omega_scale <= 0 or regul_d_scale <= 0:
        raise ValueError("regularization scales must be positive.")

    d_lower = -np.inf
    d_upper = np.inf
    if d_bounds is not None:
        if len(d_bounds) != 2:
            raise ValueError("d_bounds must contain lower and upper bounds.")
        d_lower_raw, d_upper_raw = d_bounds
        d_lower = -np.inf if d_lower_raw is None else float(d_lower_raw)
        d_upper = np.inf if d_upper_raw is None else float(d_upper_raw)
        if d_lower > d_upper:
            raise ValueError("d_bounds lower bound must not exceed upper bound.")

    width = omega_max - omega_min
    omega_eps = max(1e-10, 1e-8 * width)
    omega_lower = omega_min + omega_eps
    omega_upper = omega_max - omega_eps
    if omega_lower >= omega_upper:
        raise ValueError("cutoffs leave no room for an Omega_j initial bound.")

    if fit_mode == "uniform":
        x0 = np.array(
            [
                np.clip(float(np.mean(reference_Omega_j)), omega_lower, omega_upper),
                np.clip(float(np.mean(reference_d_j)), d_lower, d_upper),
            ],
            dtype=float,
        )
        lower_bounds = np.array([omega_lower, d_lower], dtype=float)
        upper_bounds = np.array([omega_upper, d_upper], dtype=float)

        def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            Omega_j = np.full(n_atoms, float(x[0]), dtype=float)
            d_j = np.full(n_atoms, float(x[1]), dtype=float)
            return Omega_j, d_j

    else:
        x0 = np.concatenate(
            (
                np.clip(reference_Omega_j.astype(float), omega_lower, omega_upper),
                np.clip(reference_d_j.astype(float), d_lower, d_upper),
            )
        )
        lower_bounds = np.concatenate(
            (
                np.full(n_atoms, omega_lower, dtype=float),
                np.full(n_atoms, d_lower, dtype=float),
            )
        )
        upper_bounds = np.concatenate(
            (
                np.full(n_atoms, omega_upper, dtype=float),
                np.full(n_atoms, d_upper, dtype=float),
            )
        )

        def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            Omega_j = x[:n_atoms].astype(float)
            d_j = x[n_atoms:].astype(float)
            return Omega_j, d_j

    residual_evaluations = 0
    best_l2 = np.inf
    last_data_l2 = np.inf
    last_regul_l2 = 0.0

    progress_bar = tqdm(
        desc="calibration",
        disable=not progress,
        unit="eval",
        total=None,
    )

    def regularization_residual(Omega_j: np.ndarray, d_j: np.ndarray) -> np.ndarray:
        if not include_regul:
            return np.empty(0, dtype=float)

        residual_blocks = []
        if regul_omega_weight > 0:
            omega_residual = (
                np.sqrt(regul_omega_weight / n_atoms)
                * (Omega_j - reference_Omega_j)
                / regul_omega_scale
            )
            residual_blocks.append(omega_residual)
        if regul_d_weight > 0:
            d_residual = (
                np.sqrt(regul_d_weight / n_atoms)
                * (d_j - reference_d_j)
                / regul_d_scale
            )
            residual_blocks.append(d_residual)

        if not residual_blocks:
            return np.empty(0, dtype=float)
        return np.concatenate(residual_blocks).astype(float)

    def residual(x: np.ndarray) -> np.ndarray:
        nonlocal residual_evaluations, best_l2, last_data_l2, last_regul_l2

        Omega_j, d_j = unpack(x)
        try:
            spectral_values = _profile_spectral_kernel(
                fit_profile,
                positions,
                Omega_j,
                d_j,
                {"omega_min": omega_min, "omega_max": omega_max},
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
                cache_decimals=cache_decimals,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            data_residual_values = np.full(2 * len(fit_profile), 1e12, dtype=float)
        else:
            difference = spectral_values - target_values
            data_residual_values = np.concatenate(
                (weights * difference.real, weights * difference.imag)
            )

        regul_residual_values = regularization_residual(Omega_j, d_j)
        residual_values = np.concatenate((data_residual_values, regul_residual_values))

        residual_evaluations += 1
        last_data_l2 = float(np.dot(data_residual_values, data_residual_values))
        last_regul_l2 = float(np.dot(regul_residual_values, regul_residual_values))
        current_l2 = last_data_l2 + last_regul_l2
        best_l2 = min(best_l2, current_l2)
        if progress:
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "loss": f"{current_l2:.3e}",
                    "data": f"{last_data_l2:.3e}",
                    "regul": f"{last_regul_l2:.3e}",
                    "best": f"{best_l2:.3e}",
                    "Omega/pi": f"{np.mean(Omega_j) / np.pi:.5g}",
                    "d": f"{np.mean(d_j):.5g}",
                }
            )

        return residual_values

    try:
        initial_residual = residual(x0)
        initial_data_l2 = last_data_l2
        initial_regul_l2 = last_regul_l2
        options = {
            "max_nfev": 80,
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
        }
        if optimizer_options is not None:
            options.update(dict(optimizer_options))

        result = least_squares(
            residual,
            x0,
            bounds=(lower_bounds, upper_bounds),
            **options,
        )
        Omega_j, d_j = unpack(result.x)
        final_residual = residual(result.x)
        final_data_l2 = last_data_l2
        final_regul_l2 = last_regul_l2
    finally:
        progress_bar.close()

    diagnostics = {
        "correction": True,
        "fit_mode": fit_mode,
        "n_atoms": n_atoms,
        "n_profile_points": len(profile),
        "n_fit_points": len(fit_profile),
        "relative_loss": bool(relative_loss),
        "include_regul": bool(include_regul),
        "regul_weight": float(regul_weight),
        "regul_omega_weight": float(regul_omega_weight),
        "regul_d_weight": float(regul_d_weight),
        "regul_omega_scale": float(regul_omega_scale),
        "regul_d_scale": float(regul_d_scale),
        "d_lower_bound": float(d_lower),
        "d_upper_bound": float(d_upper),
        **loss_metadata,
        "cache_decimals": int(cache_decimals),
        "omega_min": omega_min,
        "omega_max": omega_max,
        "initial_l2": float(np.dot(initial_residual, initial_residual)),
        "final_l2": float(np.dot(final_residual, final_residual)),
        "initial_data_l2": float(initial_data_l2),
        "final_data_l2": float(final_data_l2),
        "initial_regul_l2": float(initial_regul_l2),
        "final_regul_l2": float(final_regul_l2),
        "best_l2": float(best_l2),
        "residual_evaluations": int(residual_evaluations),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
    }

    if return_diagnostics:
        return Omega_j, d_j, diagnostics
    return Omega_j, d_j


get_bare_parameters = get_2d_bare_parameters
