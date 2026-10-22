"""2D spectral matching of the target and numerical Green kernels.

The calibration algorithm is independent of the waveguide renormalization.
Couplings in the current fitting/CSV workflow are real.
"""
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import j0
from tqdm.auto import tqdm
from src.xp_config import normalize_cutoffs
from .database import load_target_profile, _atom_array, _metadata_array, _numbers


def make_frequency_grid(omega_values):
    """
    Build the frequency sampling grid used to tabulate F_jk(omega).

    Each omega is an on-shell momentum norm in the paper's 2D continuum model.
    The weights W_omega are uniform by default.
    """

    data = pd.DataFrame({"omega": [float(omega) for omega in omega_values]})
    data["W_omega"] = 1 / len(data)
    return data


def generate_target_profile(
    samples,
    atom_positions,
    Omega_j,
    d_j,
    reference_cutoffs,
    id_geom=None,
    epsabs=1e-8,
    epsrel=1e-8,
    limit=200,
    cache_decimals=12,
    progress=True,
):
    """
    Compute the target Green kernel F_jk(omega) for the physical model.

    For each sampled omega, the function evaluates the effective 2D Green
    matrix of the atoms with physical parameters Omega_A=Omega_j and d_A=d_j,
    then stores F_jk(omega)=d_j G_jk(omega) d_k. This CSV target is what the
    bare parameters must reproduce after the cutoff is imposed.
    """

    atom_positions = np.asarray(atom_positions, dtype=float).reshape(-1, 2)
    Omega_j = _atom_array(Omega_j, len(atom_positions))
    d_j = _atom_array(d_j, len(atom_positions))
    cutoffs = normalize_cutoffs(reference_cutoffs)
    samples = _frequency_dataframe(samples)
    j_indices, k_indices = np.indices((len(atom_positions), len(atom_positions)))
    blocks = []

    iterator = tqdm(samples.itertuples(index=False), total=len(samples), disable=not progress)
    for row in iterator:
        omega = float(row.omega)
        kernel = _spectral_kernel(
            omega,
            atom_positions,
            Omega_j,
            d_j,
            cutoffs,
            epsabs,
            epsrel,
            limit,
            cache_decimals,
        )
        blocks.append(
            pd.DataFrame(
                {
                    "omega": omega,
                    "j": j_indices.reshape(-1),
                    "k": k_indices.reshape(-1),
                    "W": float(row.W_omega) / kernel.size,
                    "F_real": kernel.real.reshape(-1),
                    "F_imag": kernel.imag.reshape(-1),
                }
            )
        )

    profile = pd.concat(blocks, ignore_index=True)
    profile.attrs["metadata"] = {
        "id_geom": id_geom,
        "nb_omega_values": len(samples),
        "Omega_A": _numbers(Omega_j),
        "d_A": _numbers(d_j),
        "reference_omega_min_pi": cutoffs["Omega_min"] / np.pi,
        "reference_omega_max_pi": cutoffs["Omega_max"] / np.pi,
    }
    return profile


def get_2d_bare_parameters(target_profile, atom_positions, cutoffs):
    """
    Fit the bare Hamiltonian parameters Omega_0 and d_0.

    The fit minimizes the weighted difference between the target kernel
    F_jk^phys(omega), generated with Omega_A and d_A, and the cutoff kernel
    F_jk^bare(omega), generated with trial Omega_0 and d_0 on the simulation
    window [ir_cutoff, uv_cutoff].
    """

    profile = load_target_profile(target_profile, cutoffs)
    atom_positions = np.asarray(atom_positions, dtype=float).reshape(-1, 2)
    Omega_A = _metadata_array(profile, "Omega_A", len(atom_positions))
    d_A = _metadata_array(profile, "d_A", len(atom_positions))
    target_F = profile["F_real"].to_numpy() + 1j * profile["F_imag"].to_numpy()
    weights = np.sqrt(profile["W"].to_numpy())
    cutoffs = normalize_cutoffs(cutoffs)

    def residual(parameters):
        Omega_0 = parameters[: len(atom_positions)]
        d_0 = parameters[len(atom_positions) :]
        trial_F = effective_spectral_kernel(profile, atom_positions, Omega_0, d_0, cutoffs)
        error = trial_F - target_F
        return np.concatenate((weights * error.real, weights * error.imag))

    result = least_squares(
        residual,
        np.concatenate((np.clip(Omega_A, cutoffs["Omega_min"], cutoffs["Omega_max"]), d_A)),
        bounds=(
            np.concatenate((np.full(len(atom_positions), cutoffs["Omega_min"]), np.full(len(atom_positions), -np.inf))),
            np.concatenate((np.full(len(atom_positions), cutoffs["Omega_max"]), np.full(len(atom_positions), np.inf))),
        ),
        max_nfev=80,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    return result.x[: len(atom_positions)], result.x[len(atom_positions) :]


def effective_spectral_kernel(profile, atom_positions, Omega_j, d_j, cutoffs):
    """
    Evaluate F_jk(omega) on exactly the rows of a target profile.

    This is the forward map used inside the least-squares fit: given a trial
    Hamiltonian parameterization Omega_j, d_j, it returns the corresponding
    cutoff Green kernel at the same omega,j,k points as the target.
    """

    values = np.empty(len(profile), dtype=complex)
    atom_positions = np.asarray(atom_positions, dtype=float).reshape(-1, 2)
    Omega_j = _atom_array(Omega_j, len(atom_positions))
    d_j = _atom_array(d_j, len(atom_positions))

    for omega, row_index in profile.groupby("omega", sort=False).groups.items():
        indices = np.asarray(row_index, dtype=int)
        rows = profile.iloc[indices]
        kernel = _spectral_kernel(float(omega), atom_positions, Omega_j, d_j, cutoffs)
        values[indices] = kernel[
            rows["j"].to_numpy(dtype=int),
            rows["k"].to_numpy(dtype=int),
        ]

    return values


def two_dimensional_self_energy(
    omega,
    atom_positions,
    d_j,
    cutoffs,
    epsabs=1e-8,
    epsrel=1e-8,
    limit=200,
    cache_decimals=12,
):
    """
    Compute the 2D self-energy matrix Sigma_jk(omega).

    Sigma contains the principal-value real part and the on-shell imaginary
    part induced by the continuum modes between ir_cutoff and uv_cutoff. The
    Bessel factor J_0(omega |r_j-r_k|) is the angular integral of the 2D bath.
    """

    atom_positions = np.asarray(atom_positions, dtype=float).reshape(-1, 2)
    d_j = _atom_array(d_j, len(atom_positions))
    cutoffs = normalize_cutoffs(cutoffs)
    radii = np.linalg.norm(atom_positions[:, None, :] - atom_positions[None, :, :], axis=2)
    radius_keys = np.round(radii.reshape(-1), cache_decimals)

    pv_by_radius = {}
    for key in np.unique(radius_keys):
        radius = float(radii.reshape(-1)[np.flatnonzero(radius_keys == key)[0]])
        pv_by_radius[float(key)] = _pv_bessel_integral(
            float(omega),
            radius,
            cutoffs["Omega_min"],
            cutoffs["Omega_max"],
            epsabs,
            epsrel,
            limit,
        )

    pv = np.asarray([pv_by_radius[float(key)] for key in radius_keys]).reshape(radii.shape)
    d_outer = d_j[:, None] * d_j[None, :]
    return d_outer * pv / (2 * np.pi) - 1j * float(omega) * d_outer * j0(float(omega) * radii) / 2


def _spectral_kernel(
    omega,
    atom_positions,
    Omega_j,
    d_j,
    cutoffs,
    epsabs=1e-8,
    epsrel=1e-8,
    limit=200,
    cache_decimals=12,
):
    """
    Return F_jk(omega)=d_j G_jk(omega) d_k for one frequency.

    Here G(omega) is the dressed atomic Green matrix
    [omega I - Omega - Sigma(omega)]^{-1}.
    """

    sigma = two_dimensional_self_energy(
        omega,
        atom_positions,
        d_j,
        cutoffs,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        cache_decimals=cache_decimals,
    )
    green_inverse = omega * np.eye(len(atom_positions), dtype=complex) - np.diag(Omega_j) - sigma
    green = np.linalg.solve(green_inverse, np.eye(len(atom_positions), dtype=complex))
    return d_j[:, None] * green * d_j[None, :]


def _pv_bessel_integral(omega, radius, ir_cutoff, uv_cutoff, epsabs, epsrel, limit):
    """
    Principal-value radial integral entering Re Sigma_jk(omega).

    The cauchy-weighted branch is used when omega lies inside the cutoff window,
    which is the regime relevant to the fitted target profile.
    """

    def numerator(omega_prime):
        return omega_prime * j0(omega_prime * radius)

    if ir_cutoff < omega < uv_cutoff:
        value, _ = quad(
            numerator,
            ir_cutoff,
            uv_cutoff,
            weight="cauchy",
            wvar=omega,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        return -float(value)

    def integrand(omega_prime):
        return numerator(omega_prime) / (omega - omega_prime)

    value, _ = quad(integrand, ir_cutoff, uv_cutoff, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return float(value)


def _frequency_dataframe(samples):
    if isinstance(samples, pd.DataFrame):
        data = samples.copy()
    else:
        data = make_frequency_grid(samples)
    if "W_omega" not in data:
        data["W_omega"] = 1 / len(data)
    return data[["omega", "W_omega"]].astype(float)
