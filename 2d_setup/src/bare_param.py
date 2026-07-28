import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import j0
from tqdm.auto import tqdm

from .xp_config import ExperimentConfig, normalize_cutoffs


"""
Cutoff correction tools for the 2D model.

This file plays the same role as waveguide_model/src/bare_param.py: it is the
place where the physical parameters used in the paper, Omega_A and d_A, are
converted into the bare Hamiltonian parameters, Omega_0 and d_0. In 2D this
cannot be written as a simple closed formula, so the correction is obtained by
matching the component-resolved target Green kernel F_jk(omega).
"""


DATABASE_DIR = Path(__file__).resolve().parents[1] / "database_fiber"
BARE_PARAM_DIR = DATABASE_DIR / "bare_param"
TARGET_PROFILE_DIR = DATABASE_DIR / "target_profiles"


# ---------------------------------------------------------------------------
# CSV paths and metadata
# ---------------------------------------------------------------------------


def bare_param_path(bare_param_id=None, directory=BARE_PARAM_DIR):
    """
    Return the CSV path storing one corrected pair (Omega_0, d_0).

    These files are the 2D analogue of the bare parameters returned directly by
    get_bare_param_n in the waveguide model. Each file is indexed by an id in
    database_fiber/bare_param.
    """

    return _database_path(directory, "bare_param_id", bare_param_id)


def target_profile_path(target_profile_id=None, directory=TARGET_PROFILE_DIR):
    """
    Return the CSV path storing one target profile F_jk(omega).

    The target profile is computed from the physical model with parameters
    Omega_A and d_A. It is later used as the reference object to fit Omega_0 and
    d_0 in the cutoff Hamiltonian.
    """

    return _database_path(directory, "target_profile_id", target_profile_id)


def read_bare_parameters_metadata(path):
    """
    Read the header of a bare-parameter CSV.

    The header records which physical target (geometry id, target-profile id,
    Omega_A, d_A, and cutoff window) produced the stored Omega_0 and d_0.
    """

    raw = _read_metadata(path)
    return {
        "id_geom": int(raw["id_geom"]) if "id_geom" in raw else None,
        "id_target_profile": int(raw["id_target_profile"]) if "id_target_profile" in raw else None,
        "Omega_A": _array(raw["Omega_A"]),
        "d_A": _array(_metadata_value(raw, "d_A", "g_A")),
        "ir_cutoff": _metadata_cutoff(raw, "omega_min"),
        "uv_cutoff": _metadata_cutoff(raw, "omega_max"),
    }


def read_target_profile_metadata(path):
    """
    Read the header of a target-profile CSV.

    The metadata identifies the physical 2D scattering problem used to build
    F_jk(omega): geometry, Omega_A, d_A, and the reference cutoff window.
    """

    raw = _read_metadata(path)
    raw_id = Path(path).stem.removeprefix("target_profile_id")
    return {
        "id_target_profile": int(raw_id) if raw_id.isdigit() else None,
        "id_geom": int(raw["id_geom"]) if "id_geom" in raw else None,
        "nb_omega_values": int(raw["nb_omega_values"]) if "nb_omega_values" in raw else None,
        "Omega_A": _array(raw["Omega_A"]),
        "d_A": _array(_metadata_value(raw, "d_A", "g_A")),
        "reference_omega_min": _metadata_cutoff(raw, "reference_omega_min"),
        "reference_omega_max": _metadata_cutoff(raw, "reference_omega_max"),
        "reference_ir_cutoff": _metadata_cutoff(raw, "reference_omega_min"),
        "reference_uv_cutoff": _metadata_cutoff(raw, "reference_omega_max"),
    }


def load_bare_parameters_csv(path):
    """
    Load the bare Hamiltonian arrays Omega_0 and d_0 from disk.

    These are the parameters inserted in ExperimentConfig when the 2D simulation
    is run with correction=True.
    """

    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    return (
        np.asarray([float(row["Omega_0"]) for row in rows]),
        np.asarray([float(_row_value(row, "d_0", "g_0")) for row in rows]),
    )


def save_bare_parameters_csv(Omega_0, d_0, target_metadata, cutoffs, path):
    """
    Save the fitted bare arrays Omega_0 and d_0.

    The CSV keeps the physical parameters in the header and only stores the bare
    atom-by-atom values in the table. This makes each bare_param_id traceable to
    the target profile and cutoff window used in the paper's correction step.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cutoffs = normalize_cutoffs(cutoffs)

    with path.open("w", newline="") as stream:
        if target_metadata.get("id_geom") is not None:
            stream.write(f"# id_geom: {int(target_metadata['id_geom'])}\n")
        if target_metadata.get("id_target_profile") is not None:
            stream.write(f"# id_target_profile: {int(target_metadata['id_target_profile'])}\n")
        stream.write(f"# Omega_A: {_numbers(target_metadata['Omega_A'])}\n")
        stream.write(f"# d_A: {_numbers(_metadata_value(target_metadata, 'd_A', 'g_A'))}\n")
        stream.write(f"# reference_omega_min_pi: {_cutoff_pi(target_metadata, 'reference_omega_min'):.16g}\n")
        stream.write(f"# reference_omega_max_pi: {_cutoff_pi(target_metadata, 'reference_omega_max'):.16g}\n")
        stream.write(f"# omega_min_pi: {cutoffs['ir_cutoff'] / np.pi:.16g}\n")
        stream.write(f"# omega_max_pi: {cutoffs['uv_cutoff'] / np.pi:.16g}\n")
        writer = csv.writer(stream)
        writer.writerow(["j", "Omega_0", "d_0"])
        for j, (omega, d_value) in enumerate(zip(Omega_0, d_0)):
            writer.writerow([j, f"{omega:.16g}", f"{d_value:.16g}"])
    return path


# ---------------------------------------------------------------------------
# Experiment config: physical parameters in, bare parameters out
# ---------------------------------------------------------------------------


def make_fiber_config(param_photon, param_atom, param_time_evol, cutoffs, correction=False):
    """
    Build the ExperimentConfig used by the time-domain 2D simulation.

    If correction=False, the Hamiltonian is parameterized directly by the
    physical values Omega_A and d_A. If correction=True, the function replaces
    them by the fitted bare values Omega_0 and d_0 associated with the requested
    geometry, target profile, and cutoff window.
    """

    geometry = param_atom["geometry"]
    atom_positions = np.asarray(geometry["atom_positions"], dtype=float)
    Omega_A = _atom_array(param_atom["Omega_A"], len(atom_positions))
    d_A = _atom_array(_metadata_value(param_atom, "d_A", "g_A"), len(atom_positions))

    if correction:
        path = find_bare_param_path(
            Omega_A,
            d_A,
            cutoffs,
            id_geom=geometry.get("id_geom"),
            id_target_profile=param_atom.get("id_target_profile"),
        )
        Omega_0, d_0 = load_bare_parameters_csv(path)
    else:
        Omega_0, d_0 = Omega_A, d_A

    return ExperimentConfig(
        lengths=geometry["lengths"],
        atom_positions=atom_positions,
        atom_frequencies=Omega_0,
        atom_couplings=d_0,
        photon=param_photon,
        time=param_time_evol,
        cutoffs=cutoffs,
    )


def find_bare_param_path(
    Omega_A,
    d_A,
    cutoffs,
    id_geom=None,
    id_target_profile=None,
    directory=BARE_PARAM_DIR,
):
    """
    Locate the CSV containing the bare parameters for this physical setup.

    The match is made against the physical data that define the correction:
    Omega_A, d_A, the cutoff window [ir_cutoff, uv_cutoff], and optionally the
    geometry and target-profile ids. This is the only database lookup needed by
    make_fiber_config(correction=True).
    """

    cutoffs = normalize_cutoffs(cutoffs)
    Omega_A = np.asarray(Omega_A, dtype=float).reshape(-1)
    d_A = np.asarray(d_A, dtype=float).reshape(-1)

    for path in sorted(Path(directory).glob("bare_param_id*.csv")):
        metadata = read_bare_parameters_metadata(path)
        if id_geom is not None and metadata["id_geom"] != int(id_geom):
            continue
        if id_target_profile is not None and metadata["id_target_profile"] != int(id_target_profile):
            continue
        if (
            np.allclose(metadata["Omega_A"], Omega_A)
            and np.allclose(metadata["d_A"], d_A)
            and np.isclose(metadata["ir_cutoff"], cutoffs["ir_cutoff"])
            and np.isclose(metadata["uv_cutoff"], cutoffs["uv_cutoff"])
        ):
            return path

    raise FileNotFoundError("Requested bare parameters do not exist in database_fiber.")


# ---------------------------------------------------------------------------
# Target profile: physical two-dimensional Green kernel to match
# ---------------------------------------------------------------------------


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
        "reference_omega_min_pi": cutoffs["ir_cutoff"] / np.pi,
        "reference_omega_max_pi": cutoffs["uv_cutoff"] / np.pi,
    }
    return profile


def save_target_profile_csv(profile, path):
    """
    Save the tabulated target kernel F_jk(omega).

    The rows contain omega, component indices j,k, quadrature weight W, and the
    real/imaginary parts of F_jk. The header records the physical parameters
    Omega_A and d_A that define the target model.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = profile.attrs["metadata"]

    with path.open("w", newline="") as stream:
        if metadata.get("id_geom") is not None:
            stream.write(f"# id_geom: {int(metadata['id_geom'])}\n")
        stream.write(f"# nb_omega_values: {int(metadata['nb_omega_values'])}\n")
        stream.write(f"# Omega_A: {metadata['Omega_A']}\n")
        stream.write(f"# d_A: {_metadata_value(metadata, 'd_A', 'g_A')}\n")
        stream.write(f"# reference_omega_min_pi: {float(metadata['reference_omega_min_pi']):.16g}\n")
        stream.write(f"# reference_omega_max_pi: {float(metadata['reference_omega_max_pi']):.16g}\n")
        profile.to_csv(stream, index=False)
    return path


# ---------------------------------------------------------------------------
# Bare-parameter fit: find Omega_0, d_0 from the target profile
# ---------------------------------------------------------------------------


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
        np.concatenate((np.clip(Omega_A, cutoffs["ir_cutoff"], cutoffs["uv_cutoff"]), d_A)),
        bounds=(
            np.concatenate((np.full(len(atom_positions), cutoffs["ir_cutoff"]), np.full(len(atom_positions), -np.inf))),
            np.concatenate((np.full(len(atom_positions), cutoffs["uv_cutoff"]), np.full(len(atom_positions), np.inf))),
        ),
        max_nfev=80,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    return result.x[: len(atom_positions)], result.x[len(atom_positions) :]


def load_target_profile(target_profile, cutoffs):
    """
    Load the target profile on the active simulation window.

    The paper's correction only compares frequencies retained by the numerical
    Hamiltonian, so the stored F_jk(omega) table is restricted to
    ir_cutoff < omega < uv_cutoff and its weights are renormalized.
    """

    if isinstance(target_profile, pd.DataFrame):
        profile = target_profile.copy()
    else:
        path = Path(target_profile)
        if not path.exists():
            raise FileNotFoundError(f"{path.name} does not exist in database_fiber.")
        profile = pd.read_csv(path, comment="#")
        profile.attrs["metadata"] = read_target_profile_metadata(path)

    cutoffs = normalize_cutoffs(cutoffs)
    inside = (profile["omega"] > cutoffs["ir_cutoff"]) & (
        profile["omega"] < cutoffs["uv_cutoff"]
    )
    profile = profile.loc[inside, ["omega", "j", "k", "W", "F_real", "F_imag"]]
    profile = profile.reset_index(drop=True)
    profile["W"] = profile["W"] / float(profile["W"].sum())
    return profile


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
            cutoffs["ir_cutoff"],
            cutoffs["uv_cutoff"],
            epsabs,
            epsrel,
            limit,
        )

    pv = np.asarray([pv_by_radius[float(key)] for key in radius_keys]).reshape(radii.shape)
    d_outer = d_j[:, None] * d_j[None, :]
    return d_outer * pv / (2 * np.pi) - 1j * float(omega) * d_outer * j0(float(omega) * radii) / 2


get_bare_parameters = get_2d_bare_parameters


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


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


def _database_path(directory, prefix, item_id):
    directory = Path(directory)
    if item_id is None:
        ids = [
            int(path.stem.removeprefix(prefix))
            for path in directory.glob(f"{prefix}*.csv")
            if path.stem.removeprefix(prefix).isdigit()
        ]
        item_id = max(ids, default=0) + 1
    return directory / f"{prefix}{int(item_id)}.csv"


def _frequency_dataframe(samples):
    if isinstance(samples, pd.DataFrame):
        data = samples.copy()
    else:
        data = make_frequency_grid(samples)
    if "W_omega" not in data:
        data["W_omega"] = 1 / len(data)
    return data[["omega", "W_omega"]].astype(float)


def _read_metadata(path):
    metadata = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            key, value = line[1:].strip().split(":", maxsplit=1)
            metadata[key.strip()] = value.strip()
    return metadata


def _array(value):
    return np.asarray([float(part) for part in str(value).split(";") if part])


def _atom_array(value, n_atoms):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n_atoms, array.item())
    return array.reshape(-1)


def _metadata_array(profile, key, n_atoms):
    values = _metadata_value(profile.attrs["metadata"], key, "g_A" if key == "d_A" else None)
    if isinstance(values, str):
        return _array(values).reshape(n_atoms)
    return np.asarray(values, dtype=float).reshape(n_atoms)


def _metadata_value(metadata, key, legacy_key=None):
    if key in metadata:
        return metadata[key]
    if legacy_key is not None and legacy_key in metadata:
        return metadata[legacy_key]
    raise KeyError(key)


def _row_value(row, key, legacy_key=None):
    if key in row and row[key] not in (None, ""):
        return row[key]
    if legacy_key is not None and legacy_key in row:
        return row[legacy_key]
    raise KeyError(key)


def _numbers(values):
    return ";".join(f"{value:.16g}" for value in np.asarray(values, dtype=float).reshape(-1))


def _cutoff_pi(metadata, key):
    if f"{key}_pi" in metadata:
        return float(metadata[f"{key}_pi"])
    return float(metadata[key]) / np.pi


def _metadata_cutoff(metadata, key):
    if f"{key}_pi" in metadata:
        return float(metadata[f"{key}_pi"]) * np.pi
    return float(metadata[key])
