"""2D CSV storage; numerical calibration is in bare_param.py.

Set SINGLE_PHOTON_DATABASE or pass directory/database_dir to use another database.
The default is the database_fiber directory next to this module.
"""
import csv
import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.xp_config import normalize_cutoffs

DATABASE_DIR = Path(os.environ.get("SINGLE_PHOTON_DATABASE", Path(__file__).resolve().parent / "database_fiber"))
BARE_PARAM_DIR = DATABASE_DIR / "bare_param"
TARGET_PROFILE_DIR = DATABASE_DIR / "target_profiles"


def bare_param_path(bare_param_id=None, directory=BARE_PARAM_DIR):
    """
    Return the CSV path storing one corrected pair (Omega_0, d_0).

    These files are the 2D analogue of the bare parameters returned directly by
    get_bare_parameters in the waveguide model. Each file is indexed by an id in
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
        "d_A": _array(raw["d_A"]),
        "Omega_min": _metadata_cutoff(raw, "omega_min"),
        "Omega_max": _metadata_cutoff(raw, "omega_max"),
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
        "d_A": _array(raw["d_A"]),
        "reference_omega_min": _metadata_cutoff(raw, "reference_omega_min"),
        "reference_omega_max": _metadata_cutoff(raw, "reference_omega_max"),
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
        np.asarray([float(row["d_0"]) for row in rows]),
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
        stream.write(f"# d_A: {_numbers(target_metadata['d_A'])}\n")
        stream.write(f"# reference_omega_min_pi: {_cutoff_pi(target_metadata, 'reference_omega_min'):.16g}\n")
        stream.write(f"# reference_omega_max_pi: {_cutoff_pi(target_metadata, 'reference_omega_max'):.16g}\n")
        stream.write(f"# omega_min_pi: {cutoffs['Omega_min'] / np.pi:.16g}\n")
        stream.write(f"# omega_max_pi: {cutoffs['Omega_max'] / np.pi:.16g}\n")
        writer = csv.writer(stream)
        writer.writerow(["j", "Omega_0", "d_0"])
        for j, (omega, d_value) in enumerate(zip(Omega_0, d_0)):
            writer.writerow([j, f"{omega:.16g}", f"{d_value:.16g}"])
    return path


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
            and np.isclose(metadata["Omega_min"], cutoffs["Omega_min"])
            and np.isclose(metadata["Omega_max"], cutoffs["Omega_max"])
        ):
            return path

    raise FileNotFoundError("Requested bare parameters do not exist in database_fiber.")


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
        stream.write(f"# d_A: {metadata['d_A']}\n")
        stream.write(f"# reference_omega_min_pi: {float(metadata['reference_omega_min_pi']):.16g}\n")
        stream.write(f"# reference_omega_max_pi: {float(metadata['reference_omega_max_pi']):.16g}\n")
        profile.to_csv(stream, index=False)
    return path


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
    inside = (profile["omega"] > cutoffs["Omega_min"]) & (
        profile["omega"] < cutoffs["Omega_max"]
    )
    profile = profile.loc[inside, ["omega", "j", "k", "W", "F_real", "F_imag"]]
    profile = profile.reset_index(drop=True)
    profile["W"] = profile["W"] / float(profile["W"].sum())
    return profile


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
    values = profile.attrs["metadata"][key]
    if isinstance(values, str):
        return _array(values).reshape(n_atoms)
    return np.asarray(values, dtype=float).reshape(n_atoms)


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
