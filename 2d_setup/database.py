"""Read and write the calibration CSV files in database_geom."""

from pathlib import Path

import numpy as np
import pandas as pd

DATABASE_DIR = Path(__file__).resolve().parent / "database_geom"
BARE_PARAM_DIR = DATABASE_DIR / "bare_param"
TARGET_PROFILE_DIR = DATABASE_DIR / "target_profiles"


def bare_param_path(bare_param_id=None, directory=BARE_PARAM_DIR):
    """Path for an existing calibration, or the next available id."""
    directory = Path(directory)
    if bare_param_id is None:
        ids = [int(p.stem.removeprefix("bare_param_id"))
               for p in directory.glob("bare_param_id*.csv")]
        bare_param_id = max(ids, default=0) + 1
    return directory / f"bare_param_id{int(bare_param_id)}.csv"


def target_profile_path(target_profile_id=None, directory=TARGET_PROFILE_DIR):
    """Path for an existing target profile, or the next available id."""
    directory = Path(directory)
    if target_profile_id is None:
        ids = [int(p.stem.removeprefix("target_profile_id"))
               for p in directory.glob("target_profile_id*.csv")]
        target_profile_id = max(ids, default=0) + 1
    return directory / f"target_profile_id{int(target_profile_id)}.csv"


def read_metadata(path):
    """Read a target/calibration header into arrays, integers and frequencies.

    Frequencies stored in units of pi are converted on reading. The returned
    dictionary has the same types as a freshly generated target profile.
    """
    metadata = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            key, value = (part.strip() for part in line[1:].split(":", 1))
            if key in ("Omega_A", "d_A"):
                metadata[key] = np.fromstring(value, sep=";")
            elif key in ("id_geom", "id_target_profile", "nb_omega_values"):
                metadata[key] = int(value)
            else:
                # All remaining header fields are frequency bounds.
                if key.endswith("_pi"):
                    key, value = key[:-3], float(value) * np.pi
                key = {"omega_min": "Omega_min", "omega_max": "Omega_max"}.get(key, key)
                metadata[key] = float(value)
    if Path(path).stem.startswith("target_profile_id"):
        metadata["id_target_profile"] = int(Path(path).stem.removeprefix("target_profile_id"))
    return metadata


def load_bare_parameters_csv(path):
    """Return the atomic Omega_0 and d_0 arrays."""
    data = pd.read_csv(path, comment="#", float_precision="round_trip")
    return data["Omega_0"].to_numpy(), data["d_0"].to_numpy()


def save_bare_parameters_csv(Omega_0, d_0, target_metadata, cutoffs, path):
    """Write fitted parameters and the physical target that produced them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "id_geom": target_metadata.get("id_geom"),
        "id_target_profile": target_metadata.get("id_target_profile"),
        "Omega_A": ";".join(f"{v:.16g}" for v in target_metadata["Omega_A"]),
        "d_A": ";".join(f"{v:.16g}" for v in target_metadata["d_A"]),
        "reference_omega_min_pi": target_metadata["reference_omega_min"] / np.pi,
        "reference_omega_max_pi": target_metadata["reference_omega_max"] / np.pi,
        "omega_min_pi": cutoffs["Omega_min"] / np.pi,
        "omega_max_pi": cutoffs["Omega_max"] / np.pi,
    }
    data = pd.DataFrame({"j": np.arange(len(Omega_0)), "Omega_0": Omega_0, "d_0": d_0})
    with path.open("w", newline="") as stream:
        for key, value in header.items():
            if value is not None:
                stream.write(f"# {key}: {value}\n")
        data.to_csv(stream, index=False, float_format="%.16g")
    return path


def find_bare_param_path(Omega_A, d_A, cutoffs, id_geom=None,
                         id_target_profile=None, directory=BARE_PARAM_DIR):
    """Find the calibration matching the target, geometry and frequency window."""
    for path in sorted(Path(directory).glob("bare_param_id*.csv")):
        metadata = read_metadata(path)
        if id_geom is not None and metadata.get("id_geom") != id_geom:
            continue
        if id_target_profile is not None and metadata.get("id_target_profile") != id_target_profile:
            continue
        if (np.allclose(metadata["Omega_A"], Omega_A)
                and np.allclose(metadata["d_A"], d_A)
                and np.isclose(metadata["Omega_min"], cutoffs["Omega_min"])
                and np.isclose(metadata["Omega_max"], cutoffs["Omega_max"])):
            return path
    raise FileNotFoundError("Requested bare parameters do not exist in database_geom.")


def save_target_profile_csv(profile, path):
    """Write the spectral target and its physical parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = profile.attrs["metadata"]
    header = {
        "id_geom": metadata.get("id_geom"),
        "nb_omega_values": metadata["nb_omega_values"],
        "Omega_A": ";".join(f"{v:.16g}" for v in metadata["Omega_A"]),
        "d_A": ";".join(f"{v:.16g}" for v in metadata["d_A"]),
        "reference_omega_min_pi": metadata["reference_omega_min"] / np.pi,
        "reference_omega_max_pi": metadata["reference_omega_max"] / np.pi,
    }
    with path.open("w", newline="") as stream:
        for key, value in header.items():
            if value is not None:
                stream.write(f"# {key}: {value}\n")
        profile.to_csv(stream, index=False)
    return path


def load_target_profile(target_profile, cutoffs):
    """Load a target, keep frequencies inside the window and normalize weights."""
    if isinstance(target_profile, pd.DataFrame):
        profile = target_profile.copy()
    else:
        profile = pd.read_csv(target_profile, comment="#")
        profile.attrs["metadata"] = read_metadata(target_profile)
    inside = (profile["omega"] > cutoffs["Omega_min"]) & (profile["omega"] < cutoffs["Omega_max"])
    profile = profile.loc[inside].reset_index(drop=True)
    profile["W"] /= profile["W"].sum()
    return profile
