import csv
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, Rectangle

from src.one_dimensional_renormalization import get_1d_bare_parameters
from src.xp_config import ExperimentConfig


DEFAULT_GEOMETRY_DIR = Path(__file__).resolve().parents[1] / "2d_setup" / "geometries"


def geometry_filename(
    kind: str,
    lengths,
    matter_width: float,
    matter_height: float,
    N_A: int,
    N_layer: int | None = None,
) -> str:
    geometry = _geometry_parameters(kind, lengths, matter_width, matter_height, N_A, N_layer)
    return (
        f"matter_geom_{geometry['kind']}_"
        f"L{_numbers_label(geometry['lengths'])}_"
        f"MW{_number_label(geometry['matter_width'])}_"
        f"MH{_number_label(geometry['matter_height'])}_"
        f"NA{geometry['N_A']}_NL{geometry['N_layer']}.csv"
    )


def create_geometry(
    kind: str,
    lengths,
    matter_width: float,
    matter_height: float,
    N_A: int,
    N_layer: int | None = None,
    directory: str | Path = DEFAULT_GEOMETRY_DIR,
) -> dict:
    geometry = _build_geometry(kind, lengths, matter_width, matter_height, N_A, N_layer)
    path = Path(directory) / geometry_filename(
        kind,
        lengths,
        matter_width,
        matter_height,
        N_A,
        N_layer,
    )
    _save_geometry(path, geometry)
    geometry["path"] = path
    return geometry


def load_geometry(
    kind: str,
    lengths,
    matter_width: float,
    matter_height: float,
    N_A: int,
    N_layer: int | None = None,
    directory: str | Path = DEFAULT_GEOMETRY_DIR,
) -> dict:
    path = Path(directory) / geometry_filename(
        kind,
        lengths,
        matter_width,
        matter_height,
        N_A,
        N_layer,
    )
    return _read_geometry(path)


def plot_geometry(
    kind: str,
    lengths,
    matter_width: float,
    matter_height: float,
    N_A: int,
    N_layer: int | None = None,
    directory: str | Path = DEFAULT_GEOMETRY_DIR,
    ax=None,
    photon_position: tuple[float, float] | list[float] | None = None,
) -> dict:
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    geometry = load_geometry(
        kind,
        lengths,
        matter_width,
        matter_height,
        N_A,
        N_layer,
        directory=directory,
    )
    _draw_geometry(ax, geometry, photon_position=photon_position)
    return geometry


def make_fiber_config(
    param_photon: dict[str, object],
    param_atom: dict[str, object],
    param_time_evol: dict[str, float],
    cutoffs: dict[str, float],
    correction_scheme: str | None = None,
) -> ExperimentConfig:
    required = {"Omega_A", "g_A"}
    missing = required.difference(param_atom)
    if missing:
        raise ValueError(f"param_atom is missing keys: {sorted(missing)}.")

    if "geometry" in param_atom:
        geometry = param_atom["geometry"]
        lengths = _as_1d_array(geometry["lengths"], float)
        atom_positions = _as_2d_positions(geometry["atom_positions"])
    else:
        geometry_keys = {"lengths", "positions"}
        missing_geometry = geometry_keys.difference(param_atom)
        if missing_geometry:
            raise ValueError(
                "param_atom must contain 'geometry' or both 'lengths' and 'positions'."
            )
        lengths = _as_1d_array(param_atom["lengths"], float)
        atom_positions = _as_2d_positions(param_atom["positions"])

    if len(lengths) != 2:
        raise ValueError("2D fiber configs require param_atom['lengths'] = [L1, L2].")

    n_atoms = len(atom_positions)
    target_Omega_A = _atom_parameter_array(
        param_atom["Omega_A"], n_atoms, float, "Omega_A"
    )
    target_g_A = _atom_parameter_array(param_atom["g_A"], n_atoms, float, "g_A")

    if correction_scheme is None:
        Omega_0 = target_Omega_A
        g_0 = target_g_A
    elif correction_scheme == "1d_scheme":
        Omega_0, g_0 = _effective_1d_corrected_parameters(
            target_Omega_A,
            target_g_A ,
            lengths,
            cutoffs,
        )

    elif correction_scheme == "2d_scheme":
        raise NotImplementedError("correction_scheme='2d_scheme' is not implemented yet.")
    else:
        raise ValueError(
            "correction_scheme must be None, '1d_scheme', or '2d_scheme'."
        )

    return ExperimentConfig(
        dimension=2,
        lengths=lengths,
        atom_positions=atom_positions,
        atom_frequencies=Omega_0,
        atom_couplings=g_0,
        photon=param_photon,
        time=param_time_evol,
        cutoffs=cutoffs,
    )


def _build_geometry(
    kind: str,
    lengths,
    matter_width: float,
    matter_height: float,
    N_A: int,
    N_layer: int | None,
) -> dict:
    geometry = _geometry_parameters(kind, lengths, matter_width, matter_height, N_A, N_layer)

    if geometry["kind"] == "slab":
        layer_counts = _layer_counts(geometry["N_A"], geometry["N_layer"])
        atom_positions = _slab_positions(
            layer_counts,
            geometry["matter_width"],
            geometry["matter_height"],
        )
    else:
        layer_counts = ()
        atom_positions = _ellipse_positions(
            geometry["N_A"],
            geometry["matter_width"],
            geometry["matter_height"],
        )

    label = Path(
        geometry_filename(
            geometry["kind"],
            geometry["lengths"],
            geometry["matter_width"],
            geometry["matter_height"],
            geometry["N_A"],
            geometry["N_layer"],
        )
    ).stem.removeprefix("matter_geom_")

    geometry.update(
        {
            "atom_positions": atom_positions,
            "label": label,
            "layer_counts": layer_counts,
            "L1": geometry["lengths"][0],
            "L2": geometry["lengths"][1],
            "width": geometry["lengths"][0],
            "height": geometry["lengths"][1],
            "layers": None if geometry["kind"] == "disk" else geometry["N_layer"],
            "interface_radius": _disk_radius(geometry),
            "semi_axes": (geometry["matter_width"] / 2, geometry["matter_height"] / 2),
        }
    )
    return geometry


def _geometry_parameters(
    kind: str,
    lengths,
    matter_width: float,
    matter_height: float,
    N_A: int,
    N_layer: int | None,
) -> dict:
    kind = _geometry_kind(kind)
    lengths = _geometry_lengths(lengths)
    matter_width = float(matter_width)
    matter_height = float(matter_height)
    N_A = int(N_A)
    N_layer = _normalized_layer_count(kind, N_A, N_layer)

    if N_A <= 0:
        raise ValueError("N_A must be positive.")
    if matter_width <= 0 or matter_height <= 0:
        raise ValueError("matter_width and matter_height must be positive.")
    if matter_width > lengths[0] or matter_height > lengths[1]:
        raise ValueError("matter dimensions must fit inside lengths = [L1, L2].")

    return {
        "kind": kind,
        "lengths": lengths,
        "matter_width": matter_width,
        "matter_height": matter_height,
        "N_A": N_A,
        "N_layer": N_layer,
    }


def _geometry_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    if normalized not in {"disk", "slab"}:
        raise ValueError("kind must be 'disk' or 'slab'.")
    return normalized


def _geometry_lengths(lengths) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=float).reshape(-1)
    if len(lengths) != 2:
        raise ValueError("2D geometries require lengths = [L1, L2].")
    if np.any(lengths <= 0):
        raise ValueError("lengths must be positive.")
    return lengths


def _normalized_layer_count(kind: str, N_A: int, N_layer: int | None) -> int:
    if kind == "disk":
        return 0
    if N_layer is None:
        raise ValueError("N_layer must be set for slab geometries.")

    N_layer = int(N_layer)
    if N_layer <= 0:
        raise ValueError("N_layer must be positive for slab geometries.")
    if N_layer > N_A:
        raise ValueError("N_layer must be smaller than or equal to N_A.")
    return N_layer


def _number_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _numbers_label(values) -> str:
    return "x".join(_number_label(value) for value in np.asarray(values, dtype=float).reshape(-1))


def _disk_radius(geometry: dict) -> float:
    if geometry["kind"] != "disk":
        return np.nan
    if np.isclose(geometry["matter_width"], geometry["matter_height"]):
        return geometry["matter_width"] / 2
    return np.nan


def _ellipse_positions(N_A: int, matter_width: float, matter_height: float) -> np.ndarray:
    if N_A == 1:
        return np.zeros((1, 2), dtype=float)

    positions = [np.zeros(2)]
    remaining = N_A - 1
    n_rings = max(1, int(np.ceil(np.sqrt(remaining / 6))))
    ring_counts = _split_counts(remaining, n_rings)

    for ring_index, count in enumerate(ring_counts, start=1):
        radius_fraction = ring_index / n_rings
        angle_offset = np.pi / count if ring_index % 2 == 0 else 0.0
        angles = 2 * np.pi * np.arange(count) / count + angle_offset
        positions.extend(
            np.column_stack(
                (
                    matter_width / 2 * radius_fraction * np.cos(angles),
                    matter_height / 2 * radius_fraction * np.sin(angles),
                )
            )
        )

    return np.asarray(positions[:N_A], dtype=float)


def _slab_positions(
    layer_counts: tuple[int, ...],
    matter_width: float,
    matter_height: float,
) -> np.ndarray:
    n_layers = len(layer_counts)
    x_values = (
        np.array([0.0])
        if n_layers == 1
        else np.linspace(-matter_width / 2, matter_width / 2, n_layers)
    )

    positions = []
    for x_value, count in zip(x_values, layer_counts):
        y_values = (
            np.array([0.0])
            if count == 1
            else np.linspace(-matter_height / 2, matter_height / 2, count)
        )
        positions.extend((x_value, y_value) for y_value in y_values)

    return np.asarray(positions, dtype=float)


def _layer_counts(N_A: int, N_layer: int) -> tuple[int, ...]:
    counts = np.full(N_layer, N_A // N_layer, dtype=int)
    remaining = N_A - int(np.sum(counts))
    if remaining:
        centers = np.arange(N_layer) - (N_layer - 1) / 2
        order = np.argsort(np.abs(centers), kind="mergesort")
        counts[order[:remaining]] += 1
    return tuple(int(count) for count in counts)


def _split_counts(total: int, bins: int) -> np.ndarray:
    weights = np.arange(1, bins + 1, dtype=float)
    raw_counts = total * weights / np.sum(weights)
    counts = np.floor(raw_counts).astype(int)
    remaining = total - int(np.sum(counts))
    order = np.argsort(raw_counts - counts)[::-1]
    counts[order[:remaining]] += 1
    return counts


def _save_geometry(path: Path, geometry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "atom_index",
                "x",
                "y",
                "kind",
                "lengths",
                "matter_width",
                "matter_height",
                "N_A",
                "N_layer",
            ]
        )
        for atom_index, (x_value, y_value) in enumerate(geometry["atom_positions"]):
            writer.writerow(
                [
                    atom_index,
                    f"{x_value:.16g}",
                    f"{y_value:.16g}",
                    geometry["kind"],
                    _csv_numbers(geometry["lengths"]),
                    f"{geometry['matter_width']:.16g}",
                    f"{geometry['matter_height']:.16g}",
                    geometry["N_A"],
                    geometry["N_layer"],
                ]
            )


def _read_geometry(path: str | Path) -> dict:
    path = Path(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise ValueError(f"Geometry file is empty: {path}")

    first = rows[0]
    atom_positions = np.asarray(
        [(float(row["x"]), float(row["y"])) for row in rows],
        dtype=float,
    )

    kind = _geometry_kind(first.get("kind") or ("slab" if _read_layer_count(first) > 0 else "disk"))
    geometry = _build_geometry(
        kind,
        _read_lengths(first),
        float(first.get("matter_width", first.get("width", first.get("L1")))),
        float(first.get("matter_height", first.get("height", first.get("L2")))),
        _read_atom_count(first),
        _read_layer_count(first),
    )
    geometry["atom_positions"] = atom_positions
    geometry["path"] = path
    return geometry


def _read_lengths(row: dict) -> np.ndarray:
    if row.get("lengths"):
        return _csv_number_array(row["lengths"])
    return np.asarray([float(row["L1"]), float(row["L2"])], dtype=float)


def _read_atom_count(row: dict) -> int:
    if row.get("N_A"):
        return int(float(row["N_A"]))
    if row.get("atom_layers"):
        return int(np.sum(_csv_number_array(row["atom_layers"], dtype=int)))
    raise ValueError("Geometry CSV is missing N_A.")


def _read_layer_count(row: dict) -> int:
    if row.get("N_layer"):
        return int(float(row["N_layer"]))
    if row.get("atom_layers"):
        return len(_csv_number_array(row["atom_layers"], dtype=int))
    return 0


def _csv_numbers(values) -> str:
    return ";".join(f"{float(value):.16g}" for value in values)


def _csv_number_array(value: str, dtype=float) -> np.ndarray:
    if dtype is int:
        return np.asarray(
            [int(float(part)) for part in value.split(";") if part != ""],
            dtype=int,
        )
    return np.asarray([dtype(part) for part in value.split(";") if part != ""], dtype=dtype)


def _draw_geometry(
    ax,
    geometry: dict,
    photon_position: tuple[float, float] | list[float] | None = None,
) -> None:
    box_width, box_height = geometry["lengths"]
    matter_width = geometry["matter_width"]
    matter_height = geometry["matter_height"]
    atom_positions = geometry["atom_positions"]

    ax.add_patch(
        Rectangle(
            (-box_width / 2, -box_height / 2),
            box_width,
            box_height,
            fill=False,
            edgecolor="#2f3a45",
            linewidth=1.2,
        )
    )

    if geometry["kind"] == "disk":
        ax.add_patch(
            Ellipse(
                (0.0, 0.0),
                matter_width,
                matter_height,
                fill=False,
                edgecolor="#25855a",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
            )
        )
    else:
        _draw_slab_matter(ax, matter_width, matter_height, atom_positions)

    ax.scatter(
        atom_positions[:, 0],
        atom_positions[:, 1],
        s=34,
        color="#1f8f4d",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    if photon_position is not None:
        _draw_photon(ax, photon_position, box_width, box_height)

    x_margin = max(0.02 * box_width, 0.5)
    y_margin = max(0.12 * box_height, 0.02)
    ax.set_xlim(-box_width / 2 - x_margin, box_width / 2 + x_margin)
    ax.set_ylim(-box_height / 2 - y_margin, box_height / 2 + y_margin)
    ax.set_aspect("auto" if box_height / box_width < 0.04 else "equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(color="0.9", linewidth=0.5)


def _draw_slab_matter(
    ax,
    matter_width: float,
    matter_height: float,
    atom_positions: np.ndarray,
) -> None:
    ax.add_patch(
        Rectangle(
            (-matter_width / 2, -matter_height / 2),
            matter_width,
            matter_height,
            fill=False,
            edgecolor="#5e6874",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
    )

    for x_value in np.unique(np.round(atom_positions[:, 0], decimals=12)):
        ax.plot(
            [x_value, x_value],
            [-matter_height / 2, matter_height / 2],
            color="#5e6874",
            linestyle=":",
            linewidth=0.8,
            alpha=0.75,
        )


def _draw_photon(
    ax,
    photon_position: tuple[float, float] | list[float],
    box_width: float,
    box_height: float,
) -> None:
    arrow_width = max(0.02 * box_height, 0.003)
    arrow_head_width = max(0.12 * box_height, 0.015)
    arrow_head_length = max(0.01 * box_width, 0.25)

    ax.scatter(
        [photon_position[0]],
        [photon_position[1]],
        marker=">",
        s=60,
        color="#3759a8",
        zorder=4,
    )
    ax.arrow(
        photon_position[0] + 0.4,
        photon_position[1],
        2.0,
        0.0,
        width=arrow_width,
        head_width=arrow_head_width,
        head_length=arrow_head_length,
        color="#3759a8",
        length_includes_head=True,
    )


def _effective_1d_corrected_parameters(
    target_Omega_A: np.ndarray,
    target_g_A: np.ndarray,
    lengths: np.ndarray,
    cutoffs: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    if not _is_effectively_1d(lengths, cutoffs):
        raise NotImplementedError(
            "correction_scheme='1d_scheme' is only supported when the retained "
            "2D modes are effectively 1D, i.e. all retained modes have transverse "
            "index n_y=0."
        )

    if not _is_uniform_array(target_Omega_A) or not _is_uniform_array(target_g_A):
        raise ValueError(
            "effective 1D correction requires uniform target Omega_A and g_A."
        )

    omega_min, omega_max = _cutoff_bounds(cutoffs)
    Omega_0, g_0 = get_1d_bare_parameters(
        float(target_Omega_A[0]),
        float(target_g_A[0]),
        omega_min,
        omega_max,
        apply_corrections=True,
    )
    if not np.isfinite(Omega_0) or not np.isfinite(g_0):
        raise ValueError(
            "1D correction failed to find finite bare parameters. "
            "Check that Omega_A lies safely inside the cutoff window."
        )

    return (
        np.full(len(target_Omega_A), Omega_0, dtype=float),
        np.full(len(target_g_A), g_0, dtype=float),
    )


def _is_effectively_1d(lengths: np.ndarray, cutoffs: dict[str, float]) -> bool:
    mode_indices = _retained_mode_indices(lengths, cutoffs)
    if len(mode_indices) == 0:
        raise ValueError("cutoffs retain no modes, so the effective dimension is undefined.")
    return np.all(mode_indices[:, 1] == 0)


def _retained_mode_indices(lengths: np.ndarray, cutoffs: dict[str, float]) -> np.ndarray:
    omega_min, omega_max = _cutoff_bounds(cutoffs)
    lengths = _as_1d_array(lengths, float)
    delta_p = 2 * np.pi / lengths
    n_max = np.floor(omega_max / delta_p).astype(int)

    axes = [np.arange(-limit, limit + 1) for limit in n_max]
    grids = np.meshgrid(*axes, indexing="ij")
    mode_indices = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    momenta = mode_indices * delta_p[None, :]
    omegas = np.linalg.norm(momenta, axis=1)
    mask = (omegas >= omega_min) & (omegas <= omega_max)
    return mode_indices[mask]


def _cutoff_bounds(cutoffs: dict[str, float]) -> tuple[float, float]:
    expected = {"omega_min", "omega_max"}
    if set(cutoffs) != expected:
        raise ValueError("cutoffs must contain exactly omega_min and omega_max.")

    omega_min = float(cutoffs["omega_min"])
    omega_max = float(cutoffs["omega_max"])
    if omega_min < 0:
        raise ValueError("omega_min must be non-negative.")
    if omega_max <= omega_min:
        raise ValueError("omega_max must be larger than omega_min.")
    return omega_min, omega_max


def _is_uniform_array(values: np.ndarray) -> bool:
    return bool(np.allclose(values, values[0], rtol=1e-12, atol=1e-12))


def _atom_parameter_array(value, n_atoms: int, dtype, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(n_atoms, array.item(), dtype=dtype)

    array = array.reshape(-1)
    if len(array) != n_atoms:
        raise ValueError(f"{name} must be scalar or contain one value per atom.")
    return array


def _as_1d_array(value, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _as_2d_positions(value) -> np.ndarray:
    positions = np.asarray(value, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, 2)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("param_atom['positions'] must have shape (n_atoms, 2).")
    return positions
