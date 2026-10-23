"""Create, load and plot atomic geometries."""

import csv
from pathlib import Path

import numpy as np

from .database import DATABASE_DIR

DEFAULT_GEOMETRY_DIR = DATABASE_DIR / "geometries"


def create_geometry(
    kind,
    lengths,
    matter_width,
    matter_height,
    N_A,
    N_layer=None,
    rotation=0.0,
    directory=DEFAULT_GEOMETRY_DIR,
    geometry_id=None,
    save=True,
):
    """Place atoms in a slab or disk, rotate them, and optionally save the geometry."""
    kind = str(kind).lower()
    lengths = np.asarray(lengths, dtype=float).reshape(2)
    matter_width = float(matter_width)
    matter_height = float(matter_height)
    N_A = int(N_A)
    rotation = float(rotation)

    if kind == "slab":
        N_layer = int(N_layer)
        counts = np.full(N_layer, N_A // N_layer, dtype=int)
        remaining = N_A - int(np.sum(counts))
        if remaining:
            centers = np.arange(N_layer) - (N_layer - 1) / 2
            counts[np.argsort(np.abs(centers), kind="mergesort")[:remaining]] += 1
        x_values = np.linspace(-matter_width / 2, matter_width / 2, N_layer)
        if N_layer == 1:
            x_values = np.array([0.0])
        positions = []
        for x_value, count in zip(x_values, counts):
            y_values = np.linspace(-matter_height / 2, matter_height / 2, count)
            if count == 1:
                y_values = np.array([0.0])
            positions.extend((x_value, y_value) for y_value in y_values)
        atom_positions = np.asarray(positions, dtype=float)
    elif kind == "disk":
        N_layer = 0
        if N_A == 1:
            atom_positions = np.zeros((1, 2), dtype=float)
        else:
            positions = [np.zeros(2)]
            remaining = N_A - 1
            n_rings = max(1, int(np.ceil(np.sqrt(remaining / 6))))
            weights = np.arange(1, n_rings + 1, dtype=float)
            ring_counts = np.floor(remaining * weights / np.sum(weights)).astype(int)
            missing = remaining - int(np.sum(ring_counts))
            order = np.argsort(remaining * weights / np.sum(weights) - ring_counts)[::-1]
            ring_counts[order[:missing]] += 1
            for ring_index, count in enumerate(ring_counts, start=1):
                radius_fraction = ring_index / n_rings
                angle_offset = np.pi / count if ring_index % 2 == 0 else 0.0
                angles = 2 * np.pi * np.arange(count) / count + angle_offset
                positions.extend(
                    np.column_stack(
                        (
                            matter_width * radius_fraction * np.cos(angles) / 2,
                            matter_height * radius_fraction * np.sin(angles) / 2,
                        )
                    )
                )
            atom_positions = np.asarray(positions[:N_A], dtype=float)
    else:
        raise ValueError("kind must be 'slab' or 'disk'.")

    if rotation:
        c, s = np.cos(rotation), np.sin(rotation)
        atom_positions = atom_positions @ np.array([[c, -s], [s, c]]).T

    directory = Path(directory)
    if geometry_id is None:
        ids = [int(p.stem.removeprefix("geometry_id"))
               for p in directory.glob("geometry_id*.csv")]
        geometry_id = max(ids, default=0) + 1
    path = directory / f"geometry_id{int(geometry_id)}.csv"
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            stream.write(f"# kind: {kind}\n")
            stream.write(f"# lengths: {lengths[0]:.16g};{lengths[1]:.16g}\n")
            stream.write(f"# matter_width: {matter_width:.16g}\n")
            stream.write(f"# matter_height: {matter_height:.16g}\n")
            stream.write(f"# N_A: {N_A}\n")
            stream.write(f"# N_layer: {N_layer}\n")
            stream.write(f"# rotation: {rotation:.16g}\n")
            writer = csv.writer(stream)
            writer.writerow(["atom_index", "x", "y"])
            for atom_index, (x_value, y_value) in enumerate(atom_positions):
                writer.writerow([atom_index, f"{x_value:.16g}", f"{y_value:.16g}"])

    return {
        "id_geom": int(geometry_id), "kind": kind, "lengths": lengths,
        "matter_width": matter_width, "matter_height": matter_height,
        "N_A": N_A, "N_layer": N_layer, "rotation": rotation,
        "atom_positions": atom_positions, "path": path, "label": path.stem,
    }


def load_geometry_by_id(geometry_id, directory=DEFAULT_GEOMETRY_DIR):
    """Read the geometry metadata and atomic positions from one CSV."""
    path = Path(directory) / f"geometry_id{int(geometry_id)}.csv"
    with path.open(newline="") as stream:
        metadata = {}
        for line in stream:
            if not line.startswith("#"):
                rows = list(csv.DictReader(stream, fieldnames=line.strip().split(",")))
                break
            key, value = (part.strip() for part in line[1:].split(":", 1))
            metadata[key] = value
    return {
        "id_geom": int(geometry_id),
        "kind": metadata["kind"],
        "lengths": np.fromstring(metadata["lengths"], sep=";"),
        "matter_width": float(metadata["matter_width"]),
        "matter_height": float(metadata["matter_height"]),
        "N_A": int(float(metadata["N_A"])),
        "N_layer": int(float(metadata["N_layer"])),
        "rotation": float(metadata.get("rotation", 0.)),
        "atom_positions": np.asarray([(float(row["x"]), float(row["y"])) for row in rows]),
        "path": path,
        "label": path.stem,
    }


def plot_geometry(geometry, ax=None, photon_position=None):
    """Plot a geometry already created or loaded by the caller."""
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    box_width, box_height = geometry["lengths"]
    ax.add_patch(Rectangle((-box_width / 2, -box_height / 2), box_width, box_height, fill=False))

    if geometry["kind"] == "disk":
        ax.add_patch(Ellipse((0, 0), geometry["matter_width"], geometry["matter_height"], fill=False, linestyle="--"))
    else:
        c, s = np.cos(geometry["rotation"]), np.sin(geometry["rotation"])
        rectangle = np.asarray(
            [
                [-geometry["matter_width"] / 2, -geometry["matter_height"] / 2],
                [geometry["matter_width"] / 2, -geometry["matter_height"] / 2],
                [geometry["matter_width"] / 2, geometry["matter_height"] / 2],
                [-geometry["matter_width"] / 2, geometry["matter_height"] / 2],
            ]
        )
        ax.add_patch(Polygon(rectangle @ np.array([[c, -s], [s, c]]).T, closed=True, fill=False, linestyle="--"))

    atom_positions = geometry["atom_positions"]
    ax.scatter(atom_positions[:, 0], atom_positions[:, 1], s=34)
    if photon_position is not None:
        ax.scatter([photon_position[0]], [photon_position[1]], marker=">", s=60)
    ax.set_xlim(-box_width / 2, box_width / 2)
    ax.set_ylim(-box_height / 2, box_height / 2)
    ax.set_aspect("equal", adjustable="box")
    return ax
