import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple

import numpy as np
from numpy.polynomial.legendre import legder, legval

from room_data import BAKED_ALIGNMENT_COMMENT, SPLAT_FIELDS, read_splat_count, read_splat_header
from room_geometry import pycolmap


PLY_LIMIT = 10_000_000_000
ROWS_PER_CHUNK = 32_768
POSITION_FIELDS = frozenset(b"x y z".split())
COLOR_FIELDS = (frozenset(b"red green blue".split()), frozenset(b"r g b".split()))


class TrainingPlyHeader(NamedTuple):
    count: int
    is_splat: bool
    is_baked: bool


def read_training_ply(path):
    """Inspect a standard point-cloud or uncompressed 3DGS PLY before preparing images."""
    error = f"Invalid or unsupported training PLY: {path}"
    count = 0
    fields, elements = set(), set()
    is_baked = False
    with path.open("rb") as stream:
        if stream.readline(4096).strip() != b"ply" or stream.readline(4096).split() not in (
            [b"format", b"ascii", b"1.0"], [b"format", b"binary_little_endian", b"1.0"],
            [b"format", b"binary_big_endian", b"1.0"],
        ):
            raise ValueError(error)
        element = None
        for _ in range(256):
            line = stream.readline(4096).strip()
            if not line:
                raise ValueError(error)
            parts = line.split()
            if line == b"end_header":
                if not stream.read(1):
                    raise ValueError(f"Training PLY has no vertex data: {path}")
                break
            if line == b"comment " + BAKED_ALIGNMENT_COMMENT:
                is_baked = True
            if len(parts) == 3 and parts[0] == b"element":
                element = parts[1]
                if element in elements:
                    raise ValueError(error)
                elements.add(element)
                if element == b"vertex":
                    if not parts[2].isdigit():
                        raise ValueError(error)
                    count = int(parts[2])
            if parts[:1] == [b"property"] and element == b"vertex":
                if len(parts) != 3 or parts[-1] in fields:
                    raise ValueError(error)
                fields.add(parts[-1])
        else:
            raise ValueError(error)

        if count < 1 or not POSITION_FIELDS.issubset(fields):
            raise ValueError(f"Expected a PLY with x, y, z vertices. Export compressed splats as standard PLY: {path}")
        is_splat = bool((SPLAT_FIELDS - POSITION_FIELDS) & fields) or any(
            name.startswith(b"f_rest_") for name in fields
        )
        rgb = fields & (COLOR_FIELDS[0] | COLOR_FIELDS[1])
        if is_splat:
            if not SPLAT_FIELDS.issubset(fields) or rgb:
                raise ValueError(f"Expected standard Gaussian parameters in the splat PLY: {path}")
            if sh_layout(fields)[0] > 3:
                raise ValueError(f"The trainers support spherical harmonics up to degree 3: {path}")
        elif rgb and rgb not in COLOR_FIELDS:
            raise ValueError(f"Expected all three RGB color channels in the point cloud: {path}")
        if is_baked:
            stream.seek(0)
            if read_splat_header(stream) is None:
                raise ValueError(f"Incomplete room PLY or unsupported baked alignment: {path}")
    return TrainingPlyHeader(count, is_splat, is_baked)


def real_sh_band(directions, degree):
    """Evaluate the 3DGS real spherical harmonics in order -degree through +degree."""
    x, y, z = directions.T
    azimuth = np.arctan2(y, x)
    radius = np.hypot(x, y)
    polynomial = np.zeros(degree + 1)
    polynomial[degree] = 1
    columns = []
    for order in range(-degree, degree + 1):
        m = abs(order)
        associated = (-radius) ** m * legval(z, legder(polynomial, m))
        normalization = math.sqrt(
            (2 * degree + 1) / (4 * math.pi)
            * math.factorial(degree - m) / math.factorial(degree + m)
        )
        if order != 0:
            angular = np.sin if order < 0 else np.cos
            associated *= math.sqrt(2) * angular(m * azimuth)
        columns.append(normalization * associated)
    return np.column_stack(columns)


def sh_layout(fields):
    count = sum(name.startswith(b"f_rest_") for name in fields)
    max_degree = math.isqrt(count // 3 + 1) - 1
    per_channel = (max_degree + 1) ** 2 - 1
    rest_names = [f"f_rest_{index}".encode("ascii") for index in range(count)]
    if max_degree > 4 or count != 3 * per_channel or not all(name in fields for name in rest_names):
        raise ValueError("Invalid spherical harmonics layout in Gaussian-splat PLY.")
    return max_degree, rest_names


def sh_rotations(fields, rotation):
    max_degree, rest_names = sh_layout(fields)
    if max_degree == 0:
        return []

    # Sample directions around the sphere to find the change of basis within each band.
    sample = np.arange(32)
    z = 1 - (2 * sample + 1) / len(sample)
    azimuth = sample * (math.pi * (3 - math.sqrt(5)))
    radius = np.sqrt(1 - z * z)
    directions = np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))
    original_directions = directions @ rotation
    channels = np.array([fields[name] for name in rest_names]).reshape(3, -1)
    bands = []
    for degree in range(1, max_degree + 1):
        new_basis = real_sh_band(directions, degree)
        old_basis = real_sh_band(original_directions, degree)
        coefficients = np.linalg.lstsq(new_basis, old_basis, rcond=None)[0]
        indices = channels[:, degree**2 - 1 : (degree + 1)**2 - 1]
        bands.append((indices, coefficients.T))
    return bands


def transformed_header(header, *, bake):
    # Brush interprets "Vertical axis: z" as -Z. The alignment comment declares +Z.
    comments = [line for line in header.comments
                if line != BAKED_ALIGNMENT_COMMENT and not line.lower().startswith(b"vertical axis:")]
    if bake:
        comments.append(BAKED_ALIGNMENT_COMMENT)
    lines = [b"ply", b"format binary_little_endian 1.0"]
    lines.extend(b"comment " + line for line in comments)
    lines.append(f"element vertex {header.count}".encode("ascii"))
    lines.extend(b"property float " + name for name in header.fields)
    lines.append(b"end_header\n")
    return b"\n".join(lines)


def bake_room_ply(source, output, world_to_blender):
    """Apply the room's rigid transform to a complete splat PLY and publish it atomically."""
    return transform_room_ply(source, output, world_to_blender, bake=True)


def restore_training_ply(source, output, world_to_blender):
    """Move a room PLY back into the camera coordinate system used for training."""
    return transform_room_ply(source, output, world_to_blender, bake=False)


def transform_room_ply(source, output, world_to_blender, *, bake):
    transform = np.asarray(world_to_blender, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Room alignment must be a finite 4x4 matrix.")
    rotation = transform[:3, :3]
    offset = transform[:3, 3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-7)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-7)
        or np.linalg.det(rotation) <= 0
    ):
        raise ValueError("Room alignment must be a rigid rotation and translation.")
    x, y, z, w = pycolmap.Rotation3d(rotation).quat
    if not bake:
        rotation = rotation.T
        offset = -rotation @ offset
        x, y, z = -x, -y, -z

    # Left multiplication composes the room rotation with each splat's wxyz rotation.
    quaternion = np.array([
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w],
    ])

    temporary = None
    try:
        with source.open("rb") as stream:
            header = read_splat_header(stream)
            if header is None:
                raise ValueError(f"Incomplete or unsupported Gaussian-splat PLY: {source}")
            is_baked = BAKED_ALIGNMENT_COMMENT in header.comments
            if bake and is_baked:
                raise ValueError(f"Room alignment is already baked into {source}.")
            if not bake and not is_baked:
                raise ValueError(f"Missing baked room alignment in {source}.")
            output_header = transformed_header(header, bake=bake)
            width = len(header.fields)
            if len(output_header) + header.count * width * 4 > PLY_LIMIT:
                raise ValueError("The final PLY exceeds the 10 GB limit.")

            fields = {name: index for index, name in enumerate(header.fields)}
            positions = [fields[name] for name in (b"x", b"y", b"z")]
            rotations = [fields[name] for name in (b"rot_0", b"rot_1", b"rot_2", b"rot_3")]
            normals = [fields[name] for name in (b"nx", b"ny", b"nz") if name in fields]
            if normals and len(normals) != 3:
                raise ValueError("Incomplete normals in Gaussian-splat PLY.")
            sh_bands = sh_rotations(fields, rotation)

            with NamedTemporaryFile(mode="wb", dir=output.parent, suffix=".ply.tmp", delete=False) as dest:
                temporary = Path(dest.name)
                dest.write(output_header)
                for start in range(0, header.count, ROWS_PER_CHUNK):
                    count = min(ROWS_PER_CHUNK, header.count - start)
                    batch = np.fromfile(stream, dtype="<f4", count=count * width).reshape(count, width)
                    batch[:, positions] = batch[:, positions] @ rotation.T + offset
                    batch[:, rotations] = batch[:, rotations] @ quaternion.T
                    if normals:
                        batch[:, normals] = batch[:, normals] @ rotation.T
                    for indices, coefficients in sh_bands:
                        batch[:, indices] = batch[:, indices] @ coefficients
                    batch.tofile(dest)
                dest.flush()
                os.fsync(dest.fileno())
        if read_splat_count(temporary) != header.count:
            raise RuntimeError("The aligned PLY is incomplete.")
        temporary.replace(output)
        return header.count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
