from hashlib import file_digest
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
from typing import NamedTuple


POSE_EDGE = 1280
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
HEIF_EXTENSIONS = {".heic", ".heif"}
VIDEO_EXTENSIONS = {".mov", ".mp4"}
SOURCE_EXTENSIONS = IMAGE_EXTENSIONS | HEIF_EXTENSIONS | VIDEO_EXTENSIONS
CAPTURE_LOG = "frames.jsonl"
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_identity(path, folder=None):
    before = path.stat()
    with path.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while being read: {path}")
    name = path.name if folder is None else path.relative_to(folder).as_posix()
    return {"name": name, "sha256": digest, "bytes": after.st_size}


def source_files(folder):
    entries = list(folder.iterdir())
    for child in tuple(entries):
        if child.is_dir() and ((child / CAPTURE_LOG).is_file() or (child / "session.json").is_file()):
            entries.extend(child.iterdir())
    return sorted(
        (p for p in entries if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS),
        key=lambda p: p.relative_to(folder).as_posix().casefold(),
    )


def find_initial_ply(folder, selected):
    if selected is not None:
        path = selected.resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".ply":
            raise ValueError(f"Expected a PLY file for --init-ply: {path}")
        return path
    candidates = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".ply")
    if len(candidates) <= 1:
        return next(iter(candidates), None)
    named = [path for path in candidates if path.name.lower() == "init.ply"]
    if len(named) == 1:
        return named[0]
    raise ValueError("Multiple PLYs in the input folder: name the one to use init.ply or specify --init-ply (-InitPly).")


def find_ply_run(runs, identity):
    """Find the saved run for an exact copy of a previously exported room PLY."""
    for output in sorted(runs.glob("*/room.ply"), reverse=True):
        try:
            if output.stat().st_size != identity["bytes"]:
                continue
            report = read_json(output.parent / "result.json")
            if isinstance(report, dict) and source_identity(output)["sha256"] == identity["sha256"]:
                return output.parent, report
        except (OSError, ValueError, RuntimeError):
            continue
    return None


def capture_direction(orientation):
    if not isinstance(orientation, dict):
        return None
    angles = [orientation.get(f"{axis}_deg") for axis in ("alpha", "beta", "gamma")]
    if any(type(angle) not in {int, float} or not -360 <= angle <= 360 for angle in angles):
        return None
    age = orientation.get("sample_age_ms", 0)
    if type(age) not in {int, float} or not -50 <= age <= 500:
        return None
    # DeviceOrientation rotates by Z, X', Y''. A rear camera looks along device -Z.
    alpha, beta, gamma = map(math.radians, angles)
    ca, sa = math.cos(alpha), math.sin(alpha)
    sb, cb = math.sin(beta), math.cos(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    return [-ca * sg - sa * sb * cg, -sa * sg + ca * sb * cg, -cb * cg]


def capture_frames(folder):
    """Read filename and matching hints from a Wi-Fi capture sidecar."""
    rows = {}
    try:
        with (folder / CAPTURE_LOG).open(encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (ValueError, RecursionError):
                    continue
                if not isinstance(row, dict):
                    continue
                name, sequence = row.get("file"), row.get("sequence")
                if isinstance(name, str) and type(sequence) is int and sequence >= 1:
                    rows[name] = {"sequence": sequence, "direction": capture_direction(row.get("orientation"))}
    except FileNotFoundError:
        pass
    return rows


def matches_identity(cached, current):
    return all(cached.get(key) == current.get(key) for key in ("name", "sha256", "bytes", "capture"))


def image_count(model):
    with (model / "images.bin").open("rb") as stream:
        return struct.unpack("<Q", stream.read(8))[0]


def compatible_calibration(runs, sources, fps, reuse_run=None, *, matcher=None, pairing=None):
    by_name = {record["name"]: record for record in sources}
    paths = [runs / reuse_run / "calibration.json"] if reuse_run else list(runs.glob("*/calibration.json"))
    candidates = []
    for path in paths:
        try:
            record = read_json(path)
            if (
                record["version"] != 1 or record["video_fps"] != fps
                or record["pose_edge"] != POSE_EDGE
                or (matcher and record.get("matcher", "sift") != matcher)
                or (pairing and record.get("pairing") != pairing)
            ):
                continue
            previous = record["sources"]
            if not previous or any(not matches_identity(s, by_name.get(s["name"], {})) for s in previous):
                continue
            parent = path.parent
            model = parent / record["model"]
            if (
                not (parent / record["database"]).is_file()
                or image_count(model) != record["registered_frames"]
            ):
                continue
            candidates.append((len(previous), record["registered_frames"], parent, record))
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            continue
    if not candidates:
        if reuse_run:
            raise RuntimeError(f"The selected calibration does not match these inputs, capture logs, "
                               f"{fps:g} fps and the selected matcher.")
        return None
    _, _, path, record = max(candidates, key=lambda item: item[:3])
    return path, record


def allocate_sources(identities, paths, prior):
    previous = {record["name"]: record for record in prior}
    clip = max(
        (int(record["relative"].split("-")[-1]) for record in prior if record["kind"] == "video"),
        default=0,
    )
    photo = max(
        (int(Path(record["relative"]).stem.split("-")[-1]) for record in prior if record["kind"] == "photo"),
        default=0,
    )
    sources = []
    for identity, path in zip(identities, paths, strict=True):
        cached = previous.get(identity["name"])
        if cached is not None:
            sources.append(cached.copy())
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            clip += 1
            sources.append({**identity, "kind": "video", "relative": f"clip-{clip:02d}"})
        else:
            photo += 1
            sources.append({**identity, "kind": "photo", "relative": f"photos/photo-{photo:05d}.png", "count": 1})
    return sources


def staged_image_names(source):
    relative = source["relative"]
    if source["kind"] == "video":
        for index in range(1, source["count"] + 1):
            yield f"{relative}/frame-{index:05d}.png"
    else:
        yield relative


def pose_video_filter(fps, edge):
    filters = f"fps={fps}"
    if edge:
        filters += f",scale=w={edge}:h={edge}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    return filters


def link_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def link_images(paths, source, destination):
    for path in paths:
        link_file(path, destination / path.relative_to(source))


def copy_database(source, destination):
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(destination) as copy:
            original.backup(copy)


def direction_pairs(frames, directions, representatives):
    """Suggest a few distant views that point in a similar direction within one recording."""
    min_similarity = math.cos(math.radians(35))
    for index, frame in enumerate(frames):
        if frame not in directions or index in representatives:
            continue
        direction = directions[frame]
        candidates = []
        for other_index in representatives:
            other = frames[other_index]
            if abs(index - other_index) <= 10 or other not in directions:
                continue
            similarity = sum(a * b for a, b in zip(direction, directions[other]))
            if similarity >= min_similarity:
                candidates.append((-similarity, other))
        for _, other in sorted(candidates)[:2]:
            yield tuple(sorted((frame, other)))


def paired_images(names, new_names=None, captures=None, *, choose_anchor_pairs=None):
    captures = captures or {}
    clips = {}
    recordings = {}
    anchors = []
    for name in sorted(names):
        if name in captures:
            session, hint = captures[name]
            recordings.setdefault(session, []).append((hint["sequence"], name))
            continue
        folder = name.split("/", 1)[0]
        if re.fullmatch(r"clip-\d+", folder):
            clips.setdefault(folder, []).append(name)
        else:
            anchors.append(name)

    sequences = list(clips.values())
    sequences.extend([name for _, name in sorted(rows)] for rows in recordings.values())
    directions = {name: hint["direction"] for name, (_, hint) in captures.items() if hint["direction"] is not None}
    pairs = set()
    for frames in sequences:
        stride_count = min(20, len(frames))
        if stride_count == 1:
            representatives = {0}
        else:
            representatives = {round(i * (len(frames) - 1) / (stride_count - 1)) for i in range(stride_count)}
        anchors.extend(frames[index] for index in representatives)
        for i, frame in enumerate(frames):
            pairs.update(tuple(sorted((frame, other))) for other in frames[i + 1:i + 11])
        pairs.update(direction_pairs(frames, directions, representatives))
    anchors.sort()
    if choose_anchor_pairs is None:
        pairs.update(combinations(anchors, 2))
    else:
        pairs.update(choose_anchor_pairs(anchors, pairs))
    if new_names is not None:
        pairs = {pair for pair in pairs if pair[0] in new_names or pair[1] in new_names}
    return pairs


def dataset_is_complete(dataset):
    images = dataset / "images"
    registered = image_count(dataset / "sparse")
    return registered > 0 and registered == sum(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in images.rglob("*")
    )


def saved_training_data(runs, name):
    owner = runs / name
    try:
        result = owner / "result.json"
        report = read_json(result if result.is_file() else owner / "dataset-ready.json")
        dataset_run = report.get("dataset_run", name)
        calibration_run = report.get("calibration_run", name)
        if not (RUN_NAME.fullmatch(dataset_run) and RUN_NAME.fullmatch(calibration_run)):
            raise ValueError("Invalid saved run reference.")
        for candidate in dict.fromkeys((runs / dataset_run, owner)):
            dataset = candidate / "dataset"
            try:
                if dataset_is_complete(dataset):
                    return owner, dataset, report
            except (OSError, struct.error):
                continue
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    raise ValueError(f"Run {name} has no complete saved training images and camera poses.")


def dataset_from_run(root, calibration, edge):
    """Find the latest completed training pass for the given calibration and image size."""
    for result_file in sorted(root.glob("*/result.json"), reverse=True):
        try:
            report = read_json(result_file)
            parent = result_file.parent
            owner = report.get("calibration_run", parent.name)
            # A PLY-only pass can reuse lower-resolution images even when the user asks for original size.
            if (report.get("saved_training_run") or owner != calibration.name
                    or report["requested_long_edge"] != (edge or None)):
                continue
            reference = report.get("dataset_run", parent.name)
            if not RUN_NAME.fullmatch(reference):
                continue
            dataset = root / reference / "dataset"
            if dataset_is_complete(dataset):
                return dataset
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            continue
    # A calibration can survive Interrupted Brush training; a dataset is complete only
    # when lens correction and validation have written this receipt.
    for receipt in sorted(root.glob("*/dataset-ready.json"), reverse=True):
        try:
            report = read_json(receipt)
            dataset = receipt.parent / "dataset"
            if (
                report["calibration_run"] == calibration.name
                and report["requested_long_edge"] == (edge or None)
                and dataset_is_complete(dataset)
            ):
                return dataset
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            continue
    return None


BAKED_ALIGNMENT_COMMENT = b"Room alignment: baked (+Z up)"
SPLAT_FIELDS = frozenset(b"x y z rot_0 rot_1 rot_2 rot_3 scale_0 scale_1 scale_2 f_dc_0 f_dc_1 f_dc_2 opacity".split())


class SplatHeader(NamedTuple):
    count: int
    fields: tuple[bytes, ...]
    comments: tuple[bytes, ...]
    data_offset: int


def read_splat_header(stream):
    """Read the binary float vertex layout and confirm the splat payload is complete."""
    if stream.readline(4096).strip() != b"ply":
        return None
    if stream.readline(4096).strip() != b"format binary_little_endian 1.0":
        return None

    count = 0
    fields = []
    comments = []
    for _ in range(128):
        line = stream.readline(4096).strip()
        if line.startswith(b"comment "):
            comments.append(line.removeprefix(b"comment "))
        elif line.startswith(b"element vertex ") and count == 0:
            try:
                count = int(line.split()[-1])
            except ValueError:
                return None
        elif line.startswith(b"property float ") and count > 0:
            fields.append(line.split()[-1])
        elif line == b"end_header":
            offset = stream.tell()
            if count > 0 and len(fields) == len(set(fields)) and SPLAT_FIELDS.issubset(fields):
                if os.fstat(stream.fileno()).st_size == offset + count * len(fields) * 4:
                    return SplatHeader(count, tuple(fields), tuple(comments), offset)
            return None
        else:
            return None
    return None


def read_splat_count(path):
    with path.open("rb") as stream:
        header = read_splat_header(stream)
    return header.count if header else None


def splat_has_baked_alignment(path):
    with path.open("rb") as stream:
        header = read_splat_header(stream)
    if header is None:
        raise ValueError(f"Incomplete or unsupported Gaussian-splat PLY: {path}")
    return BAKED_ALIGNMENT_COMMENT in header.comments


def latest_output(runs):
    candidates = []
    for run in runs.iterdir():
        result = run / "result.json"
        ply = run / "room.ply"
        if result.is_file() and ply.is_file():
            candidates.append((ply.stat().st_mtime_ns, ply, run))
        # Direct Brush exports using the existing documented brush-* subfolder convention.
        for folder in run.glob("brush-*"):
            for export in folder.glob("room-*.ply"):
                candidates.append((export.stat().st_mtime_ns, export, run))
    for _, ply, run in sorted(candidates, reverse=True):
        try:
            if read_splat_count(ply):
                return ply, run
        except (OSError, ValueError):
            continue
    raise RuntimeError("No complete Gaussian-splat PLY in the runs folder.")
