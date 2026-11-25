from collections import defaultdict, deque
from itertools import combinations
import os
import sqlite3

import numpy as np

from room_data import paired_images


MATCHERS = {"loma": "LoMa-B128", "sift": "SIFT"}
DEFAULT_MATCHER = "loma"
EXHAUSTIVE_LIMIT = 80
ANCHOR_PARTNERS = 12
BRIDGE_PARTNERS = 6
PAIR_ID_LIMIT = 2147483647


def matcher_options(matcher, colmap, faster_gs):
    if matcher == "sift":
        return ["--SiftExtraction.max_num_features", "4096"], [], None
    models = colmap.parent.parent / "models/loma-b128"
    detector = models / "loma_detector.onnx"
    descriptor = models / "loma_descriptor_dedode_b.onnx"
    weights = models / "loma_matcher_B128.onnx"
    for path in (detector, descriptor, weights):
        if not path.is_file():
            raise FileNotFoundError(f"LoMa-B128 weights are missing: {path}. See the setup section in README.md.")
    environment = dict(os.environ)
    torch_libraries = faster_gs / ".venv/Lib/site-packages/torch/lib"
    if torch_libraries.is_dir():
        environment["PATH"] = str(torch_libraries) + os.pathsep + environment["PATH"]
    return [
        "--FeatureExtraction.type", "LOMA_B128",
        "--LomaExtraction.max_num_features", "2048", "--LomaExtraction.use_fast_resize", "1",
        "--LomaExtraction.detector_model_path", detector,
        "--LomaExtraction.descriptor_b128_model_path", descriptor,
    ], [
        "--FeatureMatching.type", "LOMA_B128", "--LomaMatching.b128_model_path", weights,
    ], environment


def pairing_mode(requested, count):
    if requested != "auto":
        return requested
    return "exhaustive" if count <= EXHAUSTIVE_LIMIT else "selective"


def appearance_vectors(database, names, matcher):
    wanted = set(names)
    dtype = np.float32 if matcher == "loma" else np.uint8
    means = {}
    with sqlite3.connect(database) as con:
        for name, rows, data in con.execute(
            "SELECT i.name, d.rows, d.data FROM images i JOIN descriptors d USING(image_id) WHERE d.rows > 0"
        ):
            if name in wanted:
                descriptors = np.frombuffer(data, dtype=dtype).reshape(rows, -1)
                means[name] = descriptors.mean(axis=0, dtype=np.float32)
    if not means:
        return means
    center = np.mean(list(means.values()), axis=0)
    for name, vector in means.items():
        centered = vector - center
        means[name] = centered / max(1e-6, np.linalg.norm(centered))
    return means


def closest_pairs(queries, targets, vectors, limit, excluded, new_names=None, via=None):
    queries = [name for name in queries if name in vectors]
    targets = [name for name in targets if name in vectors]
    if not queries or not targets:
        return set()
    target_vectors = np.stack([vectors[name] for name in targets]).T
    result = set()
    for start in range(0, len(queries), 64):
        batch = queries[start:start + 64]
        scores = np.stack([vectors[name] for name in batch]) @ target_vectors
        for name, row in zip(batch, scores):
            partners = 0
            via_counts = via(name) if via else {}
            if via_counts:
                # Prefer a plausible second reference close to a verified neighbor.
                row += np.asarray([0.15 * min(3, via_counts.get(other, 0)) for other in targets])
            for i in np.argsort(-row, kind="stable"):
                other = targets[i]
                if name == other or (new_names is not None and name not in new_names and other not in new_names):
                    continue
                pair = tuple(sorted((name, other)))
                if pair in excluded:
                    continue
                result.add(pair)
                partners += 1
                if partners >= limit:
                    break
    return result


def selective_pairs(database, names, matcher, new_names, captures):
    def ranked_anchors(anchors, neighbors):
        if len(anchors) <= EXHAUSTIVE_LIMIT:
            return combinations(anchors, 2)
        vectors = appearance_vectors(database, anchors, matcher)
        return closest_pairs(anchors, anchors, vectors, ANCHOR_PARTNERS, neighbors, new_names)

    return paired_images(names, new_names, captures, choose_anchor_pairs=ranked_anchors)


def match_graph(database):
    with sqlite3.connect(database) as con:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        neighbors = defaultdict(list)
        tested = set()
        for pair_id, count, config in con.execute("SELECT pair_id, rows, config FROM two_view_geometries"):
            a, b = divmod(pair_id, PAIR_ID_LIMIT)
            if a not in names or b not in names:
                continue
            tested.add(tuple(sorted((names[a], names[b]))))
            if count >= 15 and config >= 2 and config != 7:
                neighbors[a].append(b)
                neighbors[b].append(a)

    return names, neighbors, tested


def additional_pairs(database, selected, matcher, known_names=None):
    names, neighbors, tested = match_graph(database)

    def connected(seeds):
        visited = set(seeds)
        queue = deque(visited)
        while queue:
            for image_id in neighbors[queue.popleft()]:
                if image_id not in visited:
                    visited.add(image_id)
                    queue.append(image_id)
        return visited

    if known_names is not None:
        anchored = connected({i for i, name in names.items() if name in known_names})
    else:
        unseen = set(names)
        anchored = set()
        for image_id in sorted(names):
            if image_id not in unseen:
                continue
            component = connected([image_id])
            unseen.difference_update(component)
            if len(component) > len(anchored):
                anchored = component
    if len(anchored) < 3:
        return set()
    references = sorted(names[i] for i in anchored)
    pending = sorted(set(selected).difference(references))
    if not pending:
        return set()
    vectors = appearance_vectors(database, pending + references, matcher)
    return closest_pairs(pending, references, vectors, BRIDGE_PARTNERS, tested)


def alternate_pairs(database, pending, registered, matcher):
    """Try additional direct links to cameras already placed in the reconstruction."""
    names, neighbors, tested = match_graph(database)
    ids = {name: image_id for image_id, name in names.items()}

    def via(name):
        candidates = defaultdict(int)
        for neighbor in neighbors.get(ids[name], ()):
            for other in neighbors.get(neighbor, ()):
                if names[other] in registered:
                    candidates[names[other]] += 1
        return candidates

    references = sorted(registered)
    vectors = appearance_vectors(database, [*pending, *references], matcher)
    return closest_pairs(pending, references, vectors, BRIDGE_PARTNERS, tested, via=via)
