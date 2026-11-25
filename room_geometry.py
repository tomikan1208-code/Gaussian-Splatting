from collections import defaultdict
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "apps/pycolmap-4.2.1-python313"))
import pycolmap


def registered_names(model):
    reconstruction = pycolmap.Reconstruction(model)
    return {reconstruction.images[i].name for i in reconstruction.reg_image_ids()}


def rescale_model(model, output, dimensions):
    """Copy the reconstruction and move its intrinsics/observations into native pixels."""
    reconstruction = pycolmap.Reconstruction(model)
    targets = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        size = dimensions[image.name]
        if targets.setdefault(image.camera_id, size) != size:
            raise RuntimeError(f"Native dimensions differ within camera {image.camera_id}: {image.name}")

    scale = {}
    for camera_id, (width, height) in targets.items():
        camera = reconstruction.cameras[camera_id]
        factors = (width / camera.width, height / camera.height)
        if factors != (1.0, 1.0):
            camera.rescale(width, height)
            scale[camera_id] = factors

    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        if image.camera_id in scale:
            factors = np.array(scale[image.camera_id])
            for point in image.points2D:
                point.xy = point.xy * factors

    output.mkdir()
    reconstruction.write(output)


def representative_images(reconstruction):
    groups = defaultdict(list)
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        groups[image.name.split("/", 1)[0]].append(image)
    chosen = []
    for name, images in sorted(groups.items()):
        images.sort(key=lambda image: image.name)
        count = min(len(images), 32 if name == "photos" else 12)
        slots = np.linspace(0, len(images) - 1, count).round().astype(int)
        chosen.extend(images[index] for index in slots)
    return chosen


def write_orientation_sample(model, target):
    reconstruction = pycolmap.Reconstruction(model)
    keep = {image.frame_id for image in representative_images(reconstruction)}
    for frame_id in list(reconstruction.reg_frame_ids()):
        if frame_id not in keep:
            reconstruction.deregister_frame(frame_id)
    target.mkdir()
    reconstruction.write(target)
    return len(keep)


def normalise(vector):
    return vector / np.linalg.norm(vector)


def camera_up(images):
    # Photos were decoded with EXIF orientation, and FFmpeg applied video display rotation.
    # The recovered camera axes give the sign and dominant direction of "image up".
    photos = [image for image in images if image.name.startswith("photos/")]
    preferred = photos if len(photos) >= 8 else images
    directions = np.array([-image.cam_from_world().rotation.matrix()[1] for image in preferred])
    guess = np.median(directions, axis=0)
    if np.linalg.norm(guess) < 0.1:
        guess = directions[0]
    accepted = directions[directions @ normalise(guess) > 0.55]
    return normalise(accepted.mean(axis=0) if len(accepted) else guess)


def basis_with_up(up, preferred_x):
    x = preferred_x - up * np.dot(up, preferred_x)
    if np.linalg.norm(x) < 0.1:
        axis = np.eye(3)[np.argmin(np.abs(up))]
        x = axis - up * np.dot(up, axis)
    x = normalise(x)
    return np.array([x, np.cross(up, x), up])


def find_floor(points, cameras):
    # Only search low points and planes near horizontal. A broad floor outranks small
    # planes such as a chair seat, and the camera centres should be above that floor.
    limits = np.quantile(points, [0.01, 0.99], axis=0)
    span = np.quantile(points[:, :2], [0.05, 0.95], axis=0)
    footprint = (span[1] - span[0]).clip(min=1e-8)
    width = max(footprint)
    tolerance = 0.004 * width
    ceiling = min(np.quantile(points[:, 2], 0.40), np.quantile(cameras[:, 2], 0.30) - 0.05 * width)
    mask = (points[:, 2] < ceiling) & np.all((points >= limits[0]) & (points <= limits[1]), axis=1)
    low = points[mask]
    if len(low) < 50:
        return None

    rng = np.random.default_rng(17)
    sample = low[rng.choice(len(low), min(22000, len(low)), replace=False)]
    a, b, c = sample[rng.integers(len(sample), size=(384, 3))].transpose(1, 0, 2)
    directions = np.cross(b - a, c - a)
    norms = np.linalg.norm(directions, axis=1)
    usable = (norms > 1e-8) & (np.abs(directions[:, 2]) > math.cos(math.radians(14)) * norms)
    if not np.any(usable):
        return None
    directions = directions[usable] / norms[usable, None]
    directions *= np.sign(directions[:, 2:3])
    distances = np.einsum("ij,ij->i", a[usable], directions)

    votes = np.zeros(len(directions), dtype=int)
    for start in range(0, len(directions), 32):
        stop = start + 32
        votes[start:stop] = (np.abs(sample @ directions[start:stop].T - distances[start:stop]) < tolerance).sum(axis=0)

    # Recheck the strongest hypotheses against the full point cloud, not just the fitting sample.
    for candidate in np.argsort(-votes)[:24]:
        normal = directions[candidate]
        offset = distances[candidate]
        inliers = points[np.abs(points @ normal - offset) < tolerance]
        if len(inliers) < max(45, 0.018 * len(low)):
            continue
        spread = np.diff(np.quantile(inliers[:, :2], [0.08, 0.92], axis=0), axis=0)[0]
        if np.min(spread / footprint) < 0.30:
            continue
        if np.quantile(cameras @ normal, 0.15) < offset + 0.03 * width:
            continue
        centre = np.mean(inliers, axis=0)
        residuals = inliers - centre
        _, vectors = np.linalg.eigh(residuals.T @ residuals)
        fitted = vectors[:, 0] * np.sign(vectors[2, 0])
        if fitted[2] > math.cos(math.radians(14)):
            return fitted, centre, len(inliers), float(np.median(np.abs(residuals @ fitted))), tolerance
    return None


def camera_view(image, reconstruction):
    camera = reconstruction.cameras[image.camera_id]
    return {
        "name": image.name,
        "world_to_camera": image.cam_from_world().rotation.matrix().tolist(),
        "centre": image.projection_center().tolist(),
        "width": camera.width,
        "height": camera.height,
        "fx": camera.focal_length_x,
        "fy": camera.focal_length_y,
        "cx": camera.principal_point_x,
        "cy": camera.principal_point_y,
    }


def calculate_alignment(model, manhattan=None):
    reconstruction = pycolmap.Reconstruction(model)
    images = [reconstruction.images[i] for i in reconstruction.reg_image_ids()]
    if not images:
        raise RuntimeError("No registered camera positions to align.")
    rough_up = camera_up(representative_images(reconstruction))
    basis = basis_with_up(rough_up, np.array([1.0, 0, 0]))
    method = "camera-up"
    if manhattan and (manhattan / "images.bin").is_file():
        aligned = pycolmap.Reconstruction(manhattan)
        shared = sorted(set(aligned.reg_image_ids()).intersection(reconstruction.reg_image_ids()))
        if shared:
            image_id = shared[0]
            before = reconstruction.images[image_id].cam_from_world().rotation.matrix()
            after = aligned.images[image_id].cam_from_world().rotation.matrix()
            axes = after.T @ before
            membership = axes @ rough_up
            index = int(np.argmax(np.abs(membership)))
            if abs(membership[index]) > 0.75:
                up = normalise(axes[index] * np.sign(membership[index]))
                basis = basis_with_up(up, axes[(index + 1) % 3])
                method = "Manhattan axes and recovered camera up"

    cameras = np.array([image.projection_center() for image in images])
    points = np.array([p.xyz for p in reconstruction.points3D.values() if p.error < 2 and p.track.length() >= 3])
    floor = None
    offset = np.zeros(3)
    if len(points) >= 60:
        provisional_basis = basis.copy()
        fit = find_floor(points @ basis.T, cameras @ basis.T)
        if fit is not None:
            normal, centre, support, residual, tolerance = fit
            basis = basis_with_up(normal @ basis, basis[0])
            origin = centre @ provisional_basis
            offset[2] = -(basis @ origin)[2]
            floor = (support, residual, tolerance)
            method += " and floor plane"

    if floor is not None:
        support, residual, tolerance = floor

    camera_bounds = np.quantile(cameras @ basis.T, [0.08, 0.92], axis=0)
    offset[:2] = -camera_bounds.mean(axis=0)[:2]

    # Prefer a sharp, wider level shot with many triangulated observations.
    def view_score(image):
        orientation = image.cam_from_world().rotation.matrix()
        camera = reconstruction.cameras[image.camera_id]
        upright = -np.dot(orientation[1], basis[2])
        tilt = abs(np.dot(orientation[2], basis[2]))
        photo_bonus = 1.35 if image.name.startswith("photos/") else 1.0
        landscape_bonus = 1.25 if camera.width >= camera.height else 1.0
        return image.num_points3D * max(0.1, upright) ** 2 * (1 - 0.8 * tilt) * photo_bonus * landscape_bonus

    start_view = max(images, key=view_score)
    transform = np.eye(4)
    transform[:3, :3] = basis
    transform[:3, 3] = offset
    return {"method": method, "world_to_blender": transform.tolist(), "view": camera_view(start_view, reconstruction),
            "floor": None if floor is None else {"points": support, "median_residual": residual,
                                                "threshold": tolerance}}
