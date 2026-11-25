from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Matrix, Vector


root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from room_data import latest_output, read_json, splat_has_baked_alignment


def show_room(runs):
    ply, owner = latest_output(runs)
    baked_alignment = splat_has_baked_alignment(ply)
    # The launcher already uses --factory-startup. Resetting Blender here
    # invalidates the window context that the PLY import operator requires.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    if not bpy.ops.wm.ply_import.poll():
        raise RuntimeError("PLY import needs an active Blender GUI window.")

    result = bpy.ops.wm.ply_import(
        filepath=str(ply), forward_axis="Y", up_axis="Z", import_attributes=True
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"PLY import failed: {ply} ({result})")

    room = next((obj for obj in bpy.context.scene.objects
                 if obj.type == "POINTCLOUD"), None)
    if room is None or room.type != "POINTCLOUD" or "radiance:base" not in room.data.attributes:
        raise RuntimeError(f"Could not open {ply} as native Gaussian splats.")
    room.name = f"Room Splats - {owner.name}"
    room["source_ply"] = str(ply)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    scene.eevee.taa_render_samples = 192
    alignment_path = owner / "alignment.json"
    if alignment_path.is_file():
        alignment = read_json(alignment_path)
    else:
        from room_geometry import calculate_alignment
        report = owner / "result.json"
        dataset_run = read_json(report).get("dataset_run", owner.name) if report.is_file() else owner.name
        model = runs / dataset_run / "dataset/sparse"
        alignment = calculate_alignment(model) if (model / "images.bin").is_file() else None

    if alignment:
        transform = Matrix(alignment["world_to_blender"])
        if not baked_alignment:
            room.matrix_world = transform @ room.matrix_world
        room["alignment_method"] = alignment["method"]
        room["floor_support"] = alignment["floor"]["points"] if alignment["floor"] else 0
        view = alignment["view"]
        optics = bpy.data.cameras.new("Room overview")
        optics.sensor_fit = "HORIZONTAL"
        optics.lens = optics.sensor_width * view["fx"] / view["width"]
        optics.shift_x = (view["width"] / 2 - view["cx"]) / view["width"]
        optics.shift_y = (view["cy"] - view["height"] / 2) / view["width"] * view["fx"] / view["fy"]
        optics.clip_start = 0.005
        optics.clip_end = 1000
        camera = bpy.data.objects.new(optics.name, optics)
        camera["source_frame"] = view["name"]
        scene.collection.objects.link(camera)
        world_to_camera = Matrix(view["world_to_camera"])
        forward = transform.to_3x3() @ world_to_camera[2]
        camera.location = transform @ Vector(view["centre"])
        camera.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        scene.render.resolution_x = view["width"]
        scene.render.resolution_y = view["height"]
        scene.render.pixel_aspect_y = view["fx"] / view["fy"]
        scene.render.resolution_percentage = 100

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.overlay.show_overlays = False
                if scene.camera:
                    space.region_3d.view_perspective = "CAMERA"
                    space.region_3d.view_camera_zoom = 3
    print(f"Opened {len(room.data.points):,} Gaussian splats from {ply}")
    return room


def _open_when_ready():
    # --python may execute before Blender has an active GUI context.
    wm = bpy.context.window_manager
    windows = wm.windows if wm is not None else ()
    if not windows:
        return 0.2  # Try again on the next UI timer tick.

    try:
        with bpy.context.temp_override(window=windows[0]):
            show_room(root / "runs")
    except Exception:
        traceback.print_exc()
        print("Room import failed; check the traceback above.")
    return None  # Do not run again.


if __name__ == "__main__":
    if bpy.app.background:
        raise RuntimeError("Launch Blender with its GUI (without --background).")
    bpy.app.timers.register(_open_when_ready, first_interval=0.2)
