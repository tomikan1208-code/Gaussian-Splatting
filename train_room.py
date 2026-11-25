import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
from datetime import datetime
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic


ROOT = Path(__file__).resolve().parent
COLMAP = ROOT / "apps/colmap-4.2.1-cuda/bin/colmap.exe"
BRUSH = ROOT / "apps/brush-0.3.0/brush_app.exe"
OPTIMIZED_BRUSH = ROOT / "apps/brush-0.3.0-training/brush_app.exe"
FASTER_GS = ROOT / "apps/faster-gs"
FASTER_GS_PYTHON = FASTER_GS / ".venv/Scripts/python.exe"
TRAINERS = ("brush", "faster-gs", "faster-gs-optimized")
TRAIN_PROGRESS = re.compile(r"\bTrain iter (\d+), (\d+) splats,")
FFMPEG = ROOT / "apps/ffmpeg/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe"
FFPROBE = FFMPEG.with_name("ffprobe.exe")

STAGES = (
    ("prepare", "Prepare images"),
    ("features", "Extract features"),
    ("match", "Match images"),
    ("mapper", "Estimate camera poses"),
    ("undistort", "Correct lens distortion"),
    ("level", "Align the room"),
    ("train", "Train splats"),
)
STAGE_LABELS = dict(STAGES)
STAGE_NUMBERS = {key: number for number, (key, _) in enumerate(STAGES, 1)}
sys.path.insert(0, str(ROOT / "apps/heif-python-packages"))
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from room_data import (
    POSE_EDGE, IMAGE_EXTENSIONS, RUN_NAME, atomic_json, read_json, source_identity, source_files,
    find_initial_ply, find_ply_run, saved_training_data, capture_frames,
    compatible_calibration, allocate_sources, link_file, link_images, copy_database,
    staged_image_names, image_count, dataset_from_run, read_splat_count, pose_video_filter,
)
from room_geometry import registered_names, rescale_model, write_orientation_sample, calculate_alignment
from room_matching import (
    DEFAULT_MATCHER, MATCHERS, additional_pairs, alternate_pairs, matcher_options, pairing_mode, selective_pairs,
)
from room_ply import bake_room_ply, read_training_ply, restore_training_ply


def duration_text(seconds):
    if seconds is None:
        return "measuring"
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02}m"
    return f"{minutes}m{seconds:02}s"


class Timeline:
    """Estimate remaining time from a prior run and replace it with live progress."""

    def __init__(self, estimates, log):
        self.estimates = estimates
        self.log = log
        self.started = monotonic()
        self.seconds = {}

    def start(self, stage):
        self.stage = stage
        self.stage_started = monotonic()
        self.last_report = -math.inf
        self.fraction = None
        self.estimated_end = self.stage_started + self.estimates[stage]
        self.detail = ""
        self.report(force=True)

    def update(self, done, total, unit):
        if total:
            fraction = min(1.0, max(0.0, done / total))
            now = monotonic()
            elapsed = now - self.stage_started
            if fraction and (self.fraction is None or fraction > self.fraction) and elapsed >= 2:
                self.estimated_end = self.stage_started + elapsed / fraction
            self.fraction = fraction
            self.detail = f"{done}/{total} {unit}"

    def report(self, force=False):
        now = monotonic()
        if not force and now - self.last_report < 10:
            return
        self.last_report = now
        remaining = self.estimated_end - now
        if self.fraction == 1:
            remaining = 0
        elif remaining <= 0:
            remaining = None
        later = sum(
            value for stage, value in self.estimates.items()
            if STAGE_NUMBERS[stage] > STAGE_NUMBERS[self.stage]
        )
        total = None if remaining is None else remaining + later
        prefix = f"[{STAGE_NUMBERS[self.stage]}/{len(STAGES)}] {STAGE_LABELS[self.stage]}"
        if self.detail:
            prefix += f" ({self.detail})"
        write_status(
            self.log,
            f"{prefix} | elapsed {duration_text(now - self.started)}"
            f" | stage ETA ~{duration_text(remaining)} | total ETA ~{duration_text(total)}",
        )

    def finish(self):
        self.seconds[self.stage] = round(monotonic() - self.stage_started, 2)
        write_status(
            self.log,
            f"{STAGE_LABELS[self.stage]} complete in {duration_text(self.seconds[self.stage])}"
            f" | elapsed {duration_text(monotonic() - self.started)}",
        )


def write_status(log, line):
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def run_tool(args, timeline, on_line=None, on_tick=None, env=None):
    args = [str(arg) for arg in args]
    log = timeline.log
    log.write(subprocess.list2cmdline(args) + "\n")
    log.flush()
    tail = deque(maxlen=12)
    lines = Queue()
    with subprocess.Popen(
        args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        def read_output():
            try:
                for raw in process.stdout:
                    lines.put(raw.rstrip("\r\n"))
            finally:
                lines.put(None)

        Thread(target=read_output, daemon=True).start()
        next_tick = monotonic()
        try:
            while True:
                try:
                    line = lines.get(timeout=0.5)
                except Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    log.write(line + "\n")
                    tail.append(line)
                    if on_line:
                        on_line(line)
                now = monotonic()
                if now >= next_tick:
                    if on_tick:
                        on_tick()
                    timeline.report()
                    next_tick = now + 1
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    log.flush()
    if code:
        excerpt = "\n".join(tail)
        raise RuntimeError(f"{Path(args[0]).name} exited {code}.\n{excerpt}")


def image_size(path):
    with Image.open(path) as image:
        return image.size


def video_info(path):
    args = [
        str(FFPROBE), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,nb_frames,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    output = subprocess.run(args, check=True, capture_output=True, text=True)
    report = json.loads(output.stdout)
    stream = report["streams"][0]
    duration = stream.get("duration") or report.get("format", {}).get("duration")
    if duration in {None, "N/A"}:
        from fractions import Fraction
        try:
            duration = int(stream["nb_frames"]) / float(Fraction(stream["avg_frame_rate"]))
        except (KeyError, ValueError, ZeroDivisionError):
            duration = 0
    return (stream["width"], stream["height"]), float(duration)


def pixels_at(size, edge):
    width, height = size
    factor = min(1.0, edge / max(size)) if edge else 1.0
    return width * height * factor * factor


def estimates_for(frame_count, average_pixels, steps, prepare_seconds):
    # Reference on this PC: 266 images at 1280, matching 130s, mapping 130s,
    # undistorting 216 images took 4.3s. The 1,621-image preview trained 3,000 steps in 122.51s.
    n = max(frame_count, 1)
    pixels_ratio = average_pixels / (1280 * 720)
    # The earlier stopped 1,519-image run supplied a second matching-rate observation.
    match_seconds_per_pair = 0.0017 + 0.0020 * min(1, 266 / n)
    return {
        "prepare": prepare_seconds,
        "features": max(1, 13 * n / 266),
        "match": max(1, (n * (n - 1) / 2) * match_seconds_per_pair),
        "mapper": max(1, 130 * (n / 266) ** 1.20),
        "undistort": max(1, 4.3 * n / 266 * pixels_ratio * 2),
        "level": min(90, max(5, frame_count * 0.06)),
        "train": max(1, 122.51 * (steps / 3000) * pixels_ratio),
    }


def write_photo(source, destination):
    if source.suffix.lower() == ".png":
        shutil.copy2(source, destination)
        return
    with Image.open(source) as photo:
        image = ImageOps.exif_transpose(photo)
        image.save(destination, "PNG", compress_level=3)


def frame_selection(indices):
    """Build an FFmpeg select expression for sorted, zero-based frame indices."""
    ranges = []
    for index in indices:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], index)
        else:
            ranges.append((index, index))

    # The filter needs to stay shallow even when a long clip has many gaps.
    def expression(start, stop):
        if stop - start == 1:
            first, last = ranges[start]
            return f"between(n,{first},{last})" if first != last else f"eq(n,{first})"
        middle = (start + stop) // 2
        return f"if(lt(n,{ranges[middle][0]}),{expression(start, middle)},{expression(middle, stop)})"

    return expression(0, len(ranges))


def prepare_images(frames, records, paths, fps, edge, timeline, selected=None, photos_from=None):
    expected_total = sum(
        record["count"] if selected is None else len(selected.get(record["relative"], ()))
        for record in records
    )
    index = 0
    for record in records:
        source = paths[record["name"]]
        relative = record["relative"]
        names = None if selected is None else selected.get(relative, ())
        if names is not None and not names:
            continue
        if record["kind"] == "photo":
            output = frames / relative
            if photos_from is not None:
                link_file(photos_from / relative, output)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                write_photo(source, output)
                timeline.log.write(f"Prepared photo: {source.name} -> {relative}\n")
            index += 1
            timeline.update(index, expected_total, "images")
            timeline.report()
            continue

        clip = frames / relative
        clip.mkdir(parents=True)
        video_filter = pose_video_filter(fps, edge)
        pattern = "frame-%05d.png"
        filter_options = ["-vf", video_filter]
        expected = record["count"]
        if names is not None:
            numbered_names = sorted((int(Path(name).stem.split("-")[-1]), name) for name in names)
            expected = len(numbered_names)
            selection = frame_selection(number - 1 for number, _ in numbered_names)
            filter_file = clip / "select.filter"
            filter_file.write_text(f"{video_filter},select='{selection}'", encoding="ascii")
            pattern = "selected-%05d.png"
            # Reading the filter from a file also avoids Windows' command-line length limit.
            filter_options = ["-/filter:v", filter_file, "-fps_mode", "passthrough"]

        def parse_line(line):
            match = re.fullmatch(r"frame=(\d+)", line)
            if match:
                timeline.update(index + min(int(match[1]), expected), expected_total, "images")

        run_tool(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
             "-nostats", "-progress", "pipe:1", "-stats_period", "1", "-i", source,
             "-map", "0:v:0", "-map_metadata", "-1", *filter_options,
             "-c:v", "png", "-compression_level", "3", clip / pattern],
            timeline, on_line=parse_line,
        )
        if names is None:
            count = sum(1 for _ in clip.glob("*.png"))
            record["count"] = count
        else:
            count = sum(1 for _ in clip.glob("selected-*.png"))
            if count != expected:
                raise RuntimeError(f"Expected {expected} native frames from {source.name}, extracted {count}.")
            for output_index, (_, name) in enumerate(numbered_names, 1):
                (clip / f"selected-{output_index:05d}.png").rename(frames / name)
        index += count


def image_paths(folder):
    return sorted(
        path for path in folder.rglob("*") if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def stage_pose_images(frames, records, paths, fps, timeline, cache):
    frames.mkdir()
    previous = {}
    if cache:
        owner, saved = cache
        previous = {record["name"]: record["count"] for record in saved["sources"]}
        cached_frames = owner / saved.get("frames", "frames")
        existing = [cached_frames / name for record in saved["sources"] for name in staged_image_names(record)]
        if cached_frames.is_dir() and all(path.is_file() for path in existing):
            link_images(existing, cached_frames, frames)
            records = [record for record in records if record["name"] not in previous]
        else:
            write_status(timeline.log, f"Regenerating temporary pose images from the original inputs for {owner.name}.")
    prepare_images(frames, records, paths, fps, POSE_EDGE, timeline)
    for record in records:
        if record["count"] != previous.get(record["name"], record["count"]):
            raise RuntimeError(f"Recovered frame count changed for {record['name']}; saved camera poses cannot be reused.")


def run_colmap(frames, files, args, run, work, timeline, old=None, new_names=None, captures=None):
    database = run / "database.db"
    sparse = run / "sparse"
    sparse.mkdir()
    all_names = [path.relative_to(frames).as_posix() for path in files]
    selected = all_names if new_names is None else sorted(new_names)
    known = None
    if old:
        old_run, old_record = old
        copy_database(old_run / old_record["database"], database)
        known = registered_names(old_run / old_record["model"])

    matcher = args.matcher
    mode = pairing_mode(args.pairing, len(files))
    extract_options, match_options, feature_env = matcher_options(matcher, COLMAP, FASTER_GS)
    seconds_per_image = 3.1 if matcher == "loma" else 59.5 / 1621
    seconds_per_pair = 0.126 if matcher == "loma" else 0.006
    pose_edge = (args.max_edge or 1600) if args.frames_path else POSE_EDGE
    common = [
        COLMAP, "feature_extractor", "--database_path", database, "--image_path", frames,
        "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.max_image_size", pose_edge,
        "--FeatureExtraction.num_threads", "4", *extract_options,
    ]
    write_status(timeline.log, f"Camera matching: {MATCHERS[matcher]}, {mode} pairing.")
    timeline.estimates["features"] = max(1, seconds_per_image * len(selected))
    timeline.start("features")
    groups = []
    if args.frames_path:
        groups.append((len(selected), ["--ImageReader.single_camera_per_folder", "1"]))
    else:
        for label, names, flags in (
            ("photos", [name for name in selected if name.startswith("photos/")], []),
            ("video-frames", [name for name in selected if name.startswith("clip-")],
             ["--ImageReader.single_camera_per_folder", "1"]),
        ):
            if names:
                listing = run / f"{label}.txt"
                listing.write_text("\n".join(names) + "\n", encoding="utf-8")
                groups.append((len(names), ["--image_list_path", listing, *flags]))

    done = 0
    for group_size, flags in groups:
        def feature_line(line):
            match = re.search(r"Processed file \[(\d+)/\d+\]", line)
            if match:
                timeline.update(done + int(match[1]), len(selected), "images")

        run_tool([*common, *flags], timeline, on_line=feature_line, env=feature_env)
        done += group_size
    timeline.finish()

    matching_flags = ["--database_path", database, "--FeatureMatching.use_gpu", "1",
                      "--FeatureMatching.num_threads", "4", *match_options]
    possible_pairs = len(files) * (len(files) - 1) // 2
    pairs = selective_pairs(database, all_names, matcher, new_names, captures) if mode == "selective" else None
    timeline.estimates["match"] = max(1, (possible_pairs if pairs is None else len(pairs)) * seconds_per_pair)
    timeline.start("match")

    def matching_line(line):
        block = re.search(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]", line)
        if block:
            row, rows, column, columns = map(int, block.groups())
            timeline.update((row - 1) * columns + column - 1, rows * columns, "blocks")
        block = re.search(r"Processing block \[(\d+)/(\d+)\]", line)
        if block:
            index, total = map(int, block.groups())
            timeline.update(index - 1, total, "blocks")

    def match_list(candidates, filename, status):
        listing = run / filename
        listing.write_text("".join(f"{a} {b}\n" for a, b in sorted(candidates)), encoding="utf-8")
        write_status(timeline.log, f"{status} {len(candidates):,} image pairs.")
        run_tool([COLMAP, "matches_importer", *matching_flags, "--match_type", "pairs",
                  "--match_list_path", listing], timeline, on_line=matching_line, env=feature_env)

    if pairs is None:
        write_status(timeline.log, f"Considering all {possible_pairs:,} unordered image pairs.")
        run_tool([COLMAP, "exhaustive_matcher", *matching_flags], timeline, on_line=matching_line, env=feature_env)
    elif pairs:
        match_list(pairs, "match-pairs.txt", "Matching")
    if mode == "selective":
        extra = additional_pairs(database, all_names, matcher, known)
        if extra:
            match_list(extra, "additional-pairs.txt", "Trying bridges between disconnected groups using")
    timeline.finish()

    use_global = not old and (
        args.mapper == "global" or (args.mapper == "auto" and len(files) >= 200)
    )
    if use_global:
        timeline.estimates["mapper"] = max(5, 460 * len(files) / 1621)
    timeline.start("mapper")

    def mapper_line(line):
        match = re.search(r"num_reg_frames=(\d+)", line)
        if match:
            timeline.detail = f"{match[1]} poses in active component"

    models = []
    mapper = "incremental"
    if use_global:
        # View-graph calibration adjusts intrinsics in place, on this disposable copy.
        global_db = work / "global-calibration.db"
        global_models = run / "global-sparse"
        global_models.mkdir()
        try:
            copy_database(database, global_db)
            run_tool([COLMAP, "view_graph_calibrator", "--database_path", global_db], timeline)
            run_tool(
                [COLMAP, "global_mapper", "--database_path", global_db, "--image_path", frames,
                 "--output_path", global_models, "--GlobalMapper.num_threads", "6",
                 "--GlobalMapper.ba_ceres_use_gpu", "0"], timeline,
            )
            models = collect_models(global_models)
            if models:
                mapper = "global"
        except RuntimeError as exc:
            write_status(timeline.log, f"Global mapping failed, continuing with incremental mapping: {exc}")
    if not models:
        continue_args = []
        if old:
            continue_args = ["--input_path", old_run / old_record["model"],
                             "--Mapper.fix_existing_frames", "1", "--Mapper.multiple_models", "0"]
        run_tool(
            [COLMAP, "mapper", "--database_path", database, "--image_path", frames,
             "--output_path", sparse, "--Mapper.num_threads", "6", *continue_args],
            timeline, on_line=mapper_line,
        )
        models = collect_models(sparse)

    if not models:
        raise RuntimeError("No connected camera model was reconstructed.")
    registered, model = max(models)
    if registered < len(files) and (mode == "selective" or mapper == "global"):
        placed = registered_names(model)
        pending = sorted(set(all_names) - placed)
        extra = alternate_pairs(database, pending, placed, matcher) if mode == "selective" else set()
        if extra or mapper == "global":
            write_status(timeline.log, f"Retrying camera registration for {len(pending):,} unplaced images.")
            if extra:
                match_list(extra, "registration-retry-pairs.txt", "Checking alternate registered views using")
            retry_models = run / "retry-sparse"
            retry_models.mkdir()
            # Keep placed cameras fixed while looking for 2D-to-3D support from more views.
            try:
                run_tool([COLMAP, "mapper", "--database_path", database, "--image_path", frames,
                          "--input_path", model, "--output_path", retry_models, "--Mapper.num_threads", "6",
                          "--Mapper.fix_existing_frames", "1", "--Mapper.multiple_models", "0"],
                         timeline, on_line=mapper_line)
                better = [candidate for candidate in collect_models(retry_models) if candidate[0] > registered]
                if better:
                    previous = registered
                    models.remove((registered, model))
                    registered, model = max(better)
                    models.append((registered, model))
                    write_status(timeline.log, f"Registered {registered - previous:,} more images on retry.")
            except RuntimeError as exc:
                write_status(timeline.log, f"Camera registration retry failed, keeping the initial model: {exc}")
    write_status(timeline.log, f"Using {registered} of {len(files)} images in one connected model.")
    if old:
        added = len(registered_names(model).intersection(selected))
        write_status(timeline.log, f"Registered {added} of {len(selected)} newly staged images.")
    run_tool([COLMAP, "model_analyzer", "--path", model], timeline)
    timeline.finish()
    return model, registered, [count for count, _ in models], mapper, mode


def collect_models(folder):
    candidates = [folder] if (folder / "images.bin").is_file() else sorted(folder.iterdir())
    return [(image_count(path), path) for path in candidates if (path / "images.bin").is_file()
            and image_count(path) >= 3]


def make_dataset(frames, model, run, edge, timeline):
    dataset = run / "dataset"

    def undistort_line(line):
        match = re.search(r"Undistorting image \[(\d+)/(\d+)\]", line)
        if match:
            timeline.update(*map(int, match.groups()), "images")

    run_tool(
        [COLMAP, "image_undistorter", "--image_path", frames, "--input_path", model,
         "--output_path", dataset, "--output_type", "COLMAP", "--max_image_size",
         edge or -1, "--num_threads", "4"],
        timeline, on_line=undistort_line,
    )
    images = dataset / "images"
    actual = {path.relative_to(images).as_posix() for path in image_paths(images)}
    if not actual or actual != registered_names(dataset / "sparse"):
        raise RuntimeError("The lens-corrected images do not match the registered camera names.")
    return dataset


def native_images(frames, model, work, records, paths, fps, timeline):
    required = registered_names(model)
    selected = defaultdict(list)
    for name in sorted(required):
        key = name.split("/", 1)[0] if name.startswith("clip-") else name
        selected[key].append(name)

    output = work / "native-frames"
    output.mkdir()
    prepare_images(output, records, paths, fps, 0, timeline, dict(selected), frames)
    dimensions = {name: image_size(output / name) for name in required}
    native_model = work / "native-sparse"
    rescale_model(model, native_model, dimensions)
    return output, native_model


def orient_dataset(dataset, run, work, timeline, shared_alignment=None):
    result = run / "alignment.json"
    if shared_alignment is not None and shared_alignment.is_file():
        shutil.copy2(shared_alignment, result)
        write_status(timeline.log, f"Reusing room alignment: {shared_alignment}")
        return
    timeline.start("level")
    source = dataset / "sparse"
    subset = work / "orientation-sample"
    aligned = work / "orientation-aligned"
    write_orientation_sample(source, subset)
    aligned.mkdir()
    try:
        run_tool([COLMAP, "model_orientation_aligner", "--image_path", dataset / "images",
                  "--input_path", subset, "--output_path", aligned,
                  "--method", "MANHATTAN-WORLD", "--max_image_size", "768"], timeline)
    except RuntimeError as exc:
        write_status(timeline.log, f"Image-axis estimation unavailable: {exc}")
    alignment = calculate_alignment(source, aligned)
    atomic_json(result, alignment)
    write_status(timeline.log, f"Room orientation: {alignment['method']}")
    timeline.finish()


def stage_brush_dataset(dataset, destination):
    for folder in ("images", "sparse", "masks"):
        files = (path for path in (dataset / folder).rglob("*")
                 if path.is_file() and path.suffix.lower() != ".ply")
        link_images(files, dataset, destination)


def train(dataset, run, args, timeline, brush, *, trainer="brush", export_interval=None,
          robust_camera_extent=True, initial_ply=None):
    sizes = [image_size(path) for path in image_paths(dataset / "images")]
    if not sizes:
        raise RuntimeError("The camera model did not produce any training images.")
    train_edge = max(map(max, sizes))
    average_pixels = sum(width * height for width, height in sizes) / len(sizes)
    timeline.estimates["train"] = estimates_for(len(sizes), average_pixels, args.steps, 0)["train"]
    timeline.start("train")
    write_status(timeline.log, f"Training engine: {brush.parent.name if trainer == 'brush' else trainer}")
    write_status(timeline.log, f"Training images: {len(sizes)}, largest edge {train_edge} px.")

    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    if export_interval is None:
        export_interval = min(1000, max(100, args.steps // 12))
    last_checkpoint = 0
    last_reported_step = 0

    def training_progress(step, splat_count):
        nonlocal last_reported_step
        if step > last_reported_step:
            last_reported_step = step
            timeline.update(step, args.steps, "steps")
            timeline.detail += f", {splat_count:,}/{args.max_splats:,} splats"

    def report_training_line(line):
        match = TRAIN_PROGRESS.search(line)
        if match:
            training_progress(*map(int, match.groups()))

    def check_training():
        nonlocal last_checkpoint
        files = sorted(checkpoints.glob("room_*.ply"))
        for candidate in reversed(files):
            step = int(candidate.stem.removeprefix("room_"))
            if step <= last_checkpoint:
                break
            splat_count = read_splat_count(candidate)
            if splat_count:
                last_checkpoint = step
                training_progress(step, splat_count)
                for older in files:
                    if older < candidate:
                        older.unlink(missing_ok=True)
                timeline.report(force=True)
                break

    growth_stop = min(15000, int(args.steps * 0.85))
    env = dict(os.environ)
    if trainer == "brush":
        env["RUST_LOG_STYLE"] = "never"
        env["RUST_LOG"] = "warn,brush_cli=info,brush_process::train_stream=info,cubecl_wgpu::runtime=info"
        if initial_ply is not None:
            env["RUST_LOG"] += ",brush_dataset::formats=info"
        command = [brush, "--total-steps", args.steps, "--max-splats", args.max_splats,
                   "--max-resolution", train_edge, "--growth-stop-iter", growth_stop,
                   "--export-path", checkpoints, "--export-name", "room_{iter}.ply",
                   "--export-every", export_interval, dataset]
    else:
        env["PYTHONIOENCODING"] = "utf-8"
        command = [FASTER_GS_PYTHON, "-u", FASTER_GS / "train_colmap.py", "--dataset", dataset,
                   "--checkpoints", checkpoints, "--steps", args.steps, "--max-splats", args.max_splats,
                   "--interval", export_interval, "--growth-stop", growth_stop]
        if initial_ply is not None:
            command.extend(("--init-ply", initial_ply))
        if trainer == "faster-gs-optimized":
            command.append("--prefetch")
            if robust_camera_extent:
                command.append("--robust-camera-extent")
    run_tool(command, timeline, on_line=report_training_line, on_tick=check_training, env=env)
    check_training()
    checkpoint = checkpoints / f"room_{args.steps:0{len(str(args.steps))}}.ply"
    if not checkpoint.is_file():
        raise RuntimeError(f"{trainer} exited without exporting the final PLY.")
    result = run / "room.ply"
    alignment = read_json(run / "alignment.json")
    write_status(timeline.log, "Baking the room alignment into the final PLY.")
    count = bake_room_ply(checkpoint, result, alignment["world_to_blender"])
    for old in checkpoints.glob("room_*.ply"):
        old.unlink()
    checkpoints.rmdir()
    timeline.finish()
    return result, count, train_edge


def main(args):
    if args.trainer != "brush" and not FASTER_GS_PYTHON.is_file():
        raise FileNotFoundError(f"Faster-GS Python is missing. See {FASTER_GS / 'README.md'}.")
    register_heif_opener()
    runs = args.runs_dir.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    prepared = args.frames_path.resolve(strict=True) if args.frames_path else None
    input_folder = prepared if prepared is not None else args.input_folder.resolve(strict=True)
    initial_ply = find_initial_ply(input_folder, args.init_ply)
    seed_record = seed_origin = seed_calibration = seed_alignment = None
    if initial_ply is not None:
        seed_stamp = initial_ply.stat()
        header = read_training_ply(initial_ply)
        if header.count > args.max_splats:
            raise ValueError(f"The initial PLY has {header.count:,} vertices. Set the splat limit to at least that many.")
        seed_record = {**source_identity(initial_ply), "path": str(initial_ply), "vertices": header.count,
                       "kind": "gaussian_splat" if header.is_splat else "point_cloud"}
        if header.is_splat:
            seed_origin = find_ply_run(runs, seed_record)
        if seed_origin:
            owner, report = seed_origin
            seed_calibration = report.get("calibration_run", owner.name)
            if not isinstance(seed_calibration, str) or not RUN_NAME.fullmatch(seed_calibration):
                raise ValueError(f"Invalid camera reference for the initial PLY in {owner}.")
        if header.is_baked:
            alignment_path = initial_ply.parent / "alignment.json"
            if seed_origin and (seed_origin[0] / "alignment.json").is_file():
                alignment_path = seed_origin[0] / "alignment.json"
            if not alignment_path.is_file():
                raise ValueError("The room PLY has baked alignment. Put its original alignment.json beside it "
                                 "or keep the original PLY in its saved run folder.")
            alignment = read_json(alignment_path)
            seed_alignment = alignment.get("world_to_blender") if isinstance(alignment, dict) else None
            if seed_alignment is None:
                raise ValueError(f"Missing world_to_blender in {alignment_path}.")
            seed_record["alignment"] = str(alignment_path)

    identities, paths, records = [], {}, []
    cache = saved_data = None
    requested_matcher = args.matcher
    args.matcher = requested_matcher or DEFAULT_MATCHER
    sources = image_paths(prepared) if prepared is not None else source_files(input_folder)
    if not sources:
        if initial_ply is None:
            raise RuntimeError("Put MOV, MP4, HEIC, HEIF, JPEG or PNG files from one room in the input folder.")
        reference = args.reuse_run or (seed_origin[0].name if seed_origin else None)
        if reference is None:
            raise RuntimeError("PLY training needs photos or videos and their camera poses. Add room images "
                               "or select a compatible saved training run with --reuse-run (-ReuseRun).")
        saved_data = saved_training_data(runs, reference)
        saved_run, saved_dataset, saved_report = saved_data
        if seed_calibration and seed_calibration != saved_report.get("calibration_run", saved_run.name):
            raise ValueError(f"The initial PLY uses the camera poses from {seed_calibration}. Select that training data.")
        saved_sizes = [image_size(path) for path in image_paths(saved_dataset / "images")]
        pixel_counts = [pixels_at(size, args.max_edge) for size in saved_sizes]
    elif prepared is not None:
        if seed_calibration:
            raise ValueError("This PLY uses saved camera poses. Use its original capture set or reuse its saved "
                             "training data with the PLY in an input folder by itself.")
        pixel_counts = [pixels_at(image_size(path), args.max_edge) for path in sources]
    else:
        print(f"Checking filenames and SHA-256 hashes for {len(sources)} source files...", flush=True)
        hints = {folder: capture_frames(folder) for folder in {path.parent for path in sources}}
        for path in sources:
            identity = source_identity(path, input_folder)
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.name in hints[path.parent]:
                identity["capture"] = hints[path.parent][path.name]
            identities.append(identity)
            paths[identity["name"]] = path
        if seed_calibration and args.reuse_run and args.reuse_run != seed_calibration:
            raise ValueError(f"The initial PLY uses the camera poses from {seed_calibration}.")
        reference = args.reuse_run or seed_calibration
        cache = compatible_calibration(
            runs, identities, args.video_fps, reference,
            matcher=requested_matcher if reference else args.matcher,
            pairing=None if args.pairing == "auto" else args.pairing,
        )
        if cache:
            args.matcher = cache[1].get("matcher", "sift")
        prior = cache[1]["sources"] if cache else []
        records = allocate_sources(identities, sources, prior)
        pixel_counts = []
        for record in records:
            path = paths[record["name"]]
            if record["kind"] == "video":
                size, duration = video_info(path)
                record.setdefault("count", max(1, round(duration * args.video_fps)))
            else:
                size = image_size(path)
            pixel_counts.extend([pixels_at(size, args.max_edge)] * record["count"])
    captures = {
        record["relative"]: (record["name"].rpartition("/")[0], record["capture"])
        for record in records if "capture" in record
    }

    exact = cache and len(cache[1]["sources"]) == len(identities)
    run = runs / (datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] + f"-{os.getpid()}")
    run.mkdir()
    print(f"Run folder: {run}", flush=True)
    inputs = [*sources, initial_ply] if initial_ply else sources
    (run / "inputs.txt").write_text("\n".join(map(str, inputs)) + "\n", encoding="utf-8")
    input_report = {"video_fps": args.video_fps, "sources": records}
    if prepared is not None:
        input_report["frames_path"] = str(prepared)
    if exact:
        input_report["calibration_run"] = cache[0].name
    if seed_record:
        input_report["initial_ply"] = seed_record
    atomic_json(run / "input-sources.json", input_report)

    if args.average_steps is not None:
        args.steps = total_steps(len(pixel_counts), args.average_steps)
    average_pixels = sum(pixel_counts) / len(pixel_counts)
    estimate = estimates_for(len(pixel_counts), average_pixels, args.steps, 0.3 * len(pixel_counts))
    if exact or saved_data:
        for stage in ("prepare", "features", "match", "mapper"):
            estimate[stage] = 0
    else:
        count = len(pixel_counts)
        possible = count * (count - 1) // 2
        expected_pairs = possible if pairing_mode(args.pairing, count) == "exhaustive" else min(possible, 25 * count)
        if args.matcher == "loma":
            estimate["features"] = max(1, count * 3.1)
        estimate["match"] = max(1, expected_pairs * (0.126 if args.matcher == "loma" else 0.006))
        if len(pixel_counts) >= 200 and args.mapper != "incremental":
            estimate["mapper"] = max(5, 460 * len(pixel_counts) / 1621)
    with (run / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        timeline = Timeline(estimate, log)
        image_mode = f"maximum edge {args.max_edge} px" if args.max_edge else "original image dimensions"
        if saved_data:
            write_status(log, f"Reusing saved training images and camera poses: {saved_dataset} ({image_mode}).")
        else:
            write_status(log, f"Image settings: {image_mode}, {args.video_fps:g} frames/s for videos.")
        if seed_record:
            write_status(log, f"Initial PLY: {initial_ply} ({seed_record['vertices']:,} vertices).")
        if captures:
            directions = sum(hint["direction"] is not None for _, hint in captures.values())
            recordings = len({folder for folder, _ in captures.values()})
            write_status(log, f"Wi-Fi capture log: {len(captures):,} frames from {recordings} recording(s), "
                              f"{directions:,} with direction hints for matching.")
        write_status(log, "ETA is approximate and updates as work completes.")
        calibration_run = run
        shared_alignment = None
        with TemporaryDirectory(prefix="work-", dir=run) as scratch:
            work = Path(scratch)
            frames = prepared
            if saved_data:
                calibration_run = runs / saved_report.get("calibration_run", saved_run.name)
                dataset = saved_dataset
                registered = len(saved_sizes)
                total = saved_report.get("input_frames", registered)
                components = saved_report.get("sparse_models", [registered])
                mapper = saved_report.get("mapper", "saved")
                args.matcher = saved_report.get("matcher", "sift")
                mode = saved_report.get("pairing", "legacy")
                shared_alignment = saved_run / "alignment.json"
                if not shared_alignment.is_file():
                    shared_alignment = dataset.parent / "alignment.json"
            elif exact:
                calibration_run, saved = cache
                model = calibration_run / saved["model"]
                registered = saved["registered_frames"]
                components = saved["sparse_models"]
                mapper = saved.get("mapper", "incremental")
                mode = saved.get("pairing", "legacy")
                write_status(log, f"Matching filenames and SHA-256: {calibration_run.name}")
                write_status(log, f"Reusing {registered} solved camera poses and existing feature matches.")
                total = sum(record["count"] for record in records)
                shared_alignment = calibration_run / "alignment.json"
            else:
                new_names = None
                if prepared is None:
                    frames = work / "frames"
                    timeline.start("prepare")
                    stage_pose_images(frames, records, paths, args.video_fps, timeline, cache)
                    if cache:
                        base_run, saved = cache
                        previous = {record["name"] for record in saved["sources"]}
                        added = [record for record in records if record["name"] not in previous]
                        new_names = {name for record in added for name in staged_image_names(record)}
                        write_status(log, f"Continuing from {base_run.name} with {len(added)} new source files.")
                    timeline.finish()
                files = image_paths(frames)
                total = len(files)
                if total < 3:
                    raise RuntimeError(f"Only {total} images. Capture overlapping views from different positions.")
                model, registered, components, mapper, mode = run_colmap(
                    frames, files, args, run, work, timeline, cache, new_names, captures,
                )
                calibration = {
                    **input_report, "version": 1, "pose_edge": POSE_EDGE if prepared is None else args.max_edge or 1600,
                    "frames": "frames", "model": model.relative_to(run).as_posix(), "database": "database.db",
                    "registered_frames": registered, "sparse_models": components,
                    "mapper": mapper, "matcher": args.matcher, "pairing": mode,
                }
                atomic_json(run / "calibration.json", calibration)
                atomic_json(run / "input-sources.json", input_report)

            if args.average_steps is not None:
                args.steps = total_steps(registered, args.average_steps)
            timeline.estimates["train"] = estimates_for(registered, average_pixels, args.steps, 0)["train"]
            write_status(log, f"Training: {registered:,} registered images, {args.steps:,} steps "
                              f"({args.steps / registered:.2f} per image on average), up to {args.max_splats:,} splats.")

            previous_dataset = dataset_from_run(runs, calibration_run, args.max_edge) if exact else None
            if saved_data:
                if args.max_edge and max(map(max, saved_sizes)) > args.max_edge:
                    timeline.start("undistort")
                    dataset = make_dataset(dataset / "images", dataset / "sparse", run, args.max_edge, timeline)
                    timeline.finish()
                else:
                    timeline.estimates["undistort"] = 0
            elif previous_dataset is not None:
                timeline.estimates["undistort"] = 0
                dataset = previous_dataset
                write_status(log, f"Reusing lens-corrected training images: {dataset}")
            else:
                native = prepared is None and (args.max_edge == 0 or args.max_edge > POSE_EDGE)
                if frames is None and not native:
                    frames = work / "frames"
                    timeline.estimates["prepare"] = 0.3 * total
                    timeline.start("prepare")
                    stage_pose_images(frames, records, paths, args.video_fps, timeline, cache)
                    timeline.finish()
                timeline.start("undistort")
                if native:
                    write_status(log, "Preparing native pixels for registered images using the saved camera poses.")
                    frames, model = native_images(frames, model, work, records, paths, args.video_fps, timeline)
                    timeline.fraction = None
                    timeline.detail = ""
                dataset = make_dataset(frames, model, run, args.max_edge, timeline)
                timeline.finish()
                atomic_json(run / "dataset-ready.json", {
                    "calibration_run": calibration_run.name, "requested_long_edge": args.max_edge or None,
                })

            orient_dataset(dataset, run, work, timeline, shared_alignment)

        brush = OPTIMIZED_BRUSH if OPTIMIZED_BRUSH.is_file() else BRUSH
        scratch = TemporaryDirectory(prefix="training-", dir=run) if initial_ply is not None else nullcontext()
        with scratch as training_folder:
            training_dataset, staged_ply = dataset, None
            if initial_ply is not None:
                staged_ply = Path(training_folder) / "init.ply"
                if seed_alignment is None:
                    shutil.copyfile(initial_ply, staged_ply)
                else:
                    write_status(log, "Restoring the initial PLY to camera coordinates.")
                    restore_training_ply(initial_ply, staged_ply, seed_alignment)
                current = initial_ply.stat()
                if (current.st_size, current.st_mtime_ns) != (seed_stamp.st_size, seed_stamp.st_mtime_ns):
                    raise RuntimeError(f"Input changed while preparing training: {initial_ply}")
                if args.trainer == "brush":
                    training_dataset = staged_ply.parent
                    stage_brush_dataset(dataset, training_dataset)
            result, count, train_edge = train(training_dataset, run, args, timeline, brush, trainer=args.trainer,
                                              initial_ply=staged_ply)
        summary = {
            "input_frames": total, "registered_frames": registered, "sparse_models": components,
            "training_steps": args.steps, "average_steps_per_image": args.steps / registered,
            "requested_average_steps": args.average_steps,
            "maximum_long_edge": train_edge, "requested_long_edge": args.max_edge or None,
            "feature_extraction_long_edge": POSE_EDGE if prepared is None else args.max_edge or 1600,
            "maximum_splats": args.max_splats, "output_splats": count, "output_bytes": result.stat().st_size,
            "trainer": args.trainer,
            "calibration_run": calibration_run.name, "dataset_run": dataset.parent.name,
            "mapper": mapper, "matcher": args.matcher, "pairing": mode, "alignment": "alignment.json",
            "elapsed_seconds": round(monotonic() - timeline.started, 2), "stage_seconds": timeline.seconds,
            "output": str(result),
        }
        if args.trainer == "brush":
            summary["brush_build"] = brush.parent.name
        if seed_record:
            summary["initial_ply"] = seed_record
        if saved_data:
            summary["saved_training_run"] = saved_run.name
            summary["feature_extraction_long_edge"] = saved_report.get("feature_extraction_long_edge")
        atomic_json(run / "result.json", summary)
        write_status(log, f"Gaussian splats: {count:,} ({result.stat().st_size / 1_000_000:.1f} MB)")
        write_status(log, f"Ready to import into Blender 5.3: {result}")


def total_steps(image_count, average):
    count = max(1, round(image_count * average))
    if count > 4_294_967_295:
        raise ValueError("Requested step count exceeds the trainer's integer range.")
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaussian splats from photos and video, optionally using an initial PLY.")
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--frames-path", type=Path)
    parser.add_argument("--init-ply", type=Path, help="Initial point-cloud or Gaussian-splat PLY (defaults to the PLY in the input folder).")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max-splats", type=int, required=True)
    parser.add_argument("--max-edge", type=int, required=True)
    parser.add_argument("--video-fps", type=float, required=True)
    parser.add_argument("--average-steps", type=float)
    parser.add_argument("--reuse-run")
    parser.add_argument("--matcher", choices=MATCHERS, help="New calibrations default to LoMa-B128.")
    parser.add_argument("--pairing", choices=("auto", "exhaustive", "selective"), default="auto")
    parser.add_argument("--mapper", choices=("auto", "incremental", "global"), default="auto")
    parser.add_argument("--trainer", choices=TRAINERS, default="brush")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    try:
        options = parser.parse_args()
        if options.average_steps is not None and (
            not math.isfinite(options.average_steps) or options.average_steps <= 0
        ):
            raise ValueError("Average steps per image must be a positive finite number.")
        if options.reuse_run and not RUN_NAME.fullmatch(options.reuse_run):
            raise ValueError("Use a run folder name for --reuse-run.")
        main(options)
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
