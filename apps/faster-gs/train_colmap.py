"""Run the NeRFICG Faster-GS trainer on the project's undistorted COLMAP data."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData, PlyElement
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "nerficg/src"))
import Framework
from Datasets.Colmap import CustomDataset
from Datasets.utils import BasicPointCloud
from Methods.Base.utils import pre_training_callback, training_callback
from Methods.FasterGS import MODEL, RENDERER, TRAINING_INSTANCE


class PrefetchSampler:
    """Keep a bounded queue of CPU-decoded images in the trainer's original shuffle order."""

    def __init__(self, sampler, dataset, executor):
        self.indices = sampler.id_sampler
        self.dataset = dataset
        self.executor = executor
        self.device = Framework.config.GLOBAL.DEFAULT_DEVICE
        self.pending = deque()
        self.epoch = iter(())
        self.transforms = torch.as_tensor(
            np.stack([view.w2c_numpy for view in dataset]), dtype=torch.float32, device=self.device,
        )
        self.positions = torch.as_tensor(
            np.stack([view.position_numpy for view in dataset]), dtype=torch.float32, device=self.device,
        )

    @staticmethod
    def load_rgb(view):
        # Load the same float pixels as View.rgb before the transfer to CUDA.
        return view._rgb.image.pin_memory()

    def enqueue(self):
        index = next(self.epoch, None)
        if index is not None:
            self.pending.append((index, self.executor.submit(self.load_rgb, self.dataset[index])))

    def get(self, dataset):
        if not self.pending:
            # Transfer the whole permutation once rather than synchronizing for each .item().
            self.epoch = iter(self.indices.get(len(dataset)).tolist())
            for _ in range(6):
                self.enqueue()
        index, rgb = self.pending.popleft()
        self.enqueue()
        view = dataset[index]
        return {"view": SimpleNamespace(
            camera=view.camera, w2c=self.transforms[index], position=self.positions[index],
            rgb=rgb.result().to(self.device, non_blocking=True), alpha=view.alpha,
        )}


class RoomTrainer(TRAINING_INSTANCE):
    def __init__(self, model, renderer, checkpoints, interval, executor, initial_ply=None):
        super().__init__(model=model, renderer=renderer)
        self.checkpoints = checkpoints
        self.interval = interval
        self.executor = executor
        self.initial_ply = initial_ply
        self.training_seconds = 0.0
        self.export_seconds = 0.0

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, iteration, dataset):
        super().create_sampler(iteration, dataset)
        if self.executor is not None:
            self.train_sampler = PrefetchSampler(self.train_sampler, dataset, self.executor)

    def initialize_gaussians(self, dataset):
        if self.initial_ply is None:
            super().initialize_gaussians(dataset)
            return
        vertices = PlyData.read(self.initial_ply)['vertex'].data
        if len(vertices) == 0 or len(vertices) > self.ADAPTIVE_MAX_PRIMITIVES:
            raise ValueError('The initial PLY must have between one vertex and the configured splat limit.')
        print(f"Using initial PLY with {len(vertices):,} vertices: {self.initial_ply}", flush=True)
        fields = set(vertices.dtype.names)
        if 'opacity' in fields:
            self.model.gaussians.initialize_from_ply(vertices)
            return

        positions = np.column_stack([vertices[name] for name in ('x', 'y', 'z')]).astype(np.float32)
        colors = None
        for channels in (('red', 'green', 'blue'), ('r', 'g', 'b')):
            if fields.issuperset(channels):
                rgb = [vertices[name] for name in channels]
                scales = [np.iinfo(channel.dtype).max if channel.dtype.kind in 'iu' else 1 for channel in rgb]
                colors = torch.from_numpy(np.column_stack(rgb).astype(np.float32) / np.asarray(scales, dtype=np.float32))
                break
        if not np.isfinite(positions).all() or (colors is not None and not colors.isfinite().all()):
            raise ValueError(f"Non-finite points or colors in the initial PLY: {self.initial_ply}")
        dataset.point_cloud = BasicPointCloud(torch.from_numpy(positions), colors)
        super().initialize_gaussians(dataset)

    @pre_training_callback(priority=-100)
    def start_timer(self, *_):
        torch.cuda.synchronize()
        self.block_started = perf_counter()

    @training_callback(priority=-100)
    def report_progress(self, iteration, _):
        step = iteration + 1
        save = step < self.NUM_ITERATIONS and step % self.interval == 0
        if step % 100 and step != self.NUM_ITERATIONS and not save:
            return
        torch.cuda.synchronize()
        self.training_seconds += perf_counter() - self.block_started
        count = self.model.gaussians.means.shape[0]
        print(f"Train iter {step}, {count} splats, {self.training_seconds:.3f}s training time", flush=True)
        if save:
            self.export(step)
        self.block_started = perf_counter()

    def export(self, step):
        started = perf_counter()
        digits = len(str(self.NUM_ITERATIONS))
        path = self.checkpoints / f"room_{step:0{digits}}.ply"
        temp = path.with_suffix(".ply.tmp")
        try:
            data = self.model.get_ply_dict()
            element = PlyElement.describe(data["vertex"], "vertex")
            PlyData([element], byte_order="<", comments=data["comments"]).write(temp)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        elapsed = perf_counter() - started
        self.export_seconds += elapsed
        print(f"Export iter {step}, {elapsed:.3f}s", flush=True)


def train(options):
    started = perf_counter()
    torch.set_num_threads(1)
    run = options.checkpoints.parent
    config = {
        "GLOBAL": {"LOG_LEVEL": 2, "METHOD_TYPE": "FasterGS", "DATASET_TYPE": "Colmap",
                   "GPU_INDICES": [0], "RANDOM_SEED": 42, "FILTER_WARNINGS": True},
        "MODEL": {"PPISP": {"USE": False}},
        "RENDERER": {"PROPER_ANTIALIASING": False},
        "DATASET": {"PATH": str(options.dataset)},
        "TRAINING": {
            "MODEL_NAME": "room", "NUM_ITERATIONS": options.steps,
            "DENSIFICATION_END_ITERATION": options.growth_stop,
            "ADAPTIVE_MAX_PRIMITIVES": options.max_splats,
            "ROBUST_CAMERA_EXTENT": options.robust_camera_extent,
            "DATA": {"PRELOADING_LEVEL": 0},
            "BACKUP": {"FINAL_CHECKPOINT": False, "RENDER_TESTSET": False},
            "GUI": {"ACTIVATE": False}, "TIMING": {"ACTIVATE": False},
            "WANDB": {"ACTIVATE": False},
        },
    }
    with TemporaryDirectory(prefix="config-", dir=run) as scratch:
        config_path = Path(scratch) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="ascii")
        Framework.setup(config_path=config_path, require_custom_config=True)
        dataset = CustomDataset(str(options.dataset))
        model = MODEL(name="room").build()
        model.output_directory = run
        executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="room-rgb") if options.prefetch else nullcontext()
        with executor as pool:
            trainer = RoomTrainer(model, RENDERER(model), options.checkpoints, options.interval, pool, options.init_ply)
            trainer.run(dataset)
            if model.num_iterations_trained != options.steps:
                raise RuntimeError(f"Faster-GS stopped at step {model.num_iterations_trained} of {options.steps}.")
            trainer.export(options.steps)
        Framework.teardown()
    (run / "faster-gs-timing.json").write_text(json.dumps({
        "training_seconds": round(trainer.training_seconds, 3),
        "export_seconds": round(trainer.export_seconds, 3),
        "worker_seconds": round(perf_counter() - started, 3),
        "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "robust_camera_extent": options.robust_camera_extent,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max-splats", type=int, required=True)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--growth-stop", type=int, required=True)
    parser.add_argument("--init-ply", type=Path)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--robust-camera-extent", action="store_true")
    train(parser.parse_args())
