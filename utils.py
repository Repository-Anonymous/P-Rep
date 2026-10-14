import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


class PeriodicCubeEvalCallback(Callback):
    """Evaluate live training weights and upload a three-episode video to W&B.

    Evaluation runs inside the same training job on global rank zero. DDP peers
    wait at a barrier, so neither the model nor optimizer is moved off GPU and
    training never has to be resumed from a checkpoint.
    """

    def __init__(
        self,
        run_name,
        model_cfg,
        output_dir,
        every_n_epochs=2,
        episodes=(3053, 4053, 5053),
        start_steps=(0, 0, 0),
        goal_offset_steps=200,
        eval_budget=200,
    ):
        super().__init__()
        self.run_name = run_name
        self.model_cfg = model_cfg
        self.output_dir = Path(output_dir)
        self.every_n_epochs = int(every_n_epochs)
        self.episodes = [int(x) for x in episodes]
        self.start_steps = [int(x) for x in start_steps]
        self.goal_offset_steps = int(goal_offset_steps)
        self.eval_budget = int(eval_budget)

        if len(self.episodes) != 3 or len(self.start_steps) != 3:
            raise ValueError("Periodic Cube evaluation requires exactly three episodes/start steps")

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero and trainer.logger is not None:
            experiment = getattr(trainer.logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "define_metric"):
                experiment.define_metric("eval/epoch")
                experiment.define_metric("eval/*", step_metric="eval/epoch")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        should_eval = epoch % self.every_n_epochs == 0 or epoch == trainer.max_epochs

        if should_eval and trainer.is_global_zero:
            self._evaluate_and_log(trainer, pl_module.model, epoch)

        if should_eval and trainer.world_size > 1:
            trainer.strategy.barrier("periodic_cube_eval")

    def _evaluate_and_log(self, trainer, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained

        epoch_dir = self.output_dir / "eval" / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = f"eval_epoch_{epoch}.pt"
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.model_cfg,
            filename=checkpoint_name,
        )
        policy_path = f"{self.run_name}/{checkpoint_name}"

        command = [
            sys.executable,
            "eval.py",
            "--config-name=cube.yaml",
            f"policy={policy_path}",
            "eval.num_eval=3",
            f"eval.episodes_idx={self.episodes}",
            f"eval.start_steps={self.start_steps}",
            f"eval.goal_offset_steps={self.goal_offset_steps}",
            f"eval.eval_budget={self.eval_budget}",
            "world.terminate_at_goal=false",
            f"output_dir={epoch_dir}",
            f"hydra.run.dir={epoch_dir}",
        ]
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        with (epoch_dir / "eval.log").open("w") as log_file:
            subprocess.run(
                command,
                cwd=Path(__file__).resolve().parent,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

        videos = [epoch_dir / f"env_{index}.mp4" for index in range(3)]
        missing = [str(path) for path in videos if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cube evaluation did not produce videos: {missing}")

        triptych = epoch_dir / "cube_episodes_3053_4053_5053_triptych.mp4"
        ffmpeg = subprocess.check_output(
            [sys.executable, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
            text=True,
        ).strip()
        subprocess.run(
            [
                ffmpeg,
                "-loglevel", "error", "-y",
                "-i", str(videos[0]),
                "-i", str(videos[1]),
                "-i", str(videos[2]),
                "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
                "-map", "[v]", "-an", "-r", "15", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(triptych),
            ],
            check=True,
        )

        with (epoch_dir / "metrics.json").open() as f:
            metrics = json.load(f)
        successes = metrics.get("episode_successes", [False, False, False])

        logger = trainer.logger
        experiment = getattr(logger, "experiment", None) if logger is not None else None
        if experiment is not None and hasattr(experiment, "log"):
            import wandb

            payload = {
                "eval/epoch": epoch,
                "eval/global_step": trainer.global_step,
                "eval/success_rate": float(metrics["success_rate"]),
                "eval/evaluation_time_seconds": float(metrics["evaluation_time_seconds"]),
                "eval/episode_3053_success": int(bool(successes[0])),
                "eval/episode_4053_success": int(bool(successes[1])),
                "eval/episode_5053_success": int(bool(successes[2])),
                "eval/cube_triptych": wandb.Video(
                    str(triptych),
                    format="mp4",
                    caption=f"epoch {epoch}: episodes 3053 | 4053 | 5053",
                ),
            }
            experiment.log(payload)
