"""Train the independent V-JEPA 2.1 learned-pooling LeWM on Cube."""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from cube_video_context_dataset import CubeVideoContextDataset, make_episode_split
from module import SIGReg
from utils import SaveCkptCallback


def vjepa_lewm_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    embeddings = output["emb"]
    action_embeddings = output["act_emb"]
    context = embeddings[:, : cfg.history_size]
    context_actions = action_embeddings[:, : cfg.history_size]
    targets = embeddings[:, cfg.num_preds :]
    predictions = self.model.predict(context, context_actions)
    output["pred_loss"] = (predictions - targets).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(embeddings.transpose(0, 1))
    output["loss"] = (
        output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
    )
    self.log_dict(
        {
            f"{stage}/{key}": value.detach()
            for key, value in output.items()
            if "loss" in key
        },
        on_step=True,
        sync_dist=True,
    )
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="vjepa2_cube")
def run(cfg):
    torch.manual_seed(cfg.seed)
    dataset_path = Path(cfg.data.path)
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    with h5py.File(dataset_path, "r") as dataset:
        num_episodes = len(dataset["ep_len"])
        actions = np.asarray(dataset["action"][:], dtype=np.float32)
    valid_actions = actions[~np.isnan(actions).any(axis=1)]
    action_mean, action_std = valid_actions.mean(0), valid_actions.std(0)
    train_episodes, val_episodes = make_episode_split(
        num_episodes, cfg.train_split, cfg.seed
    )
    dataset_kwargs = dict(
        path=dataset_path,
        context_frames=cfg.data.context_frames,
        num_steps=cfg.data.num_steps,
        frameskip=cfg.data.frameskip,
        sample_stride=cfg.data.sample_stride,
        action_mean=action_mean,
        action_std=action_std,
    )
    train_set = CubeVideoContextDataset(
        episode_indices=train_episodes, **dataset_kwargs
    )
    val_set = CubeVideoContextDataset(episode_indices=val_episodes, **dataset_kwargs)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=generator
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )
    print(
        f"Cube V-JEPA samples: train={len(train_set):,} from {len(train_episodes)} "
        f"episodes; val={len(val_set):,} from {len(val_episodes)} episodes"
    )

    model = hydra.utils.instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(vjepa_lewm_forward, cfg=cfg),
        optim=optimizers,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.subdir
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    logger = WandbLogger(**cfg.wandb.config) if cfg.wandb.enabled else None
    callback = SaveCkptCallback(cfg.output_model_name, cfg.model, epoch_interval=1)
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == "__main__":
    run()
