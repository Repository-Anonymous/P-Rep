"""Train GCBC or GCIQL heads while keeping LeWM encoder/projector frozen."""

import argparse
import json
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .data import Native224GoalDataset, move_batch
from .models import gciql, policy, state_fusion
from .models.frozen_lewm import LeWMFrozenEncoder


def _atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _to_jsonable(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _default_state_path(dataset_dir, dataset):
    return Path(dataset_dir) / ".lewm-npy-cache" / dataset / "state28.npy"


def build_model(args):
    encoder = LeWMFrozenEncoder(source_checkpoint=args.lewm_checkpoint)
    if args.algorithm == "gcbc":
        if args.state_fusion:
            return state_fusion.FrozenStateVisualGCBC(
                encoder, seed=args.seed, goal_state_mode=args.goal_state_mode
            )
        return policy.FrozenLeWMGCBC(encoder, seed=args.seed)
    return gciql.FrozenLeWMGCIQL(
        encoder,
        state_fusion=args.state_fusion,
        goal_state_mode=args.goal_state_mode,
        seed=args.seed,
        discount=args.discount,
        tau=args.tau,
        expectile=args.expectile,
        actor_loss=args.actor_loss,
        alpha=args.alpha,
    )


def _sample(dataset, algorithm, batch_size, discount):
    if algorithm == "gcbc":
        return dataset.sample_gcbc(batch_size)
    return dataset.sample_gciql(batch_size, discount=discount)


def _loss(model, algorithm, batch):
    if algorithm == "gcbc":
        states = batch.get("states")
        if states is None:
            mean = model(batch["current"], batch["goal"])
        else:
            mean = model(batch["current"], batch["goal"], states["current"], states["goal"])
        value = policy.unit_gaussian_nll(mean.float(), batch["action"].float())
        return value, {"loss": value.detach(), "actor_loss": value.detach()}
    return model.compute_loss(batch)


def _checkpoint_payload(model, optimizer, step, args, train_data):
    return {
        "format": "lewm-ogbench-policy-v1",
        "algorithm": args.algorithm,
        "step": int(step),
        "source_lewm_checkpoint": str(Path(args.lewm_checkpoint).expanduser().resolve()),
        "state_fusion": bool(args.state_fusion),
        "goal_state_mode": args.goal_state_mode,
        "downstream_model": model.downstream_state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "data_rng": train_data.rng.get_state(),
        "config": _to_jsonable(args),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.state_fusion and not args.dataset.startswith("visual-cube-single-"):
        raise ValueError("The linked state28 fusion interface applies only to visual-cube-single")
    state_path = args.state_path
    val_state_path = args.val_state_path
    if args.state_fusion:
        state_path = state_path or _default_state_path(args.dataset_dir, args.dataset)
        val_state_path = val_state_path or _default_state_path(args.dataset_dir, args.dataset + "-val")
    train_data = Native224GoalDataset(args.dataset, args.dataset_dir, state_path, seed=args.seed)
    val_data = Native224GoalDataset(args.dataset + "-val", args.dataset_dir, val_state_path, seed=args.seed + 1)

    device = torch.device(args.device)
    model = build_model(args).to(device).train()
    if args.algorithm == "gcbc":
        optimizer = policy.make_optimizer(model, lr=args.lr)
    else:
        optimizer = gciql.make_optimizer(model, lr=args.lr)
    encoder_parameter_ids = {id(parameter) for parameter in model.encoder.parameters()}
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if encoder_parameter_ids & optimizer_parameter_ids:
        raise RuntimeError("Frozen LeWM parameters entered the policy optimizer")
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("LeWM encoder/projector must be frozen")

    start_step = 0
    resume = output / "training-state.pt"
    if resume.is_file() and not args.resume:
        raise FileExistsError(f"{resume} already exists; pass --resume or choose a new output directory")
    if args.resume and resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["algorithm"] != args.algorithm or state["state_fusion"] != args.state_fusion:
            raise ValueError("Resume checkpoint algorithm/interface differs")
        prior = state.get("config", {})
        for key in ("dataset", "goal_state_mode", "actor_loss"):
            if str(prior.get(key)) != str(getattr(args, key)):
                raise ValueError(f"Resume setting differs for {key}")
        if Path(state["source_lewm_checkpoint"]).resolve() != Path(args.lewm_checkpoint).resolve():
            raise ValueError("Resume LeWM checkpoint differs")
        model.load_downstream_state_dict(state["downstream_model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        train_data.rng.set_state(state["data_rng"])

    config = _to_jsonable(args)
    config.update(
        interface="frozen LeWM encoder+projector followed by trainable downstream heads",
        frozen_lewm=True,
        trainable_scope="GCBC actor only" if args.algorithm == "gcbc" else "GCIQL actor, value, Q1 and Q2 only",
        input_resolution=224,
        image_source="renderer-native 224 RGB; no resize/upscale",
        action_domain="original OGBench normalized Box [-1,1]",
        state_path=None if state_path is None else str(Path(state_path).resolve()),
        val_state_path=None if val_state_path is None else str(Path(val_state_path).resolve()),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        frozen_parameters=sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
    )
    _json(output / "config.json", config)

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        run_id_path = output / "wandb-run-id.txt"
        run_id = run_id_path.read_text().strip() if run_id_path.is_file() else uuid.uuid4().hex[:12]
        run_id_path.write_text(run_id + "\n")
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"{args.algorithm}-{args.dataset}",
            id=run_id,
            resume="allow",
            mode=args.wandb_mode,
            dir=str(output),
            config=config,
        )
        _json(output / "wandb.json", {"id": wandb_run.id, "url": wandb_run.url})

    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    autocast_enabled = args.precision in ("bf16", "fp16") and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    started = time.monotonic()
    with (output / "metrics.jsonl").open("a", buffering=1) as metrics_file:
        for step in range(start_step + 1, args.steps + 1):
            raw_batch = _sample(train_data, args.algorithm, args.batch_size, args.discount)
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                loss, metrics = _loss(model, args.algorithm, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            if args.algorithm == "gciql":
                model.update_targets()
            if any(parameter.grad is not None for parameter in model.encoder.parameters()):
                raise RuntimeError("Frozen LeWM received gradients")

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    val_batch = move_batch(_sample(val_data, args.algorithm, args.val_batch_size, args.discount), device)
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                        _, val_metrics = _loss(model, args.algorithm, val_batch)
                model.train()
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    "grad_norm": float(grad_norm),
                    **{f"train/{key}": float(value) for key, value in metrics.items()},
                    **{f"val/{key}": float(value) for key, value in val_metrics.items()},
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if wandb_run is not None:
                    wandb_run.log(record, step=step)

            if step % args.save_every == 0 or step == args.steps:
                payload = _checkpoint_payload(model, optimizer, step, args, train_data)
                _atomic_save(resume, payload)
                _atomic_save(output / f"policy_step_{step}.pt", {
                    key: payload[key]
                    for key in (
                        "format", "algorithm", "step", "source_lewm_checkpoint", "state_fusion",
                        "goal_state_mode", "downstream_model", "config"
                    )
                })

    _json(output / "complete.json", {
        "completed": True,
        "algorithm": args.algorithm,
        "steps": args.steps,
        "frozen_lewm": True,
        "checkpoint": str(output / f"policy_step_{args.steps}.pt"),
    })
    if wandb_run is not None:
        wandb_run.finish()
    train_data.close()
    val_data.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("gcbc", "gciql"), required=True)
    parser.add_argument("--dataset", default="visual-cube-single-play-v0")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/home/coder/P-Rep/ogbench"))
    parser.add_argument(
        "--lewm-checkpoint",
        type=Path,
        default=Path("/home/coder/P-Rep/le-wm/.stable-wm/checkpoints/cube-lewm"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-fusion", action="store_true")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--val-state-path", type=Path)
    parser.add_argument("--goal-state-mode", choices=("object_xyz", "full"), default="object_xyz")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--actor-loss", choices=("ddpgbc", "awr"), default="ddpgbc")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-entity", default="visual-intelligence-laboratory")
    parser.add_argument("--wandb-project", default="P-Rep")
    parser.add_argument("--wandb-group", default="lewm-ogbench-policy")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
