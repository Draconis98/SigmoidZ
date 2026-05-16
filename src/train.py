from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import time

from safetensors.torch import save_model
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
from transformers import get_cosine_schedule_with_warmup

from config import load_config, to_dict
from data import build_batcher
from model import CausalLM
from utils import amp_dtype, is_ddp, rank_info, resolve_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--keep_checkpoints", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def setup_distributed(device: torch.device) -> tuple[int, int, int, torch.device]:
    rank, local_rank, world_size = rank_info()
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device


def checkpoint_dir(out_dir: Path) -> Path:
    return out_dir / "checkpoints"


def latest_checkpoint(out_dir: Path) -> Path | None:
    paths = sorted(checkpoint_dir(out_dir).glob("step_*.pt"))
    return paths[-1] if paths else None


def rotate_checkpoints(out_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    paths = sorted(checkpoint_dir(out_dir).glob("step_*.pt"))
    for path in paths[:-keep]:
        path.unlink()


def save_checkpoint(
    out_dir: Path,
    step: int,
    raw_model: CausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    cfg_dict: dict,
    keep: int,
) -> None:
    path = checkpoint_dir(out_dir) / f"step_{step:08d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg_dict,
        },
        tmp_path,
    )
    tmp_path.replace(path)
    rotate_checkpoints(out_dir, keep)


def resolve_resume_path(out_dir: Path, resume_from: str | None) -> Path | None:
    if resume_from is None:
        return None
    if resume_from == "latest":
        return latest_checkpoint(out_dir)
    return Path(resume_from)


def load_checkpoint(
    path: Path,
    raw_model: CausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["step"])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.preset)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.device is not None:
        cfg.train.device = args.device
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.save_interval is not None:
        cfg.train.save_interval = args.save_interval
    if args.keep_checkpoints is not None:
        cfg.train.keep_checkpoints = args.keep_checkpoints
    if args.wandb:
        cfg.train.wandb_enabled = True
    if args.resume is not None:
        cfg.train.resume_from = args.resume

    device = resolve_device(cfg.train.device)
    rank, local_rank, world_size, device = setup_distributed(device)
    seed_everything(cfg.train.seed + rank)

    out_dir = Path(cfg.train.out_dir)
    batcher, vocab_size = build_batcher(cfg.train, cfg.model, device, rank, world_size)
    cfg.model.vocab_size = vocab_size
    model = CausalLM(cfg.model).to(device)
    raw_model = model
    if is_ddp():
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        raw_model = model.module

    parameter_count = raw_model.parameter_count()
    global_tokens_per_step = cfg.train.batch_size * cfg.model.context_length * world_size
    if cfg.train.max_steps is None:
        assert cfg.train.token_budget_multiplier is not None, "max_steps=None requires token_budget_multiplier"
        target_tokens = int(math.ceil(parameter_count * cfg.train.token_budget_multiplier))
        cfg.train.max_steps = int(math.ceil(target_tokens / global_tokens_per_step))
    assert cfg.train.max_steps > 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.train.warmup_steps, cfg.train.max_steps)
    dtype = amp_dtype(cfg.train.dtype)
    use_amp = dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)

    start_step = 0
    resume_path = resolve_resume_path(out_dir, cfg.train.resume_from)
    if resume_path is not None:
        assert resume_path.exists(), f"Checkpoint not found: {resume_path}"
        start_step = load_checkpoint(resume_path, raw_model, optimizer, scheduler, scaler, device)

    wandb_run = None
    log_file = None
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(out_dir / "config.json", to_dict(cfg))
        log_file = (out_dir / "train.log").open("a")
        model_text = str(raw_model)
        (out_dir / "model.txt").write_text(model_text + "\n")
        print(
            f"params={parameter_count:,} max_steps={cfg.train.max_steps:,} "
            f"target_tokens={cfg.train.max_steps * global_tokens_per_step:,} "
            f"context={cfg.model.context_length} device={device} world_size={world_size}"
        )
        print(model_text)
        if resume_path is not None:
            print(f"resumed_from={resume_path} start_step={start_step}")
        if cfg.train.wandb_enabled:
            import wandb

            wandb_run = wandb.init(
                project=cfg.train.wandb_project,
                entity=cfg.train.wandb_entity,
                name=cfg.train.wandb_name,
                tags=cfg.train.wandb_tags,
                config=to_dict(cfg),
                resume="allow",
            )
            if cfg.train.wandb_watch_model:
                wandb_run.watch(raw_model, log="gradients", log_freq=max(cfg.train.log_interval, 1))

    model.train()
    start = time.time()
    iterator = trange(start_step + 1, cfg.train.max_steps + 1, disable=rank != 0)
    for step in iterator:
        x, y = batcher.next()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            _, loss = model(x, y)
        assert loss is not None
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if rank == 0 and step % cfg.train.log_interval == 0:
            elapsed = max(time.time() - start, 1e-6)
            interval_steps = step - start_step
            tokens_per_second = interval_steps * global_tokens_per_step / elapsed
            lr = scheduler.get_last_lr()[0]
            loss_value = loss.item()
            ppl = math.exp(min(loss_value, 20.0))
            tokens_seen = step * global_tokens_per_step
            log_line = (
                f"step={step} loss={loss_value:.4f} ppl={ppl:.2f} lr={lr:.2e} "
                f"grad_norm={float(grad_norm.detach().cpu()):.3f} tokens={tokens_seen} tok/s={tokens_per_second:.0f}"
            )
            print(log_line)
            assert log_file is not None
            log_file.write(log_line + "\n")
            log_file.flush()
            iterator.set_description(f"loss={loss_value:.4f} ppl={ppl:.2f} lr={lr:.2e}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss_value,
                        "train/ppl": ppl,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm.detach().cpu()),
                        "train/tokens_per_second": tokens_per_second,
                        "train/tokens": tokens_seen,
                    },
                    step=step,
                )

        if rank == 0 and cfg.train.save_interval and step % cfg.train.save_interval == 0:
            save_checkpoint(out_dir, step, raw_model, optimizer, scheduler, scaler, to_dict(cfg), cfg.train.keep_checkpoints)

    if rank == 0:
        save_checkpoint(out_dir, cfg.train.max_steps, raw_model, optimizer, scheduler, scaler, to_dict(cfg), cfg.train.keep_checkpoints)
        save_model(raw_model, out_dir / "model.safetensors")
        if wandb_run is not None:
            wandb_run.summary["parameters"] = parameter_count
            wandb_run.summary["tokens"] = cfg.train.max_steps * global_tokens_per_step
            wandb_run.finish()
        if log_file is not None:
            log_file.close()
    if is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
