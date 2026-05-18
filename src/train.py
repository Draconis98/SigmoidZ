from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_model
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
from transformers import get_cosine_schedule_with_warmup

from config import load_config, to_dict
from data import Batcher, build_batcher, build_validation_batcher
from model import CausalLM
from modules import DyT, SigmoidZNorm
from utils import (
    amp_dtype,
    is_ddp,
    rank_info,
    resolve_device,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--val_steps", type=int, default=None)
    parser.add_argument("--val_dataset_split", type=str, default=None)
    parser.add_argument("--val_text_file", type=str, default=None)
    parser.add_argument("--max_val_tokens", type=int, default=None)
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


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or name.endswith(".alpha")
            or name.endswith(".logit_bias")
            or ".norm." in name
            or name.endswith("_norm.weight")
        )
        if no_decay:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


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


METRIC_FIELDS = [
    "step",
    "tokens",
    "loss",
    "ppl",
    "lr",
    "grad_norm",
    "tokens_per_second",
    "parameters",
    "max_steps",
    "warmup_steps",
    "target_tokens",
    "context_length",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "world_size",
    "effective_batch_tokens",
    "token_budget_multiplier",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_max",
    "alpha_grad_abs_mean",
    "logit_bias_mean",
    "logit_bias_std",
    "logit_bias_abs_max",
    "norm_weight_mean",
    "norm_weight_std",
    "norm_bias_mean",
    "norm_bias_std",
    "param_norm",
]

VAL_METRIC_FIELDS = [
    "step",
    "tokens",
    "val_loss",
    "val_ppl",
    "val_steps",
    "val_dataset_split",
]


def _stack_scalars(tensors: list[torch.Tensor]) -> torch.Tensor | None:
    if not tensors:
        return None
    return torch.cat([x.detach().float().reshape(-1).cpu() for x in tensors])


def _mean_std_min_max(values: torch.Tensor | None, prefix: str) -> dict[str, float]:
    if values is None or values.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def collect_train_diagnostics(raw_model: CausalLM) -> dict[str, float]:
    alpha_values = []
    alpha_grads = []
    logit_bias_values = []
    norm_weight_values = []
    norm_bias_values = []
    for module in raw_model.modules():
        if isinstance(module, (SigmoidZNorm, DyT)):
            alpha_values.append(module.alpha)
            if module.alpha.grad is not None:
                alpha_grads.append(module.alpha.grad.detach().abs())
            norm_weight_values.append(module.weight)
            norm_bias_values.append(module.bias)
            if isinstance(module, SigmoidZNorm):
                logit_bias_values.append(module.logit_bias)

    stats = _mean_std_min_max(_stack_scalars(alpha_values), "alpha")
    alpha_grad = _stack_scalars(alpha_grads)
    stats["alpha_grad_abs_mean"] = (
        float(alpha_grad.mean().item()) if alpha_grad is not None else float("nan")
    )

    logit_bias = _stack_scalars(logit_bias_values)
    if logit_bias is None:
        stats.update(
            {
                "logit_bias_mean": float("nan"),
                "logit_bias_std": float("nan"),
                "logit_bias_abs_max": float("nan"),
            }
        )
    else:
        stats.update(
            {
                "logit_bias_mean": float(logit_bias.mean().item()),
                "logit_bias_std": float(logit_bias.std(unbiased=False).item()),
                "logit_bias_abs_max": float(logit_bias.abs().max().item()),
            }
        )

    norm_weight = _stack_scalars(norm_weight_values)
    norm_bias = _stack_scalars(norm_bias_values)
    stats["norm_weight_mean"] = float(norm_weight.mean().item()) if norm_weight is not None else float("nan")
    stats["norm_weight_std"] = float(norm_weight.std(unbiased=False).item()) if norm_weight is not None else float("nan")
    stats["norm_bias_mean"] = float(norm_bias.mean().item()) if norm_bias is not None else float("nan")
    stats["norm_bias_std"] = float(norm_bias.std(unbiased=False).item()) if norm_bias is not None else float("nan")

    param_norm_sq = 0.0
    for parameter in raw_model.parameters():
        param_norm_sq += float(parameter.detach().float().norm(2).cpu().item() ** 2)
    stats["param_norm"] = math.sqrt(param_norm_sq)
    return stats


def format_csv_row(
    row: dict[str, float | int | str | None],
    fields: list[str] = METRIC_FIELDS,
) -> str:
    values = []
    for field in fields:
        value = row[field]
        if isinstance(value, float):
            values.append(f"{value:.8g}")
        else:
            values.append(str(value))
    return ",".join(values)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    batcher: Batcher,
    steps: int,
    device: torch.device,
    dtype: torch.dtype | None,
    use_amp: bool,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = batcher.next()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(min(mean_loss, 20.0))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.preset)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.device is not None:
        cfg.train.device = args.device
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.save_interval is not None:
        cfg.train.save_interval = args.save_interval
    if args.val_interval is not None:
        cfg.train.val_interval = args.val_interval
    if args.val_steps is not None:
        cfg.train.val_steps = args.val_steps
    if args.val_dataset_split is not None:
        cfg.train.val_dataset_split = args.val_dataset_split
    if args.val_text_file is not None:
        cfg.train.val_text_file = args.val_text_file
    if args.max_val_tokens is not None:
        cfg.train.max_val_tokens = args.max_val_tokens
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
    val_batcher = None
    if rank == 0:
        val_built = build_validation_batcher(cfg.train, cfg.model, device)
        if val_built is not None:
            val_batcher, val_vocab_size = val_built
            assert val_vocab_size == cfg.model.vocab_size, (
                f"validation vocab size {val_vocab_size} does not match training vocab size "
                f"{cfg.model.vocab_size}"
            )
    model = CausalLM(cfg.model).to(device)
    raw_model = model
    if is_ddp():
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        raw_model = model.module

    parameter_count = raw_model.parameter_count()
    assert cfg.train.gradient_accumulation_steps > 0
    global_tokens_per_step = (
        cfg.train.batch_size
        * cfg.train.gradient_accumulation_steps
        * cfg.model.context_length
        * world_size
    )
    if cfg.train.max_steps is None:
        assert cfg.train.token_budget_multiplier is not None, (
            "max_steps=None requires token_budget_multiplier"
        )
        target_tokens = int(
            math.ceil(parameter_count * cfg.train.token_budget_multiplier)
        )
        cfg.train.max_steps = int(math.ceil(target_tokens / global_tokens_per_step))
    assert cfg.train.max_steps > 0
    if cfg.train.warmup_steps is None:
        assert cfg.train.warmup_ratio is not None, (
            "warmup_steps=None requires warmup_ratio"
        )
        cfg.train.warmup_steps = int(math.ceil(cfg.train.max_steps * cfg.train.warmup_ratio))
    assert cfg.train.warmup_steps >= 0

    optimizer = build_optimizer(model, cfg.train.learning_rate, cfg.train.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.train.warmup_steps, cfg.train.max_steps
    )
    dtype = amp_dtype(cfg.train.dtype)
    use_amp = dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)

    start_step = 0
    resume_path = resolve_resume_path(out_dir, cfg.train.resume_from)
    if resume_path is not None:
        assert resume_path.exists(), f"Checkpoint not found: {resume_path}"
        start_step = load_checkpoint(
            resume_path, raw_model, optimizer, scheduler, scaler, device
        )

    wandb_run = None
    log_file = None
    metrics_file = None
    val_metrics_file = None
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(out_dir / "config.json", to_dict(cfg))
        log_file = (out_dir / "train.log").open("a")
        metrics_path = out_dir / "train_metrics.csv"
        metrics_file = metrics_path.open("a")
        if metrics_path.stat().st_size == 0:
            metrics_file.write(",".join(METRIC_FIELDS) + "\n")
        if val_batcher is not None:
            val_metrics_path = out_dir / "val_metrics.csv"
            val_metrics_file = val_metrics_path.open("a")
            if val_metrics_path.stat().st_size == 0:
                val_metrics_file.write(",".join(VAL_METRIC_FIELDS) + "\n")
        model_text = str(raw_model)
        (out_dir / "model.txt").write_text(model_text + "\n")
        print(
            f"params={parameter_count:,} max_steps={cfg.train.max_steps:,} "
            f"warmup_steps={cfg.train.warmup_steps:,} "
            f"target_tokens={cfg.train.max_steps * global_tokens_per_step:,} "
            f"context={cfg.model.context_length} micro_batch={cfg.train.batch_size} "
            f"grad_accum={cfg.train.gradient_accumulation_steps} device={device} world_size={world_size}"
        )
        if val_batcher is not None:
            print(
                f"validation interval={cfg.train.val_interval} steps={cfg.train.val_steps} "
                f"split={cfg.train.val_dataset_split}"
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
            wandb.define_metric("train/tokens")
            wandb.define_metric("train/loss", step_metric="train/tokens")
            wandb.define_metric("train/ppl", step_metric="train/tokens")
            wandb.define_metric("train/lr", step_metric="train/tokens")
            wandb.define_metric("train/grad_norm", step_metric="train/tokens")
            wandb.define_metric("train/tokens_per_second", step_metric="train/tokens")
            wandb.define_metric("train/param_norm", step_metric="train/tokens")
            wandb.define_metric("val/loss", step_metric="train/tokens")
            wandb.define_metric("val/ppl", step_metric="train/tokens")
            wandb.define_metric("sigmoidz/*", step_metric="train/tokens")
            wandb.define_metric("hparams/*", step_metric="train/tokens")
            if cfg.train.wandb_watch_model:
                wandb_run.watch(
                    raw_model, log="gradients", log_freq=max(cfg.train.log_interval, 1)
                )

    model.train()
    start = time.time()
    iterator = trange(start_step + 1, cfg.train.max_steps + 1, disable=rank != 0)
    for step in iterator:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro_step in range(cfg.train.gradient_accumulation_steps):
            x, y = batcher.next()
            sync_context = (
                model.no_sync()
                if is_ddp()
                and micro_step < cfg.train.gradient_accumulation_steps - 1
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    _, loss = model(x, y)
                assert loss is not None
                loss_sum += loss.item()
                scaler.scale(loss / cfg.train.gradient_accumulation_steps).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.train.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if rank == 0 and step % cfg.train.log_interval == 0:
            elapsed = max(time.time() - start, 1e-6)
            interval_steps = step - start_step
            tokens_per_second = interval_steps * global_tokens_per_step / elapsed
            lr = scheduler.get_last_lr()[0]
            loss_value = loss_sum / cfg.train.gradient_accumulation_steps
            ppl = math.exp(min(loss_value, 20.0))
            tokens_seen = step * global_tokens_per_step
            diagnostics = collect_train_diagnostics(raw_model)
            metrics = {
                "step": step,
                "tokens": tokens_seen,
                "loss": loss_value,
                "ppl": ppl,
                "lr": lr,
                "grad_norm": float(grad_norm.detach().cpu()),
                "tokens_per_second": tokens_per_second,
                "parameters": parameter_count,
                "max_steps": cfg.train.max_steps,
                "warmup_steps": cfg.train.warmup_steps,
                "target_tokens": cfg.train.max_steps * global_tokens_per_step,
                "context_length": cfg.model.context_length,
                "micro_batch_size": cfg.train.batch_size,
                "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
                "world_size": world_size,
                "effective_batch_tokens": global_tokens_per_step,
                "token_budget_multiplier": (
                    cfg.train.token_budget_multiplier
                    if cfg.train.token_budget_multiplier is not None
                    else float("nan")
                ),
                "learning_rate": cfg.train.learning_rate,
                "weight_decay": cfg.train.weight_decay,
                "grad_clip": cfg.train.grad_clip,
                **diagnostics,
            }
            log_line = (
                f"step={step} loss={loss_value:.4f} ppl={ppl:.2f} lr={lr:.2e} "
                f"grad_norm={metrics['grad_norm']:.3f} alpha={metrics['alpha_mean']:.4f} "
                f"param_norm={metrics['param_norm']:.2f} tokens={tokens_seen} tok/s={tokens_per_second:.0f}"
            )
            print(log_line)
            assert log_file is not None
            log_file.write(log_line + "\n")
            log_file.flush()
            assert metrics_file is not None
            metrics_file.write(format_csv_row(metrics, METRIC_FIELDS) + "\n")
            metrics_file.flush()
            iterator.set_description(f"loss={loss_value:.4f} ppl={ppl:.2f} lr={lr:.2e}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss_value,
                        "train/ppl": ppl,
                        "train/lr": lr,
                        "train/grad_norm": metrics["grad_norm"],
                        "train/tokens_per_second": tokens_per_second,
                        "train/tokens": tokens_seen,
                        "train/param_norm": metrics["param_norm"],
                        "hparams/effective_batch_tokens": global_tokens_per_step,
                        "hparams/micro_batch_size": cfg.train.batch_size,
                        "hparams/gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
                        "hparams/context_length": cfg.model.context_length,
                        "hparams/warmup_steps": cfg.train.warmup_steps,
                        "hparams/max_steps": cfg.train.max_steps,
                        "hparams/target_tokens": cfg.train.max_steps * global_tokens_per_step,
                        "hparams/token_budget_multiplier": metrics["token_budget_multiplier"],
                        "sigmoidz/alpha_mean": metrics["alpha_mean"],
                        "sigmoidz/alpha_std": metrics["alpha_std"],
                        "sigmoidz/alpha_min": metrics["alpha_min"],
                        "sigmoidz/alpha_max": metrics["alpha_max"],
                        "sigmoidz/alpha_grad_abs_mean": metrics["alpha_grad_abs_mean"],
                        "sigmoidz/logit_bias_mean": metrics["logit_bias_mean"],
                        "sigmoidz/logit_bias_std": metrics["logit_bias_std"],
                        "sigmoidz/logit_bias_abs_max": metrics["logit_bias_abs_max"],
                        "sigmoidz/norm_weight_mean": metrics["norm_weight_mean"],
                        "sigmoidz/norm_weight_std": metrics["norm_weight_std"],
                        "sigmoidz/norm_bias_mean": metrics["norm_bias_mean"],
                        "sigmoidz/norm_bias_std": metrics["norm_bias_std"],
                    },
                    step=step,
                )

        if (
            rank == 0
            and val_batcher is not None
            and cfg.train.val_interval is not None
            and step % cfg.train.val_interval == 0
        ):
            val_loss, val_ppl = evaluate_loss(
                raw_model,
                val_batcher,
                cfg.train.val_steps,
                device,
                dtype,
                use_amp,
            )
            tokens_seen = step * global_tokens_per_step
            val_metrics = {
                "step": step,
                "tokens": tokens_seen,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "val_steps": cfg.train.val_steps,
                "val_dataset_split": cfg.train.val_dataset_split,
            }
            val_log_line = f"step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
            print(val_log_line)
            assert log_file is not None
            log_file.write(val_log_line + "\n")
            log_file.flush()
            assert val_metrics_file is not None
            val_metrics_file.write(format_csv_row(val_metrics, VAL_METRIC_FIELDS) + "\n")
            val_metrics_file.flush()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/tokens": tokens_seen,
                        "val/loss": val_loss,
                        "val/ppl": val_ppl,
                    },
                    step=step,
                )

        if (
            rank == 0
            and cfg.train.save_interval
            and step % cfg.train.save_interval == 0
        ):
            save_checkpoint(
                out_dir,
                step,
                raw_model,
                optimizer,
                scheduler,
                scaler,
                to_dict(cfg),
                cfg.train.keep_checkpoints,
            )

    if rank == 0:
        save_checkpoint(
            out_dir,
            cfg.train.max_steps,
            raw_model,
            optimizer,
            scheduler,
            scaler,
            to_dict(cfg),
            cfg.train.keep_checkpoints,
        )
        save_model(raw_model, out_dir / "model.safetensors")
        if wandb_run is not None:
            wandb_run.summary["parameters"] = parameter_count
            wandb_run.summary["tokens"] = cfg.train.max_steps * global_tokens_per_step
            wandb_run.finish()
        if log_file is not None:
            log_file.close()
        if metrics_file is not None:
            metrics_file.close()
        if val_metrics_file is not None:
            val_metrics_file.close()
    if is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
