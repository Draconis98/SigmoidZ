from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 128
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 512
    dropout: float = 0.0
    norm_type: str = "sigmoidz"
    block_variant: str = "conservative"
    alpha_attn: float = 0.8
    alpha_other: float = 0.2


@dataclass
class TrainConfig:
    tokenizer_name: str | None = None
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "train"
    max_train_tokens: int | None = None
    streaming: bool = False
    text_file: str | None = None
    batch_size: int = 16
    max_steps: int | None = 20
    token_budget_multiplier: float | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 5
    grad_clip: float = 1.0
    log_interval: int = 5
    save_interval: int = 100
    keep_checkpoints: int = 3
    resume_from: str | None = None
    out_dir: str = "runs/tiny"
    seed: int = 1337
    device: str = "auto"
    dtype: str = "float32"
    wandb_enabled: bool = False
    wandb_project: str = "sigmoidz"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: list[str] | None = None
    wandb_watch_model: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig
    train: TrainConfig


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(model=ModelConfig(), train=TrainConfig())


def fifty_m_config() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=32000,
            context_length=4096,
            hidden_size=384,
            num_layers=12,
            num_heads=6,
            intermediate_size=1536,
            norm_type="sigmoidz",
            block_variant="conservative",
            alpha_attn=0.8,
            alpha_other=0.2,
        ),
        train=TrainConfig(
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            dataset_name="wikitext",
            dataset_config="wikitext-103-raw-v1",
            streaming=True,
            batch_size=2,
            max_steps=None,
            token_budget_multiplier=20.0,
            warmup_steps=100,
            out_dir="runs/50m_sigmoidz",
            dtype="bfloat16",
            wandb_name="50m-sigmoidz",
            wandb_tags=["50m", "sigmoidz", "conservative"],
        ),
    )


PRESETS = {
    "tiny": tiny_config,
    "50m": fifty_m_config,
}


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    assert is_dataclass(obj)
    if isinstance(obj, ModelConfig):
        values = dict(values)
        if "omega_attn" in values:
            values["alpha_attn"] = values.pop("omega_attn")
        if "omega_other" in values:
            values["alpha_other"] = values.pop("omega_other")
    names = {f.name for f in fields(obj)}
    unknown = set(values) - names
    assert not unknown, f"Unknown config keys for {type(obj).__name__}: {sorted(unknown)}"
    for key, value in values.items():
        setattr(obj, key, value)
    return obj


def load_config(path: str | None, preset: str) -> ExperimentConfig:
    assert preset in PRESETS, f"Unknown preset {preset}; choose from {sorted(PRESETS)}"
    cfg = PRESETS[preset]()
    if path is None:
        return cfg

    raw = json.loads(Path(path).read_text())
    assert set(raw).issubset({"model", "train"})
    if "model" in raw:
        _merge_dataclass(cfg.model, raw["model"])
    if "train" in raw:
        _merge_dataclass(cfg.train, raw["train"])
    return cfg


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)
