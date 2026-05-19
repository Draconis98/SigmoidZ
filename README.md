# SigmoidZ

SigmoidZ explores a small change to the Probabilistic Transformer view of contextual representations: replace a categorical latent label with a binary vector `Z_i in {0, 1}^d`. The resulting mean-field update has a sigmoid form, which can be mapped back to Transformer blocks.

The repo keeps two variants:

- `conservative`: replace RMSNorm/LayerNorm-style layers with `SigmoidZNorm`.
- `research`: use a sigmoid mean-field-style update inside the attention path.

The default is a decoder-only causal LM with the conservative variant.

## Theory

The binary MFVI update is:

$$q_{i,a} = Q(Z_{i,a} = 1) = \sigma(S_{i,a} + G_{i,a})$$

The centered representation is:

$$2q_{i,a} - 1$$

SigmoidZNorm uses:

$$\gamma * (2 \sigma(2 \alpha x + \beta) - 1) + \delta$$

When $\beta = 0$, this is exactly DyT because:

$$\tanh(x) = 2 \sigma(2x) - 1$$

See [docs/theory.md](docs/theory.md).

## Setup

This project uses `uv` and Python 3.12.

```bash
UV_PROJECT_ENVIRONMENT=~/venv/.sigmoidz uv sync --dev
```

Install optional W&B tracking support when needed:

```bash
UV_PROJECT_ENVIRONMENT=~/venv/.sigmoidz uv sync --dev --extra tracking
```

The current macOS environment can run CPU or MPS correctness checks. CUDA training is supported when the same code is run on an Nvidia server with a CUDA-enabled PyTorch build. The model uses PyTorch `scaled_dot_product_attention`; on CUDA, PyTorch can dispatch it to efficient attention kernels when supported.

## Run Tests

```bash
UV_PROJECT_ENVIRONMENT=~/venv/.sigmoidz uv run pytest
```

## Smoke Pretraining

This runs a tiny byte-level model on a built-in text corpus. It is only a correctness check.

```bash
UV_PROJECT_ENVIRONMENT=~/venv/.sigmoidz uv run python src/train.py --preset tiny --max_steps 20 --out_dir runs/tiny
```

Or use the script:

```bash
bash scripts/smoke_train.sh
```

If `uv` is busy syncing the environment, use the existing interpreter directly:

```bash
PYTHON_BIN=~/venv/.sigmoidz/bin/python bash scripts/smoke_train.sh
```

Enable Weights & Biases logging with `--wandb` after logging in or setting `WANDB_API_KEY`:

```bash
UV_PROJECT_ENVIRONMENT=~/venv/.sigmoidz uv run python src/train.py --preset tiny --max_steps 20 --out_dir runs/tiny --wandb
```

## 50M-Style Pretraining

The included config is a small LLaMA-style decoder-only LM:

```text
hidden_size: 384
num_layers: 12
num_heads: 6
intermediate_size: 1536
context_length: 4096
token_budget_multiplier: 20.0
warmup_ratio: 0.01
gradient_accumulation_steps: 1
save_interval: 100
keep_checkpoints: 3
streaming: true
val_dataset_split: validation
val_interval: 100
val_steps: 20
```

When `max_steps` is `null`, the trainer sets:

```text
max_steps = ceil(model_parameters * token_budget_multiplier / global_tokens_per_step)
```

For the default 50M-style config this targets roughly 20 tokens per model parameter.

`batch_size` is the per-process micro-batch size. Effective tokens per optimizer step are:

```text
batch_size * gradient_accumulation_steps * context_length * world_size
```

Increase `gradient_accumulation_steps` when the target global batch does not fit in GPU memory.

Example:

```bash
bash scripts/train_50m.sh --gradient_accumulation_steps 8
```

When `warmup_steps` is `null`, the trainer sets:

```text
warmup_steps = ceil(max_steps * warmup_ratio)
```

The default 50M-style config uses `warmup_ratio: 0.01`, so warmup is 1% of total steps.

Run single process:

```bash
UV_PROJECT_ENVIRONMENT=~/venv/.sigmoidz uv run python src/train.py --config configs/50m_sigmoidz.json
```

Or use the script:

```bash
bash scripts/train_50m.sh
```

Run multi-GPU with `torchrun`:

```bash
torchrun --nproc_per_node=8 src/train.py --config configs/50m_sigmoidz.json
```

Or use the script:

```bash
bash scripts/torchrun_50m.sh
```

For W&B logging, either pass `--wandb` or set `"wandb_enabled": true` in the config. Logging is only initialized on rank 0 under DDP. Set `"wandb_watch_model": true` if you also want W&B gradient/model watching.

W&B single-process script:

```bash
bash scripts/train_50m_wandb.sh
```

Scripts pass extra arguments through to `src/train.py`, for example:

```bash
WANDB_MODE=offline bash scripts/smoke_train.sh --wandb
```

Validation runs on rank 0 when `val_interval` is set. For Hugging Face datasets, set `val_dataset_split` to the validation split name. For local text validation, set `val_text_file`.

```bash
bash scripts/train_50m.sh --val_interval 100 --val_steps 20 --val_dataset_split validation
bash scripts/smoke_train.sh --val_interval 10 --val_steps 5 --val_text_file data/valid.txt
```

## Checkpoints

Training writes full checkpoints under:

```text
runs/.../checkpoints/step_00000100.pt
```

Each checkpoint stores model weights, optimizer state, scheduler state, AMP scaler state, config, and the current step. By default it saves every 100 steps and keeps only the latest 3 checkpoints.

Rank 0 also appends scalar logs to:

```text
runs/.../train.log
runs/.../train_metrics.csv
runs/.../val_metrics.csv
```

The plain log includes `step`, `loss`, `ppl`, `lr`, `grad_norm`, `alpha`, `param_norm`, `tokens`, and `tok/s`. The CSV is intended for plotting and diagnostics; use `tokens` as the x-axis and `loss` as the y-axis. The loss is already averaged over target tokens by cross entropy, so it should not be divided by cumulative token count again.

The CSV also records repeated run settings and learned-parameter diagnostics:

- run settings: `parameters`, `max_steps`, `warmup_steps`, `target_tokens`, `context_length`, `micro_batch_size`, `gradient_accumulation_steps`, `world_size`, `effective_batch_tokens`, `token_budget_multiplier`, `learning_rate`, `weight_decay`, `grad_clip`
- SigmoidZ/DyT diagnostics: `alpha_mean`, `alpha_std`, `alpha_min`, `alpha_max`, `alpha_grad_abs_mean`, `logit_bias_mean`, `logit_bias_std`, `logit_bias_abs_max`, `norm_weight_mean`, `norm_weight_std`, `norm_bias_mean`, `norm_bias_std`, `param_norm`

When validation is enabled, `val_metrics.csv` records `val_loss` and `val_ppl` at each validation interval.

W&B also defines `train/tokens` as the step metric for `train/loss`, `train/ppl`, `train/lr`, `train/grad_norm`, `train/tokens_per_second`, `train/param_norm`, `val/loss`, `val/ppl`, `sigmoidz/*`, and `hparams/*`, so curves are plotted against training tokens rather than optimizer steps.

Resume from the newest checkpoint:

```bash
bash scripts/train_50m.sh --resume latest
```

Resume from a specific checkpoint:

```bash
bash scripts/train_50m.sh --resume runs/50m_sigmoidz/checkpoints/step_00000100.pt
```

## Tokenizers

The default 50M tokenizer is `meta-llama/Llama-2-7b-hf`, which requires Hugging Face access approval and an authenticated environment. Other practical choices are:

- `meta-llama/Llama-2-7b-hf`: default LLaMA tokenizer, requires Hugging Face access approval.
- `gpt2`: compact baseline, easy to fetch, vocab around 50k.
- `EleutherAI/gpt-neox-20b`: Pile-style GPT-NeoX tokenizer, a better match if training on The Pile-like data.
- `mistralai/Mistral-7B-v0.1`: Mistral tokenizer, useful for LLaMA-family experiments.
- `Qwen/Qwen2.5-0.5B`: Qwen tokenizer, larger multilingual coverage.

Set it in config:

```json
{
  "train": {
    "tokenizer_name": "gpt2"
  }
}
```

The model vocab size is inferred from the tokenizer at runtime.

## Streaming Data

For Hugging Face datasets, set:

```json
{
  "train": {
    "streaming": true
  }
}
```

Streaming mode tokenizes samples as training runs and packs them into fixed-length batches. It avoids flattening the full dataset into RAM. Local tiny/text-file training still uses the in-memory random-span batcher because those datasets are small.

DeepSpeed is not included in the initial dependency set. For 50M-scale experiments, `torchrun` with DDP is the simpler baseline. Add DeepSpeed later if ZeRO sharding or optimizer offload becomes necessary.

## Variants

Set these fields in a JSON config:

```json
{
  "model": {
    "norm_type": "sigmoidz",
    "attn_norm_type": "sigmoidz",
    "ffn_norm_type": "rmsnorm",
    "final_norm_type": "rmsnorm",
    "block_variant": "conservative"
  }
}
```

Supported `norm_type` values:

- `rmsnorm`
- `dyt`
- `sigmoidz`

Supported `block_variant` values:

- `conservative`
- `research`

`norm_type` remains the default for all normalization sites. Set `attn_norm_type`, `ffn_norm_type`, or `final_norm_type` to override individual sites for hybrid ablations.

For `research`, prefer RMSNorm at the surrounding normalization sites and let `SigmoidZAttentionUpdate` supply the Bernoulli update internally:

```json
{
  "model": {
    "norm_type": "rmsnorm",
    "attn_norm_type": "rmsnorm",
    "ffn_norm_type": "rmsnorm",
    "final_norm_type": "rmsnorm",
    "block_variant": "research"
  }
}
```

Use `configs/50m_sigmoidz_research.json` for this setup.

## Notes

The DyT LLM recipe trains LLaMA models on The Pile and tunes the initial DyT slope by scale and block type. SigmoidZ follows the same practical idea with:

- `alpha_attn`: initial slope for attention block normalization
- `alpha_other`: initial slope for FFN and final normalization when those sites use SigmoidZNorm

These config values are initial values. In `SigmoidZNorm`, `alpha`, `logit_bias` (`beta`), `weight` (`gamma`), and `bias` (`delta`) are all learned parameters.

The provided code is an experimental pretraining scaffold, not a reproduction of large-scale DyT results.
