# LLM Training (nanoGPT)

A local LLM training project built on [nanoGPT](https://github.com/karpathy/nanoGPT). Training runs **locally on a single NVIDIA RTX 3080 (10 GB VRAM)**, tuned to fit a GPT-2-class model within that memory budget.

## Environment

- **Python:** 3.13 (conda env named `llm-training`)
- **GPU:** NVIDIA RTX 3080, 10 GB VRAM (Ampere, `sm_86` — supports bf16)
- **PyTorch:** 2.11.0 + CUDA 12.8 (`torch==2.11.0+cu128`)
- **Key packages:** `torch`, `numpy`, `transformers`, `datasets`, `tiktoken`, `wandb`, `tqdm`

Activate the environment before doing anything:

```bash
conda activate llm-training
```

Sanity check the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## The 10 GB VRAM constraint

An RTX 3080 has only 10 GB, so the default nanoGPT config (designed for 40–80 GB A100s) will OOM. The training config must be tuned along four axes:

### 1. Small micro-batch + gradient accumulation

Keep the per-step micro-batch small so activations fit in memory, and use gradient accumulation to reach a large *effective* batch size:

```
effective_batch = batch_size × gradient_accumulation_steps × block_size  (tokens)
```

Suggested starting point in `train.py`:

```python
batch_size = 4                     # micro-batch that fits in 10 GB
gradient_accumulation_steps = 32   # accumulate to simulate a larger batch
```

Tune `batch_size` up/down first based on actual VRAM usage (`nvidia-smi`), then adjust `gradient_accumulation_steps` to hit your target effective batch.

### 2. Gradient checkpointing (ON)

Trades compute for memory by recomputing activations during the backward pass instead of storing them — typically the single biggest VRAM saver.

> **Note:** stock nanoGPT does **not** implement gradient checkpointing. It must be added manually in `model.py` by wrapping the per-block call in the forward pass with `torch.utils.checkpoint.checkpoint`. For example, in `GPT.forward`:
>
> ```python
> import torch.utils.checkpoint as cp
> for block in self.transformer.h:
>     x = cp.checkpoint(block, x, use_reentrant=False)
> ```
>
> Gate it behind a config flag so it can be disabled for inference.

### 3. bf16 mixed precision

The RTX 3080 supports bfloat16, which halves activation/gradient memory and is more numerically stable than fp16 (no GradScaler needed). nanoGPT already auto-selects it when supported, but pin it explicitly:

```python
dtype = 'bfloat16'
```

### 4. Context length ~512–1024 (not 2048)

Attention memory grows with sequence length, so keep `block_size` modest. Use **512–1024**, not 2048:

```python
block_size = 512   # raise toward 1024 only if VRAM allows
```

## Recommended `train.py` overrides

These can be set in a config file under `nanoGPT/config/` or passed on the command line. Starting point for the 3080:

```python
batch_size = 4
gradient_accumulation_steps = 32
block_size = 512
dtype = 'bfloat16'
compile = True            # PyTorch 2.x graph compilation, extra speed
grad_clip = 1.0
device = 'cuda'
# gradient_checkpointing = True   # requires the model.py change described above
```

## Running training

From the `nanoGPT/` directory:

```bash
# Train with a config file
python train.py config/your_config.py

# Or override individual settings on the command line
python train.py --batch_size=4 --gradient_accumulation_steps=32 --block_size=512 --dtype=bfloat16
```

Monitor VRAM usage in a second terminal while training:

```bash
nvidia-smi -l 1
```

If you OOM: lower `batch_size` first, then `block_size`, and confirm gradient checkpointing is enabled.

## `torch.compile` and Windows

`compile` is set to **`False`** in `config/train_gpt2_rtx3080.py`, and it should stay that way on this machine.

`torch.compile` (PyTorch 2.x graph compilation) can speed up training **~1.3–2x** on Ampere GPUs by fusing kernels and cutting Python overhead. However, it depends on **Triton**, which does **not** ship with PyTorch's native-Windows wheels. Setting `compile = True` here fails at the first forward pass with:

```
torch._inductor.exc.TritonMissing: Cannot find a working triton installation.
```

Note this is purely a **speed** feature — it changes nothing about correctness, loss, or memory. Leaving it off is fully safe; you only forgo the throughput gain.

If you want that speedup later, options in rough order of effort:

| Option | What it is | Tradeoff |
|--------|-----------|----------|
| **Leave `compile=False`** (current) | No graph compilation | ~1.3–2x slower per step, but zero risk and works today. Fine for short/Shakespeare-scale runs. |
| **`pip install triton-windows`** | Unofficial community Triton port for Windows | *May* enable compile, but it's unofficial and version-sensitive against PyTorch 2.11 — no guarantee it works. |
| **Run under WSL2** | Linux PyTorch + CUDA inside WSL2 | Triton works properly; best path for long multi-day runs. Larger setup effort. |

The speedup compounds over long pretraining runs (e.g. 600k iters), so it's worth revisiting `triton-windows` or WSL2 only if training throughput becomes a real bottleneck. For experimentation, `compile=False` is the right default.

> Gradient checkpointing uses `use_reentrant=False`, which composes correctly with `torch.compile` — so if you do enable compile later (via WSL2/triton-windows), the two features won't conflict.

## Notes

- `wandb` is available for experiment tracking; enable it via nanoGPT's `wandb_log = True` setting.
- See `nanoGPT-model-explained.md` for a block-by-block walkthrough of the model architecture.
