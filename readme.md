Check if a HuggingFace model will run on your GPU.

`fitgpu` takes a HuggingFace model ID and tells you whether the model's weights will fit in your GPU's available VRAM.

### How?

1. Gets model's file metadata from HuggingFace (weights are **not** downloaded)
2. Sums up the sizes of all weight files (`.safetensors` / `.bin`)
3. Queries your GPU's free VRAM using the NVIDIA driver
4. Compares and shows the result

### Installation

```bash
pip install fitgpu
```

### Use

```
fitgpu <model_id> [--token TOKEN]
```

- `model_id` — HuggingFace model ID (e.g. `google/gemma-2-2b`)
- `--token TOKEN` — optional, HuggingFace API token for gated/private models

#### Public models

```bash
fitgpu google/gemma-2-2b
```

#### Gated models

```bash
fitgpu meta-llama/Llama-2-7b-hf --token hf_YOUR_TOKEN
```

### Example

```
$ fitgpu google/gemma-2-2b
model : google/gemma-2-2b
size  : 4.89 GB (weights on disk)

GPU 0: NVIDIA RTX 4090
  VRAM : 24.00 GB total, 22.31 GB free
  result: fits
```
