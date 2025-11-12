# LLAMA 2 7B Inference Example with SST Integration

This example demonstrates running LLAMA 2 7B model inference on RISC-V Linux with SST accelerator integration.

## Quick Start

```bash
# 1. Build and deploy
make deploy

# 2. Start SST simulation (Terminal 1)
./run_sst.sh

# 3. Start QEMU Linux (Terminal 2)
./run_qemu.sh

# 4. In Linux, run inference
cd /apps/llama-inference
./llama_inference.py "Explain quantum computing"
```

## Files

| File | Description |
|------|-------------|
| **llama_inference.py** | Main inference application |
| **llama_sst_backend.py** | SST accelerator integration backend |
| **sst_config_llama.py** | SST simulation configuration |
| **run_sst.sh** | Launch SST simulation |
| **run_qemu.sh** | Launch QEMU with model disk |
| **test_prompts.txt** | Example prompts |
| **Makefile** | Build and deployment |

## Prerequisites

1. **Full Linux Root Filesystem** with Python 3.11+ and PyTorch
   - See [PYTORCH_LLAMA_SETUP.md](../../PYTORCH_LLAMA_SETUP.md) for setup

2. **LLAMA 2 7B Model** on virtual disk at `/mnt/models/llama-2-7b/`
   - Download from Hugging Face (requires access request)

3. **SST Components** compiled with VirtIO device support

## Usage

### Basic Inference

```bash
./llama_inference.py "Your prompt here"
```

### Batch Processing

```bash
# Process all prompts from file
cat test_prompts.txt | while read prompt; do
    ./llama_inference.py "$prompt"
done
```

### With Custom Parameters

```python
# Edit llama_inference.py:
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=200,  # Generate more tokens
    temperature=0.8,      # More creative
    top_p=0.95
)
```

## SST Accelerator Configuration

The example uses 4 specialized accelerators:

| Accelerator | Type | Workload | Latency |
|-------------|------|----------|---------|
| `accel0` | Attention | Self-attention computation | 1μs |
| `accel1` | FFN | Feed-forward network | 500ns |
| `accel2` | Embedding | Token embeddings | 100ns |
| `accel3` | General | Misc operations | 200ns |

Edit `sst_config_llama.py` to adjust accelerator parameters.

## Performance Metrics

The application reports:

- **Attention operations**: Count of self-attention calls
- **FFN operations**: Count of feed-forward network calls
- **Embedding operations**: Token embedding lookups
- **Total simulated cycles**: Aggregate accelerator cycles
- **Estimated wall-clock time**: Based on 2GHz clock

Example output:

```
SST Accelerator Statistics:
  Attention operations: 324
  FFN operations: 324
  Embedding operations: 1
  Total simulated cycles: 1,234,567
  Estimated wall-clock time: 0.617ms
```

## Customization

### Use Different Model

Edit `llama_inference.py`:

```python
MODEL_PATH = "/mnt/models/llama-2-13b"  # Larger model
# or
MODEL_PATH = "/mnt/models/tinyllama"     # Smaller model
```

### Change Accelerator Latency

Edit `sst_config_llama.py`:

```python
"latency_attention": "100ns",  # 10x faster
"latency_ffn": "50ns",
```

### Enable Quantization

Edit `llama_inference.py`:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config
)
```

## Troubleshooting

See [PYTORCH_LLAMA_SETUP.md](../../PYTORCH_LLAMA_SETUP.md#troubleshooting) for detailed troubleshooting.

### Quick Fixes

**Model not found:**
```bash
# Check mount
mount | grep models
# Mount if needed
mount /dev/vda /mnt/models
```

**SST device error:**
```bash
# Check device
ls -l /dev/sst0
# Load driver if needed
insmod /virtio-sst.ko
```

**Out of memory:**
```bash
# Restart QEMU with more RAM
# In run_qemu.sh: -m 8G
```

## Performance Tips

1. **Use Quantization**: Reduce model size (13GB → 3.5GB)
2. **Increase QEMU RAM**: `-m 8G` for larger models
3. **Optimize SST Latency**: Tune accelerator parameters
4. **Reduce Tokens**: Lower `max_new_tokens` for faster response

## Example Session

```bash
$ ./llama_inference.py "What is machine learning?"

Loading model from /mnt/models/llama-2-7b...
[████████████████████████████████] 100%
Model loaded successfully

Prompt: What is machine learning?
Generating 100 tokens...
[██████████████████████████......] 80%

============================================================
GENERATED TEXT:
============================================================
What is machine learning? Machine learning is a subset of
artificial intelligence that enables computers to learn from
data without being explicitly programmed. It involves training
algorithms on datasets to identify patterns and make
predictions or decisions...
============================================================

SST Accelerator Statistics:
  Attention operations: 324
  FFN operations: 324
  Embedding operations: 1
  Total simulated cycles: 1,234,567
  Estimated wall-clock time: 0.617ms
============================================================
```

## Related Documentation

- [PYTORCH_LLAMA_SETUP.md](../../PYTORCH_LLAMA_SETUP.md) - Complete setup guide
- [ROOTFS_MANAGEMENT.md](../../ROOTFS_MANAGEMENT.md) - Package installation
- [APP_DEVELOPMENT.md](../../APP_DEVELOPMENT.md) - SST application development
