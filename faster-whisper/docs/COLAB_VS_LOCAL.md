# Google Colab vs Local Training Comparison

## Requirements Files

### `requirements.txt` (Local/Server)
- Full dependencies for local machines
- Latest versions of all packages
- Includes development tools
- Docker deployment support
- Production monitoring stack

### `requirements-colab.txt` (Google Colab)
- Optimized for Colab environment
- Pinned versions to avoid conflicts
- Excludes Colab pre-installed packages
- GPU-specific versions (CUDA 12.2)
- Lightweight monitoring

## Key Differences

| Aspect | Local/Server | Google Colab |
|--------|-------------|-------------|
| **PyTorch** | 2.1.0-2.3.0 (flexible) | 2.1.0 (pinned) |
| **CUDA** | System dependent | 12.2 (fixed) |
| **onnxruntime** | CPU or GPU | GPU only |
| **Gradio** | 4.14.0+ | 4.8.0 (stable) |
| **DeepSpeed** | 0.11.0+ | 0.12.6 (Colab compatible) |
| **Batch Size** | 16-32 | 8-12 (limited RAM) |
| **Training Time** | 12-24h (V100/A100) | 24-48h (T4) |
| **Session** | Unlimited | 12h (free) / 24h (Pro) |
| **Storage** | Local disk | Google Drive |
| **Monitoring** | Full stack | TensorBoard only |

## Colab-Specific Adjustments

### 1. Smaller Batch Sizes
```yaml
# Local
batch_size: 16
gradient_accumulation_steps: 4

# Colab
batch_size: 8
gradient_accumulation_steps: 8
```

### 2. Reduced Workers
```yaml
# Local
num_workers: 4

# Colab
num_workers: 2
```

### 3. Checkpoint Management
```yaml
# Local - keep more checkpoints
save_top_k: 3

# Colab - save to Drive frequently
save_top_k: 2
save_last: true
```

### 4. VAD Loading
```python
# Local - install via pip
pip install silero-vad

# Colab - use torch.hub (no pip needed)
torch.hub.load('snakers4/silero-vad', 'silero_vad')
```

## Installation Commands

### Local
```bash
pip install -r requirements.txt
```

### Colab
```python
!pip install -q -r requirements-colab.txt
```

## Package Versions

### Excluded from Colab (Pre-installed)
- matplotlib
- seaborn
- requests
- Pillow
- jupyter
- ipython

### Modified for Colab
- `numpy==1.24.3` (vs 1.24.0+ locally)
- `pandas==2.0.3` (vs 2.0.0+ locally)
- `scipy==1.11.4` (vs 1.10.0+ locally)

## Troubleshooting

### Colab-Specific Issues

**Problem**: CUDA version mismatch
```python
# Check Colab CUDA
!nvcc --version
# Should be 12.2
```

**Problem**: Out of memory
```python
# Reduce batch size
batch_size: 4
gradient_accumulation_steps: 16
```

**Problem**: Session timeout
```python
# Save checkpoints to Drive every epoch
from google.colab import drive
drive.mount('/content/drive')
!cp -r checkpoints /content/drive/MyDrive/
```

## Best Practices

### For Colab
1. ✅ Mount Google Drive first
2. ✅ Use `-q` flag for quiet installs
3. ✅ Save checkpoints to Drive regularly
4. ✅ Use TensorBoard for monitoring
5. ✅ Test with small dataset first
6. ✅ Use Colab Pro for longer sessions

### For Local
1. ✅ Use virtual environment
2. ✅ Install CUDA drivers separately
3. ✅ Use Docker for production
4. ✅ Set up full monitoring stack
5. ✅ Use multiple GPUs if available

## Migration Path

### Colab → Local
1. Download model from Google Drive
2. Install full requirements.txt
3. Update config (increase batch size)
4. Continue training from checkpoint

### Local → Colab
1. Upload model to Google Drive
2. Install requirements-colab.txt
3. Update config (decrease batch size)
4. Resume training

## Performance Comparison

### Training Speed (150h dataset, whisper-medium)

| Hardware | Time/Epoch | Total Time | Cost |
|----------|-----------|------------|------|
| Local RTX 3090 | ~45min | ~11h | Hardware cost |
| Local V100 | ~35min | ~9h | Cloud cost |
| Local A100 | ~25min | ~6h | Cloud cost |
| Colab T4 (Free) | ~90min | ~23h | Free |
| Colab V100 (Pro) | ~35min | ~9h | $10/month |
| Colab A100 (Pro+) | ~25min | ~6h | $50/month |

## Recommendations

### Use Colab If:
- No local GPU available
- Testing/prototyping
- Budget constrained
- Dataset < 100 hours
- Can handle 12-24h limits

### Use Local If:
- Have GPU (RTX 3090+)
- Need 24/7 training
- Large dataset (> 200h)
- Production deployment
- Need full monitoring

### Hybrid Approach:
- Prototype on Colab (free)
- Train on local/cloud GPU
- Deploy with Docker
- Monitor with full stack
