# Setup Guide - Amharic Whisper Fine-Tuning

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- ~50GB free disk space
- LJSpeech-format Amharic dataset

## Installation Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd turkish-finetuned-faster-whisper
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n whisper-amharic python=3.10
conda activate whisper-amharic
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install DeepSpeed

For large model training:

```bash
# Linux
pip install deepspeed

# Windows (requires Visual Studio Build Tools)
pip install deepspeed --no-build-isolation
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import lightning; print(f'Lightning: {lightning.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
```

## Dataset Preparation

### LJSpeech Format Requirements

Your dataset should follow this structure:

```
ljspeech_amharic/
├── wavs/
│   ├── audio001.wav
│   ├── audio002.wav
│   └── ...
└── metadata.csv
```

### metadata.csv Format

Pipe-separated (|) file with no header:

```
audio001|የአማርኛ ጽሑፍ እዚህ
audio002|ሌላ አማርኛ ጽሑፍ
audio003|ሦስተኛው ምሳሌ
```

### Audio Requirements

- **Format**: WAV (preferred) or FLAC
- **Sampling Rate**: Any (will be resampled to 16kHz)
- **Channels**: Mono or stereo (will be converted to mono)
- **Duration**: 1-30 seconds per clip (optimal: 5-10 seconds)
- **Quality**: Clean speech, minimal background noise

## Quick Start

### Option 1: Using Gradio UI (Recommended)

```bash
python app.py
```

Navigate to `http://localhost:7860` in your browser.

### Option 2: Command Line

#### Step 1: Prepare Dataset

```bash
python prepare_ljspeech_dataset.py \
  --ljspeech_path /path/to/ljspeech_amharic \
  --output_path ./data \
  --train_split 0.9 \
  --val_split 0.05
```

#### Step 2: Train Model

Create a training config JSON:

```json
{
  "model_name": "openai/whisper-small",
  "language": "am",
  "learning_rate": 1e-5,
  "batch_size": 16,
  "num_epochs": 20,
  "data_path": "./data",
  "checkpoint_dir": "./checkpoints"
}
```

Then run:

```bash
python train_whisper_lightning.py --config training_config.json
```

## Lightning AI Setup

### 1. Create Lightning AI Account

- Go to https://lightning.ai
- Sign up for free account
- Create a new Studio session

### 2. Configure Studio

```bash
# In Lightning Studio terminal
git clone <repository-url>
cd turkish-finetuned-faster-whisper
pip install -r requirements.txt
```

### 3. Upload Dataset

Option A: Upload via UI
- Use Lightning Studio's file browser
- Upload to `/teamspace/studios/this_studio/data/`

Option B: Mount Cloud Storage
```bash
# Mount S3, GCS, or Azure Blob Storage
lightning mount s3://your-bucket /mnt/data
```

### 4. Launch Training

```bash
# Start Gradio UI
python app.py

# Or run training directly
python train_whisper_lightning.py
```

### 5. Monitor Training

- Access Gradio UI via Lightning Studio's port forwarding
- Use TensorBoard: `tensorboard --logdir ./logs`
- Check WandB dashboard if configured

## Configuration Tips

### For Limited VRAM (<16GB)

```python
config = {
    "model_name": "openai/whisper-tiny",  # Smaller model
    "batch_size": 4,                      # Reduce batch size
    "accumulate_grad_batches": 8,         # Simulate larger batches
    "precision": 16,                      # Use mixed precision
    "use_lora": True,                     # Enable LoRA
    "gradient_checkpointing": True        # Trade compute for memory
}
```

### For Fast Experimentation

```python
config = {
    "model_name": "openai/whisper-base",
    "num_epochs": 5,
    "use_lora": True,
    "lora_r": 8,
    "batch_size": 8
}
```

### For Production Training

```python
config = {
    "model_name": "openai/whisper-medium",
    "num_epochs": 30,
    "batch_size": 16,
    "accumulate_grad_batches": 4,
    "use_deepspeed": True,
    "precision": 16,
    "save_top_k": 5
}
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size`
- Increase `accumulate_grad_batches`
- Enable `use_lora`
- Use smaller model (tiny/base instead of small/medium)
- Enable gradient checkpointing

### Slow Training

- Ensure CUDA is available: `torch.cuda.is_available()`
- Use `precision=16` for mixed precision
- Increase `batch_size` if VRAM allows
- Check `num_workers` in dataloader (default: 4)

### Dataset Errors

- Verify metadata.csv format (pipe-separated)
- Check audio file paths are correct
- Ensure audio files are readable (try loading with torchaudio)
- Validate Amharic text encoding (UTF-8)

### Gradio UI Issues

- Check port 7860 is not in use
- Try different port: `demo.launch(server_port=7861)`
- Enable sharing: `demo.launch(share=True)`
- Check firewall settings

## Next Steps

1. ✅ Complete setup and verify installation
2. 📦 Prepare your Amharic dataset
3. 🧪 Run a small test training (5 epochs, 100 samples)
4. 🚀 Launch full training
5. 📊 Monitor metrics and adjust hyperparameters
6. 🎯 Evaluate on test set
7. ⚡ Convert to CTranslate2 for deployment

## Support

For issues:
- Check existing GitHub issues
- Review Lightning AI documentation
- Check Hugging Face forums
- Open a new issue with:
  - Error message
  - System info (GPU, CUDA version)
  - Config file
  - Steps to reproduce