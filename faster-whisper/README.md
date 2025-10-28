# Amharic Fine-Tuned Whisper with Gradio UI

This repository contains code and resources for fine-tuning OpenAI's Whisper model for Amharic speech recognition using LJSpeech-format datasets, with a comprehensive Gradio WebUI for training and inference. Optimized for Lightning AI remote training.

## ✨ Features

- 🎙️ **Amharic Speech Recognition**: Fine-tune Whisper for Amharic language
- 📦 **LJSpeech Dataset Support**: Convert TTS datasets for STT training
- 🚀 **Lightning AI Optimized**: Built with PyTorch Lightning for remote training
- 🎨 **Gradio WebUI**: Beautiful interface for training and inference
- 💾 **Checkpoint Management**: Save, resume, and manage training checkpoints
- ⚡ **Fast Inference**: CTranslate2 conversion for optimized deployment
- 🔧 **Advanced Features**: DeepSpeed, LoRA, mixed precision training

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Launch Gradio UI

```bash
python app.py
```

The UI will be available at `http://localhost:7860`

## 📖 Usage

### 1. Prepare Your Dataset

Your LJSpeech-format Amharic dataset should have this structure:
```
ljspeech_amharic/
├── wavs/
│   ├── audio001.wav
│   ├── audio002.wav
│   └── ...
└── metadata.csv
```

`metadata.csv` format (pipe-separated):
```
audio001|የአማርኛ ጽሑፍ እዚህ
audio002|ሌላ አማርኛ ጽሑፍ
```

### 2. Using the Gradio UI

#### Training Tab:
1. **Prepare Dataset**: Enter your LJSpeech path and click "Prepare Dataset"
2. **Configure Training**: Adjust hyperparameters (learning rate, epochs, batch size, etc.)
3. **Advanced Settings**: Enable DeepSpeed, LoRA, or adjust gradient settings
4. **Start Training**: Click "Start Training" to begin
5. **Monitor Progress**: Watch real-time training metrics
6. **Resume Training**: Select a checkpoint to continue from where you left off

#### Inference Tab:
1. **Select Model**: Choose from fine-tuned models or checkpoints
2. **Upload Audio**: Drag and drop or record audio
3. **Transcribe**: Click to get Amharic transcription

### 3. Command Line Usage

#### Prepare Dataset
```bash
python prepare_ljspeech_dataset.py \
  --ljspeech_path /path/to/ljspeech_amharic \
  --output_path ./data \
  --train_split 0.9 \
  --val_split 0.05
```

#### Train Model
```bash
python train_whisper_lightning.py \
  --data_path ./data \
  --model_name openai/whisper-small \
  --num_epochs 20 \
  --batch_size 16 \
  --learning_rate 1e-5
```

## ⚙️ Configuration

### Training Configuration

- **Model**: Choose from `whisper-tiny`, `whisper-base`, `whisper-small`, or `whisper-medium`
- **Learning Rate**: Default `1e-5`, adjust based on model size
- **Batch Size**: Default `16`, increase for better GPUs
- **Precision**: `16` for faster training, `32` for higher accuracy
- **DeepSpeed**: Enable for large models to reduce memory usage
- **LoRA**: Parameter-efficient fine-tuning for faster training

### Checkpoint Management

- **Auto-Save**: Best checkpoints saved automatically during training
- **Resume Training**: Continue from any saved checkpoint
- **Top-K Saving**: Keep only the best K checkpoints
- **Delete Checkpoints**: Remove unwanted checkpoints via UI

## 🌩️ Lightning AI Deployment

### Setup on Lightning AI Studio

1. Create a new Studio session
2. Clone this repository
3. Install dependencies: `pip install -r requirements.txt`
4. Upload your dataset to cloud storage
5. Launch Gradio UI: `python app.py`

### Training on Remote GPUs

```python
# Lightning Trainer automatically detects and uses available GPUs
trainer = L.Trainer(
    accelerator="auto",  # Auto-detect GPU/TPU
    devices="auto",      # Use all available devices
    strategy="deepspeed_stage_2",  # For large models
    precision=16         # Mixed precision
)
```

## 📊 Performance

### Expected Results
- **WER on Clean Speech**: < 30%
- **Training Time**: 2-4 hours on A100 GPU
- **Model Size**: ~500MB (fine-tuned), ~150MB (CTranslate2)
- **Inference Speed**: Real-time on GPU

### Optimization Tips

1. **Use Mixed Precision**: 16-bit training is 2-3x faster
2. **Enable LoRA**: Reduce trainable parameters by 99%
3. **Gradient Accumulation**: Simulate larger batch sizes
4. **DeepSpeed**: Handle models that don't fit in VRAM

## 🔧 Advanced Features

### LoRA Fine-Tuning
```python
config = {
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32
}
```

### DeepSpeed Integration
```python
config = {
    "use_deepspeed": True,
    "precision": 16
}
```

### CTranslate2 Conversion
```python
from inference_utils import convert_to_ctranslate2

convert_to_ctranslate2(
    model_path="./whisper_finetuned",
    output_path="./whisper_ct2_model",
    quantization="int8"
)
```

## 📁 Project Structure

```
.
├── app.py                          # Gradio WebUI
├── train_whisper_lightning.py      # Lightning training module
├── inference_utils.py              # Inference utilities
├── prepare_ljspeech_dataset.py     # Dataset preparation
├── requirements.txt                # Dependencies
├── SETUP.md                        # Detailed setup guide
├── checkpoints/                    # Saved checkpoints
├── data/                          # Processed datasets
├── whisper_finetuned/             # Fine-tuned models
└── whisper_ct2_model/             # CTranslate2 models
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for Transformers library
- Lightning AI for PyTorch Lightning
- Gradio team for the UI framework

## 📮 Contact

For questions or issues, please open an issue on GitHub.