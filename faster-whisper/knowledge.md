# Whisper Fine-tuning Project - Amharic Adaptation

## Project Overview
Fine-tuning OpenAI's Whisper model for Amharic speech recognition using LJSpeech-format datasets. Features a Gradio WebUI for training and inference, optimized for Lightning AI remote training. Supports checkpoint management, LoRA, DeepSpeed, and CTranslate2 conversion.

## Key Components

### Training
- **Gradio UI** (`app.py`): Complete training interface with progress tracking
- **Lightning Module** (`train_whisper_lightning.py`): PyTorch Lightning implementation
- **Dataset Preparation** (`prepare_ljspeech_dataset.py`): Converts LJSpeech format to HF datasets
- Uses `openai/whisper-small` base model (configurable)
- Dataset: LJSpeech-format Amharic audio + transcripts
- Supports LoRA and DeepSpeed for efficient training
- Auto-saves checkpoints to `checkpoints/`
- Final model saved to `whisper_finetuned/`

### Inference
- **Gradio UI Inference Tab**: User-friendly interface for transcription
- **Inference Utils** (`inference_utils.py`): Handles HF, CT2, and checkpoint models
- Converts PyTorch model to CTranslate2 format for optimized inference
- Uses `faster-whisper` library for CT2 models
- Model conversion: `whisper_finetuned/` → `whisper_ct2_model/`
- Supports file upload and microphone recording

## Directory Structure
- `checkpoints/`: Lightning checkpoint files (.ckpt)
- `data/`: Processed LJSpeech datasets (HF format)
- `whisper_finetuned/`: Fine-tuned Hugging Face model
- `whisper_ct2_model/`: CTranslate2 converted model for inference
- `logs/`: TensorBoard training logs

## Dependencies
- **Core**: transformers, datasets, evaluate, torch, torchaudio
- **Lightning**: lightning, torchmetrics
- **Optimization**: deepspeed, peft, bitsandbytes
- **Audio**: librosa, soundfile, ctranslate2, faster-whisper
- **UI**: gradio
- **Logging**: wandb, tensorboard

## Key Features
- **Amharic-Specific**: Ge'ez script normalization, syllable-aware tokenization
- **LJSpeech Support**: Converts TTS datasets for STT training
- **Checkpoint Management**: Resume training, delete checkpoints via UI
- **Progress Tracking**: Real-time training progress in Gradio
- **Lightning AI Ready**: Auto-detects GPUs, DeepSpeed support
- **LoRA**: Efficient fine-tuning with 99% fewer parameters
- **Mixed Precision**: 16-bit training for 2-3x speedup

## Training Metrics
- WER (Word Error Rate) as primary evaluation metric
- Target: < 30% WER on clean Amharic speech
- Model quantization: int8 for CT2 conversion
- Audio preprocessing: 16kHz sampling rate, mono channel

## Usage Notes
- Start with Gradio UI: `python app.py`
- Prepare LJSpeech dataset before training
- Use checkpoints for resuming interrupted training
- Enable LoRA for faster experimentation
- Convert to CT2 for production deployment

## Recent Fixes (Post-Review)
- Fixed double label decoding bug in validation step
- Added null check for best_model_score
- Added path validation for dataset and models
- Implemented GradioProgressCallback for real-time progress updates
- Fixed model_type parameter handling with 'auto' detection
- Improved checkpoint dropdown to handle empty state
- Removed unused imports