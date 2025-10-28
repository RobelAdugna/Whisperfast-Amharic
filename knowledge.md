# Whisper Amharic Fine-tuning Project

## Mission
Fine-tune OpenAI's Whisper model for Amharic speech recognition using PyTorch Lightning and faster-whisper.

## Project Structure
- `faster-whisper/`: Main implementation directory with training scripts, inference utilities, and model checkpoints
- `scripts/`: Dataset preparation and training scripts
- `config/`: YAML configuration files for training parameters

## Key Technologies
- PyTorch Lightning for training framework
- Hugging Face Transformers for Whisper model
- faster-whisper for optimized inference
- CTranslate2 for model conversion

## Training Workflow
1. Prepare dataset using `prepare_dataset.py` or `prepare_ljspeech_dataset.py`
2. Configure training parameters in `config/amharic_config.yaml`
3. Train using `train_whisper_lightning.py` with Lightning
4. Convert checkpoints to CTranslate2 format for faster inference
5. Run inference with `app.py` or `inference_utils.py`

## Model Outputs
- `whisper_checkpoints/`: PyTorch Lightning checkpoint files (.pt)
- `whisper_finetuned/`: Hugging Face format model
- `whisper_ct2_model/`: CTranslate2 optimized model

## Notes
- Project uses Windows paths (OS: win32)
- Training checkpoints are saved at regular intervals (epochs 33-35 visible)
- Model supports both batch and real-time inference modes