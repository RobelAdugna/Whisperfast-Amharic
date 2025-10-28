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

## SOTA Enhancements (Implemented)
- Enhanced Gradio UI with 6 tabs: Training, Inference, Streaming, Chat, Metrics Dashboard
- Real-time streaming transcription with Silero VAD
- Interactive waveform and spectrogram visualizations using Plotly
- Multimodal chat interface supporting text and voice input
- FastAPI REST API with /transcribe and /transcribe/batch endpoints
- WebSocket support for real-time streaming
- Audio augmentation pipeline (noise, time stretch, pitch shift, SpecAugment)
- Amharic-specific text processing (Ge'ez script, numeral conversion, normalization)
- Prometheus metrics collection and monitoring
- ONNX export and INT8 quantization for model optimization
- Docker deployment with multi-stage builds and docker-compose
- Nginx reverse proxy with rate limiting and SSL/TLS support
- Supervisor process management for production
- Modern custom theme with professional styling

## Amharic-Specific Optimizations (150-hour Dataset)
- Advanced Amharic tokenizer with full Ge'ez script support (4 Unicode blocks)
- Ethiopic numeral normalization (፩-፼ to 1-10000)
- Ge'ez punctuation handling (።፣፤፥፦፧፨)
- Dialect detection (Gonder, Shewa, Gojjam, Wollo, Harari)
- Code-switching detection (Amharic-English)
- Syllable-level error rate (SER) metric for Ge'ez characters
- Character Error Rate (CER) and Word Error Rate (WER)
- Comprehensive error pattern analysis
- Dataset quality analysis and filtering (150h → ~135h high-quality)
- Stratified train/val/test splits (90%/5%/5%)
- Speaker-balanced data loading
- Optimized training config for medium-resource scenarios (whisper-medium)
- Domain-aware processing (conversational, news, religious, formal)

## Dataset Requirements for 150h Amharic
- Manifest format: JSON or CSV with audio_path, text, speaker_id
- Audio: 16kHz WAV files, 1-25 seconds duration
- Text: UTF-8 Ge'ez script, normalized
- Minimum quality score: 0.8 (80% Amharic content)
- Recommended: ~10-100 speakers for diversity
- Use `amharic_150h_config.yaml` for optimal training

## Enhanced Features Flag
- Set `ENHANCED_FEATURES_AVAILABLE = True` when all dependencies are installed
- Features gracefully degrade if dependencies missing
- Run `pip install -r requirements.txt` to install all enhanced dependencies