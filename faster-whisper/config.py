"""Centralized configuration for Whisper fine-tuning project"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints"))
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "whisper_finetuned"))
CT2_MODEL_DIR = os.getenv("CT2_MODEL_DIR", str(PROJECT_ROOT / "whisper_ct2_model"))

# Audio settings
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
N_MELS = int(os.getenv("N_MELS", "80"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "30"))  # seconds

# VAD settings
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
MIN_SPEECH_DURATION_MS = int(os.getenv("MIN_SPEECH_DURATION_MS", "250"))
MIN_SILENCE_DURATION_MS = int(os.getenv("MIN_SILENCE_DURATION_MS", "100"))

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "100"))  # MB
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# CORS settings (configure for production)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Gradio settings
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# Training settings
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))
DEFAULT_LEARNING_RATE = float(os.getenv("DEFAULT_LEARNING_RATE", "1e-5"))
DEFAULT_NUM_EPOCHS = int(os.getenv("DEFAULT_NUM_EPOCHS", "20"))

# Monitoring settings
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8001"))

# WebSocket settings
MAX_WEBSOCKET_BUFFER_SIZE = int(os.getenv("MAX_WEBSOCKET_BUFFER_SIZE", "1000"))
WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "300"))  # seconds

# Model optimization settings
ONNX_OPSET_VERSION = int(os.getenv("ONNX_OPSET_VERSION", "14"))
QUANTIZATION_MODE = os.getenv("QUANTIZATION_MODE", "dynamic")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create directories if they don't exist
for directory in [CHECKPOINT_DIR, DATA_DIR, MODEL_DIR, CT2_MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)
