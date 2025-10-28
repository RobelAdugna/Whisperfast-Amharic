# Multi-stage Dockerfile for Amharic Whisper fine-tuning

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Training image
FROM base as training

# Copy application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Expose TensorBoard port
EXPOSE 6006

# Run Gradio app
CMD ["python", "app.py"]

# Stage 3: Inference-only image (smaller)
FROM base as inference

# Copy only inference-related files
COPY inference_utils.py .
COPY api/ ./api/
COPY utils/ ./utils/
COPY ui_components/ ./ui_components/

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Production image with both UI and API
FROM base as production

# Copy all application code
COPY . .

# Expose all ports
EXPOSE 7860 8000 6006

# Install supervisor for process management
RUN pip install supervisor

# Copy supervisor config
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run supervisor
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
