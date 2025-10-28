# Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (for GPU deployment)
- At least 8GB RAM
- 20GB free disk space

## Quick Start

### 1. Install Dependencies (Local Development)

```bash
cd faster-whisper
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
# .env
CHECKPOINT_DIR=./checkpoints
DATA_DIR=./data
MODEL_DIR=./whisper_finetuned

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:7860,http://localhost:3000

# Gradio Settings
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# VAD Settings
VAD_THRESHOLD=0.5

# Enable metrics
METRICS_ENABLED=true
```

### 3. Run Locally

```bash
# Run Gradio UI
python app.py

# In another terminal, run API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Docker Deployment

### Development Mode

```bash
cd docker
docker-compose up gradio
```

Access UI at http://localhost:7860

### Production Mode (Full Stack)

```bash
cd docker
docker-compose up -d
```

Services:
- **Gradio UI**: http://localhost:7860
- **API**: http://localhost:8000
- **TensorBoard**: http://localhost:6006
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### API Only (Inference)

```bash
cd docker
docker-compose up -d api
```

## Production Best Practices

### 1. Security

**Update CORS settings**:
```python
# In config.py or .env
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

**Add API authentication**:
```python
# Example with API key
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/transcribe", dependencies=[Depends(verify_api_key)])
async def transcribe(...):
    ...
```

**Enable SSL/TLS**:
1. Obtain SSL certificates (Let's Encrypt)
2. Update nginx.conf to enable HTTPS
3. Uncomment HTTPS server block

### 2. Monitoring

**Configure Grafana dashboards**:
1. Access Grafana at http://localhost:3000
2. Add Prometheus data source (http://prometheus:9090)
3. Import dashboard templates from `docs/grafana/`

### 3. Scaling

**Horizontal scaling with Docker Compose**:
```yaml
services:
  api:
    deploy:
      replicas: 3
```

**Load balancing**:
- Update nginx.conf to add multiple upstream servers
- Use Docker Swarm or Kubernetes for orchestration

### 4. Performance Tuning

**GPU optimization**:
```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Adjust worker processes**:
```bash
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Common Issues

**1. GPU not detected**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**2. Port already in use**
```bash
# Change ports in docker-compose.yml or .env
GRADIO_SERVER_PORT=7861
API_PORT=8001
```

**3. Out of memory**
- Reduce batch size in training config
- Use model quantization
- Increase Docker memory limit

**4. Slow inference**
- Use CTranslate2 models instead of PyTorch
- Enable GPU acceleration
- Reduce beam size

### Logs

**View logs**:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# With supervisor (production image)
docker exec whisper-gradio tail -f /var/log/supervisor/gradio.log
```

## Backup & Restore

### Backup

```bash
# Backup checkpoints
tar -czf checkpoints-backup-$(date +%Y%m%d).tar.gz checkpoints/

# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz whisper_finetuned/ whisper_ct2_model/
```

### Restore

```bash
# Restore checkpoints
tar -xzf checkpoints-backup-20240101.tar.gz

# Restore models
tar -xzf models-backup-20240101.tar.gz
```

## Updating

```bash
# Pull latest changes
git pull

# Rebuild Docker images
cd docker
docker-compose build

# Restart services
docker-compose up -d
```

## Health Checks

```bash
# API health
curl http://localhost:8000/health

# Gradio health  
curl http://localhost:7860/

# Prometheus targets
curl http://localhost:9090/api/v1/targets
```

## Support

For issues, check:
1. Application logs
2. Docker logs
3. Prometheus metrics
4. Grafana dashboards
