# Quick Start Guide - Amharic Whisper Fine-Tuning

## 🚀 Choose Your Environment

### Option A: Google Colab (Recommended for Beginners)

```python
# 1. Open the Colab notebook
#    Amharic_Whisper_Colab_Setup.ipynb

# 2. Select GPU runtime
#    Runtime > Change runtime type > GPU (T4)

# 3. Run all cells
#    Runtime > Run all

# 4. Wait ~24-36 hours for training
```

### Option B: Local/Server

```bash
# 1. Clone repository
git clone https://github.com/yourusername/whisper-amharic.git
cd whisper-amharic/faster-whisper

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset
python prepare_amharic_dataset.py \
  --data_dir /path/to/data \
  --manifest /path/to/manifest.json \
  --filter --split

# 4. Train
python train_whisper_lightning.py \
  --config config/amharic_150h_config.yaml

# 5. Evaluate
python evaluate_amharic.py \
  --model_path ./whisper_finetuned \
  --test_manifest data/amharic/test_manifest.json
```

## 📋 Requirements Comparison

| Environment | File | Install Command |
|------------|------|----------------|
| **Local/Server** | `requirements.txt` | `pip install -r requirements.txt` |
| **Google Colab** | `requirements-colab.txt` | `!pip install -q -r requirements-colab.txt` |
| **Docker** | Built into image | `docker-compose up` |

## 💡 Which File to Use?

### Use `requirements.txt` if:
- Running on local machine
- Running on cloud server (AWS, Azure, GCP)
- Using Docker
- Need full production stack

### Use `requirements-colab.txt` if:
- Using Google Colab
- Using Kaggle Notebooks
- Limited to specific CUDA version
- Need minimal setup

## ⚡ Quick Commands

### Dataset Preparation
```bash
# Analyze
python prepare_amharic_dataset.py --data_dir DATA --manifest MANIFEST --analyze

# Filter & Split
python prepare_amharic_dataset.py --data_dir DATA --manifest MANIFEST --filter --split
```

### Training
```bash
# Local
python train_whisper_lightning.py --config config/amharic_150h_config.yaml

# Colab (from notebook)
!python train_whisper_lightning.py --config config/amharic_colab_config.yaml
```

### Evaluation
```bash
python evaluate_amharic.py --model_path MODEL --test_manifest TEST
```

### Launch UI
```bash
# Local
python app.py

# Colab (public URL)
!python app.py --share
```

## 🔧 Configuration Files

| File | Purpose | Use Case |
|------|---------|----------|
| `config/amharic_150h_config.yaml` | Local training | Full GPU, 150h dataset |
| `config/amharic_colab_config.yaml` | Colab training | T4 GPU, limited RAM |
| `config/amharic_config.yaml` | Legacy | Basic setup |

## 📦 Expected File Sizes

```
requirements.txt        ~2 KB (65 packages)
requirements-colab.txt  ~3 KB (50 packages + notes)
wisper-medium model     ~3 GB
150h dataset           ~15-20 GB
Checkpoints (per epoch) ~3 GB
```

## 🎯 Expected Results

| Dataset Size | Model | Expected WER | Training Time (V100) |
|--------------|-------|-------------|---------------------|
| 150h | whisper-small | 18-22% | 12-18h |
| 150h | **whisper-medium** | **12-16%** | **24-36h** |
| 150h | whisper-large | 10-14% | 48-72h |

**Baseline (zero-shot):** 35-45% WER

## 🆘 Common Issues

### "No module named 'utils.amharic_tokenizer'"
```bash
# Make sure __init__.py files exist
touch utils/__init__.py ui_components/__init__.py api/__init__.py
```

### "CUDA out of memory"
```yaml
# Reduce batch size in config
batch_size: 8  # or 4
gradient_accumulation_steps: 8  # or 16
```

### "ModuleNotFoundError: No module named 'sklearn'"
```bash
# Install scikit-learn
pip install scikit-learn>=1.3.0
```

### Colab: "Session crashed"
```python
# Save checkpoints to Drive first
from google.colab import drive
drive.mount('/content/drive')
!cp -r checkpoints /content/drive/MyDrive/whisper_checkpoints/
```

## 📚 Documentation

- **Full Guide**: `docs/AMHARIC_GUIDE.md`
- **Deployment**: `docs/DEPLOYMENT.md`
- **Implementation**: `docs/IMPLEMENTATION_SUMMARY.md`
- **Colab Setup**: `Amharic_Whisper_Colab_Setup.ipynb`
- **README**: `AMHARIC_README.md`

## ✅ Verification Checklist

Before training:
- [ ] GPU available (`nvidia-smi`)
- [ ] Dependencies installed
- [ ] Dataset prepared (train/val/test splits)
- [ ] Config file updated
- [ ] Checkpoints directory created

After training:
- [ ] Model checkpoints saved
- [ ] Evaluation results < 20% WER
- [ ] Model exported to CT2
- [ ] Gradio UI works

## 🎓 Next Steps

1. **After training**: Run evaluation
2. **If WER too high**: Check data quality, increase epochs
3. **For production**: Export to CTranslate2, deploy with Docker
4. **For sharing**: Push model to Hugging Face Hub

---

**Need help?** Check `docs/AMHARIC_GUIDE.md` for detailed instructions.
