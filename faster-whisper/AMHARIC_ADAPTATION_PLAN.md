# Amharic Whisper Fine-Tuning Adaptation Plan

## 1. Dataset Preparation Module
### Objective
Convert LJSpeech-format Amharic TTS dataset to Whisper-compatible format

### Implementation Steps
- Create `prepare_ljspeech_dataset.py` script:
  - Parse metadata.csv (pipe-separated: ID|transcript)
  - Map audio file paths from wavs/ directory
  - Resample all audio to 16kHz mono using torchaudio
  - Create Hugging Face Dataset with audio and text columns
  - Split into train/validation/test (90/5/5)
  - Cast audio column with Audio(sampling_rate=16000)
  - Save to disk for training

### Key Functions
```python
def load_ljspeech_metadata(path):
    # Parse metadata.csv
    return pd.DataFrame with id, transcript, audio_path

def resample_audio(audio_path):
    # Load and resample to 16kHz
    return resampled_array

def create_hf_dataset(df):
    # Convert to Hugging Face Dataset format
    return Dataset with audio and text columns
```

## 2. Amharic Tokenizer Adaptation
### Objective
Extend Whisper tokenizer for Amharic abugida script

### Implementation Steps
- Create `setup_amharic_tokenizer.py`:
  - Load base WhisperTokenizer from openai/whisper-small
  - Add Amharic Ge'ez characters (~300-400 characters)
  - Train custom BPE on Amharic corpus if available
  - Configure processor with language="am" and task="transcribe"
  - Save extended tokenizer

### Amharic-Specific Processing
- Implement homophone normalization function
- Unicode NFC normalization for Ge'ez script
- Syllable-aware tokenization preservation

## 3. Lightning AI Training Script
### Objective
Create PyTorch Lightning training module optimized for remote execution

### Implementation Steps
- Create `train_whisper_lightning.py`:
  - Implement `WhisperLightningModule` with:
    - Model: WhisperForConditionalGeneration
    - Training step with loss computation
    - Validation step with WER metric
    - Configure optimizers (AdamW with 1e-5 lr)
  - Implement `AmharicDataModule` with:
    - setup() for loading dataset
    - train/val/test dataloaders
    - Audio preprocessing pipeline
  - Configure Lightning Trainer:
    - accelerator="auto" for GPU detection
    - devices="auto" for multi-GPU
    - precision=16 for mixed precision
    - strategy="deepspeed_stage_2" for memory optimization
    - ModelCheckpoint callback for best models
    - EarlyStopping callback (patience=3)
    - WandB logger for experiment tracking

### Training Configuration
```python
trainer = L.Trainer(
    accelerator="auto",
    devices="auto",
    precision=16,
    strategy="deepspeed_stage_2",
    max_epochs=20,
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,
    callbacks=[checkpoint, early_stop],
    logger=wandb_logger
)
```

## 4. Inference Script for CTranslate2
### Objective
Adapt inference notebook for Amharic with optimized conversion

### Implementation Steps
- Modify `inference.ipynb`:
  - Convert fine-tuned model to CTranslate2 format
  - Update language setting to Amharic ("am")
  - Configure faster-whisper with int8 quantization
  - Add Amharic text post-processing
  - Support both file and microphone input

## 5. Lightning AI Configuration
### Objective
Setup for remote training on Lightning AI Studio

### Implementation Steps
- Create `lightning_config.yaml`:
  - Specify compute resources (GPU type, memory)
  - Set environment variables
  - Configure data paths for cloud storage
  - Enable checkpointing to cloud buckets
- Update `requirements.txt` with all dependencies:
  - lightning, transformers, datasets
  - torch, torchaudio, librosa
  - ctranslate2, faster-whisper
  - jiwer, evaluate, wandb
  - deepspeed, peft (for LoRA)

## 6. Documentation Updates
### Objective
Update all documentation for Amharic adaptation

### Files to Update
- README.md: Replace Turkish with Amharic throughout
- knowledge.md: Update with Amharic-specific notes
- Add SETUP.md with step-by-step instructions
- Add DATA_PREPARATION.md for LJSpeech conversion

## 7. Testing and Validation
### Steps
- Test dataset preparation with sample data
- Validate tokenizer on Amharic text samples
- Run small training test (1 epoch, subset)
- Verify WER computation on validation set
- Test CTranslate2 conversion and inference

## Expected Outcomes
- WER < 30% on clean Amharic speech
- Training time: 2-4 hours on A100 GPU
- Model size: ~500MB fine-tuned, ~150MB CTranslate2
- Inference: Real-time on GPU, 2-3x slower on CPU