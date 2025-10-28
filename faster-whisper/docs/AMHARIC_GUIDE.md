# Amharic Whisper Fine-Tuning Guide (150-hour Dataset)

## Overview

This guide covers fine-tuning Whisper for Amharic (አማርኛ) speech recognition using your 150-hour dataset with state-of-the-art optimizations.

## Prerequisites

### Data Format

Your dataset should be organized as:

```
data/
├── audio/
│   ├── speaker_001_001.wav
│   ├── speaker_001_002.wav
│   └── ...
└── manifest.json (or manifest.csv)
```

### Manifest Format

**JSON Format** (recommended):
```json
[
  {
    "audio_path": "audio/speaker_001_001.wav",
    "text": "ሰላም እንዴት ነህ",
    "speaker_id": "speaker_001",
    "duration": 3.5,
    "domain": "conversational"
  },
  ...
]
```

**CSV Format**:
```csv
audio_path,text,speaker_id,duration,domain
audio/speaker_001_001.wav,"ሰላም እንዴት ነህ",speaker_001,3.5,conversational
...
```

## Step-by-Step Guide

### Step 1: Install Dependencies

```bash
cd faster-whisper
pip install -r requirements.txt

# Additional Amharic dependencies
pip install ethiopic
```

### Step 2: Analyze Your Dataset

```bash
python prepare_amharic_dataset.py \
  --data_dir /path/to/your/data \
  --manifest /path/to/manifest.json \
  --analyze
```

This will show:
- Total duration and sample count
- Duration distribution
- Text statistics
- Code-switching rate
- Quality issues
- Dialect distribution
- Speaker statistics

### Step 3: Filter Low-Quality Samples

```bash
python prepare_amharic_dataset.py \
  --data_dir /path/to/your/data \
  --manifest /path/to/manifest.json \
  --output_dir ./data/amharic \
  --filter \
  --quality_threshold 0.8 \
  --min_duration 1.0 \
  --max_duration 25.0
```

**Filtering Options:**
- `--remove_code_switching`: Remove Amharic-English mixed samples
- `--quality_threshold`: Minimum Amharic content ratio (0-1)
- `--min_duration`: Remove very short audio
- `--max_duration`: Remove very long audio

### Step 4: Create Train/Val/Test Splits

```bash
python prepare_amharic_dataset.py \
  --data_dir /path/to/your/data \
  --manifest /path/to/manifest.json \
  --output_dir ./data/amharic \
  --filter \
  --split \
  --train_ratio 0.9 \
  --val_ratio 0.05 \
  --test_ratio 0.05
```

Output:
- `data/amharic/train_manifest.json` (~135h)
- `data/amharic/val_manifest.json` (~7.5h)
- `data/amharic/test_manifest.json` (~7.5h)

### Step 5: Configure Training

Edit `config/amharic_150h_config.yaml`:

```yaml
data:
  train_manifest: "data/amharic/train_manifest.json"
  val_manifest: "data/amharic/val_manifest.json"
  test_manifest: "data/amharic/test_manifest.json"

model:
  name: "openai/whisper-medium"  # Recommended for 150h
  
training:
  num_epochs: 15
  batch_size: 16  # Adjust for your GPU
  learning_rate: 5e-5
```

### Step 6: Start Training

```bash
python train_whisper_lightning.py \
  --config config/amharic_150h_config.yaml
```

**Training Tips:**
- Monitor `val_wer` (Word Error Rate)
- Expected WER: <15% for 150h dataset
- Training time: ~24-48 hours on V100/A100
- Use mixed precision (`precision: 16`) for 2x speedup

### Step 7: Evaluate

```bash
python evaluate_amharic.py \
  --model_path ./whisper_finetuned \
  --test_manifest data/amharic/test_manifest.json
```

Metrics:
- **WER** (Word Error Rate): Primary metric
- **CER** (Character Error Rate): For Ge'ez script
- **SER** (Syllable Error Rate): Amharic-specific

## Amharic-Specific Features

### 1. Ge'ez Script Normalization

Automatically handles:
- Ethiopic numerals (፩-፼ → 1-10000)
- Ethiopic punctuation (።፣፤፥፦፧)
- Unicode normalization (NFC)
- Gemination markers

### 2. Dialect Support

Detects and handles:
- **Gonder** (ጎንደር): Northern dialect
- **Shewa** (ሸዋ): Central/standard
- **Gojjam** (ጎጃም): Western
- **Wollo** (ወሎ): Eastern
- **Harari** (ሐረር): City dialect

### 3. Code-Switching

Handles Amharic-English mixing:
- Detects code-switched segments
- Optional filtering
- Preserves for realistic scenarios

### 4. Data Augmentation

Applied during training:
- Time stretching (0.9x-1.1x)
- Pitch shifting (±2 semitones)
- Background noise
- SpecAugment

## Expected Results

### With 150-hour Dataset

| Model | Expected WER | CER | Training Time |
|-------|--------------|-----|---------------|
| whisper-small | 18-22% | 8-10% | 12-18h |
| whisper-medium | 12-16% | 5-7% | 24-36h |
| whisper-large | 10-14% | 4-6% | 48-72h |

### Comparison

- **Baseline Whisper (zero-shot)**: ~35-45% WER
- **Fine-tuned (150h)**: ~12-16% WER
- **Improvement**: ~60-70% relative WER reduction

## Troubleshooting

### Low Amharic Quality Score

**Problem**: Many samples filtered due to quality

**Solutions**:
- Lower `--quality_threshold` to 0.7
- Check text encoding (must be UTF-8)
- Verify Ge'ez script in transcripts

### Code-Switching Issues

**Problem**: High code-switching rate

**Solutions**:
- Keep code-switching if realistic
- Or filter with `--remove_code_switching`
- Handle in post-processing

### Imbalanced Speakers

**Problem**: Few speakers dominate dataset

**Solutions**:
- Use `balance_speakers: true` in config
- Set `max_samples_per_speaker: 1000`
- Collect more diverse speakers

### OOM (Out of Memory)

**Problem**: GPU runs out of memory

**Solutions**:
- Reduce `batch_size` (try 8 or 4)
- Increase `gradient_accumulation_steps`
- Use `precision: 16` (mixed precision)
- Try whisper-small instead of medium

## Advanced Usage

### Custom Domain Adaptation

```yaml
amharic_specific:
  domain_tags: ["conversational", "news", "religious", "formal"]
```

### Dialect-Specific Training

```yaml
amharic_specific:
  recognize_dialects: true
  dialect_tags: ["gonder", "shewa"]
```

### Multi-GPU Training

```yaml
compute:
  strategy: "ddp"
  devices: 4  # Use 4 GPUs
```

## Best Practices

### Data Quality

1. **Audio**:
   - 16kHz sample rate
   - Mono channel
   - SNR > 15dB
   - Minimal background noise

2. **Transcripts**:
   - UTF-8 encoding
   - Proper Ge'ez script
   - Consistent normalization
   - Accurate alignment

3. **Speaker Diversity**:
   - 10-100 speakers minimum
   - Gender balance
   - Age diversity
   - Dialect coverage

### Training Strategy

1. **Start Small**: Train with whisper-small first
2. **Monitor Metrics**: Track WER, CER, SER
3. **Iterate**: Adjust based on validation results
4. **Test Thoroughly**: Use diverse test set

### Deployment

1. **Export**: Convert to CTranslate2 for speed
2. **Quantize**: INT8 for 4x speedup
3. **Optimize**: Batch inference when possible
4. **Monitor**: Track production metrics

## Resources

### Amharic Language Resources

- **Unicode**: U+1200 to U+137F (Ethiopic)
- **Common Words**: ~50,000 in modern Amharic
- **Speakers**: ~32 million (Ethiopia, diaspora)

### Useful Links

- Ge'ez Unicode: https://unicode.org/charts/PDF/U1200.pdf
- Ethiopic Script: https://en.wikipedia.org/wiki/Ge%CA%BDez_script
- Amharic Phonology: https://en.wikipedia.org/wiki/Amharic#Phonology

## Citation

If you use this implementation, please cite:

```bibtex
@software{whisper_amharic_2024,
  title={SOTA Whisper Fine-Tuning for Amharic},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/whisper-amharic}
}
```

## Support

For issues or questions:
1. Check troubleshooting section
2. Review dataset statistics
3. Verify configuration
4. Check logs for errors
