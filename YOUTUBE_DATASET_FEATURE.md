# YouTube Dataset Preparation Feature

## Overview

A new SOTA-level YouTube dataset preparation feature has been added to the Gradio WebUI for creating high-quality Amharic Whisper training datasets from YouTube videos with subtitles.

## Features

### ✅ Core Capabilities
- **Automatic Amharic Subtitle Detection**: Checks if videos have Amharic SRT (manual or auto-generated)
- **Video Download**: Downloads YouTube videos with optimal audio quality
- **Background Music Removal**: Uses Demucs to extract vocals only for cleaner training data
- **SRT Parsing**: Extracts and validates Amharic subtitles with precise timestamps
- **Audio Segmentation**: Creates individual audio segments aligned with subtitles
- **Quality Control**: Filters segments by duration (1-25 seconds) and Amharic content validation
- **Batch Processing**: Process multiple videos at once to create large datasets
- **Dataset Manifest**: Generates JSON manifests compatible with Whisper training pipeline

## Installation

### 1. Update Dependencies

The required packages have been added to `requirements.txt`:

```bash
cd faster-whisper
pip install -r requirements.txt
```

**Key new dependencies:**
- `yt-dlp>=2024.0.0` - YouTube video/subtitle downloader
- `demucs>=4.0.0` - Background music separation
- `ffmpeg-python>=0.2.0` - Audio processing

### 2. Install FFmpeg (if not already installed)

**Windows:**
```powershell
# Using chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Linux (Lightning AI):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## Usage

### Accessing the Feature

1. Start the Gradio WebUI:
```bash
cd faster-whisper
python app.py
```

2. Navigate to the **🎬 YouTube Dataset** tab

### Single Video Processing

1. **Check Video**:
   - Enter a YouTube URL
   - Click "🔍 Check Video" to verify Amharic subtitle availability
   - Review video information and subtitle types

2. **Configure Settings**:
   - **Output Directory**: Where to save the dataset (default: `./data/youtube_amharic`)
   - **Background Music Removal**: Enable Demucs processing (recommended but slower)
   - **Min/Max Duration**: Segment duration constraints (default: 1-25 seconds)

3. **Process**:
   - Click "🚀 Process Video & Create Dataset"
   - Monitor progress bar
   - Review dataset preview showing first 10 segments

### Batch Processing

1. Enter multiple YouTube URLs (one per line)
2. Configure the same settings as single video
3. Click "🚀 Process All Videos"
4. Get combined manifest with all segments

## Output Structure

```
data/youtube_amharic/
├── audio/
│   └── {video_id}.wav          # Full cleaned audio
├── subtitles/
│   └── {video_id}.srt          # Original subtitles
├── segments/
│   ├── {video_id}_seg_0000.wav # Individual segments
│   ├── {video_id}_seg_0001.wav
│   └── ...
├── {video_id}_manifest.json     # Dataset manifest
└── combined_manifest.json       # Combined manifest (batch)
```

## Dataset Manifest Format

```json
[
  {
    "audio_path": "segments/video_id_seg_0000.wav",
    "text": "የአማርኛ ጽሁፍ...",
    "duration": 3.5,
    "start": 10.2,
    "end": 13.7,
    "video_id": "video_id",
    "segment_id": 0
  }
]
```

## Integration with Training Pipeline

### Use with Existing Training

The generated manifest is compatible with the Amharic dataset processor:

```python
from utils.amharic_dataset import AmharicDatasetProcessor

processor = AmharicDatasetProcessor(
    data_dir="./data/youtube_amharic"
)

# Analyze the YouTube dataset
stats = processor.analyze_dataset("./data/youtube_amharic/combined_manifest.json")

# Create train/val/test splits
processor.create_balanced_splits(
    manifest_path="./data/youtube_amharic/combined_manifest.json",
    output_dir="./data/youtube_amharic_splits"
)
```

### Combine with Other Datasets

```python
import json

# Load multiple manifests
youtube_data = json.load(open("data/youtube_amharic/combined_manifest.json"))
existing_data = json.load(open("data/amharic/train_manifest.json"))

# Combine
combined = youtube_data + existing_data

# Save
with open("data/combined_dataset.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
```

## Technical Details

### Audio Processing Pipeline

1. **Download**: Best audio quality from YouTube (yt-dlp)
2. **Music Removal** (optional): Demucs HTDemucs model extracts vocals
3. **Resampling**: Convert to 16kHz mono (Whisper standard)
4. **Segmentation**: Extract segments based on SRT timestamps
5. **Validation**: Filter by duration and Amharic content

### Demucs Background Removal

Demucs uses state-of-the-art source separation to isolate vocals:
- **Model**: HTDemucs (Hybrid Transformer Demucs)
- **Mode**: Two-stems (vocals + other)
- **Quality**: Professional-grade separation
- **Speed**: ~1-2x real-time on GPU, slower on CPU

**Note**: Demucs can be disabled for faster processing if videos already have clean audio.

## Best Practices

### Finding Quality Videos

1. **Look for**: Educational content, news, podcasts, interviews
2. **Check CC**: Videos with manual subtitles (CC) are higher quality
3. **Verify**: Use "Check Video" before processing
4. **Avoid**: Music videos, low-quality audio, code-switching content

### Optimization Tips

1. **GPU Acceleration**: Demucs runs much faster on GPU
2. **Batch Processing**: Process multiple videos overnight
3. **Storage**: Plan for ~100MB per hour of video
4. **Quality Check**: Review samples before using for training

## Troubleshooting

### Common Issues

**1. "No Amharic subtitles found"**
- Video doesn't have Amharic SRT/CC
- Try enabling auto-generated captions

**2. "Demucs not installed"**
- Install: `pip install demucs`
- Or disable background music removal

**3. "FFmpeg not found"**
- Install FFmpeg system-wide
- Windows: `choco install ffmpeg`
- Linux: `apt-get install ffmpeg`

**4. SSH Permission Denied (Lightning AI)**
- Manual pull required on Lightning AI web interface
- Or set up SSH keys properly

## Remote Testing on Lightning AI

Since SSH authentication needs to be configured, test on Lightning AI manually:

```bash
# On Lightning AI terminal (via web interface):
cd /teamspace/studios/this_studio/Whisperfast-Amharic
git pull origin main

# Install new dependencies
pip install yt-dlp demucs ffmpeg-python

# Test the feature
python faster-whisper/app.py
```

## Example Workflow

```bash
# 1. Find Amharic videos with subtitles on YouTube
# 2. Start the WebUI
python faster-whisper/app.py

# 3. Go to YouTube Dataset tab
# 4. Check videos for Amharic subtitles
# 5. Process videos (single or batch)
# 6. Review generated dataset
# 7. Combine with existing data if needed
# 8. Create train/val/test splits
# 9. Start training!
```

## Performance

### Expected Processing Times

- **Check Video**: ~2-5 seconds
- **Download**: Depends on video length and internet speed
- **Demucs Processing**: ~1-2x video duration (GPU) or 5-10x (CPU)
- **Segmentation**: ~10-30 seconds per video

### Storage Requirements

- **Raw Audio**: ~10MB per minute
- **Segments**: ~5-8MB per minute (after filtering)
- **Demucs Temp**: ~50MB per minute (cleaned up after)

## Future Enhancements

Potential improvements:
- [ ] Multi-language support
- [ ] Quality scoring per segment
- [ ] Speaker diarization
- [ ] Parallel Demucs processing
- [ ] Cloud storage integration
- [ ] Automatic dataset balancing

## Files Modified/Created

### New Files
- `faster-whisper/utils/youtube_dataset.py` - Main YouTube processing module

### Modified Files
- `faster-whisper/app.py` - Added YouTube Dataset tab and UI
- `faster-whisper/requirements.txt` - Added dependencies

## Git Commit

```bash
commit 3418293
Author: Your Name
Date: Today

    feat: Add YouTube dataset preparation feature with Amharic SRT detection and background music removal

    - Add YouTubeDatasetPreparator class with full processing pipeline
    - Integrate new tab in Gradio WebUI with single and batch processing
    - Add yt-dlp, demucs, and ffmpeg-python dependencies
    - Support automatic Amharic subtitle detection and validation
    - Implement background music removal using Demucs
    - Create precise audio segments aligned with SRT timestamps
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review error messages in WebUI
3. Test with a known working Amharic video first
4. Verify all dependencies are installed correctly

## License

This feature follows the same license as the main project.
