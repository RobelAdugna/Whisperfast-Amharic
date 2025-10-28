import gradio as gr
import os
import json
import torch
from pathlib import Path
import lightning as L
from datetime import datetime
from typing import Dict, List, Optional
import glob
import numpy as np

# Import training and inference modules
from train_whisper_lightning import WhisperLightningModule, AmharicDataModule, start_training
from inference_utils import load_model_for_inference, transcribe_audio
from prepare_ljspeech_dataset import prepare_dataset

# Import new utilities and components
try:
    from utils.vad import VADProcessor
    from utils.audio_augmentation import AudioAugmentor
    from utils.amharic_processing import AmharicTextProcessor
    from utils.monitoring import MetricsCollector
    from utils.youtube_dataset import YouTubeDatasetPreparator
    from ui_components.waveform import create_waveform_plot, create_spectrogram_plot, create_audio_stats_display
    from ui_components.metrics_dashboard import create_training_metrics_plot, create_realtime_loss_plot
    from ui_components.chat_interface import create_chat_interface, create_streaming_chat
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

# Global state for managing checkpoints and models
CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data"
MODEL_DIR = "./whisper_finetuned"
CT2_MODEL_DIR = "./whisper_ct2_model"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize enhanced components if available
if ENHANCED_FEATURES_AVAILABLE:
    vad_processor = VADProcessor()
    audio_augmentor = AudioAugmentor()
    amharic_processor = AmharicTextProcessor()
    metrics_collector = MetricsCollector()
else:
    vad_processor = None
    audio_augmentor = None
    amharic_processor = None
    metrics_collector = None

# Global metrics storage for live updates
training_metrics = {
    'loss': [],
    'val_loss': [],
    'wer': [],
    'lr': [],
    'grad_norm': []
}

def get_available_checkpoints() -> List[str]:
    """List all available checkpoint files"""
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.ckpt"))
    return sorted(checkpoints, key=os.path.getmtime, reverse=True)

def get_checkpoint_info(checkpoint_path: str) -> Dict:
    """Extract information from checkpoint filename"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    
    filename = os.path.basename(checkpoint_path)
    stat = os.stat(checkpoint_path)
    size_mb = stat.st_size / (1024 * 1024)
    modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        "filename": filename,
        "size_mb": f"{size_mb:.2f} MB",
        "modified": modified,
        "path": checkpoint_path
    }

def format_checkpoint_dropdown() -> List[tuple]:
    """Format checkpoints for dropdown display"""
    checkpoints = get_available_checkpoints()
    if not checkpoints:
        return []
    
    formatted = []
    for ckpt in checkpoints:
        info = get_checkpoint_info(ckpt)
        label = f"{info['filename']} ({info['size_mb']}, {info['modified']})"
        formatted.append((label, ckpt))
    
    return formatted

def load_training_config(config_path: Optional[str] = None) -> Dict:
    """Load training configuration from JSON"""
    default_config = {
        "model_name": "openai/whisper-small",
        "language": "am",
        "learning_rate": 1e-5,
        "batch_size": 16,
        "num_epochs": 20,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 4,
        "precision": 16,
        "early_stopping_patience": 3,
        "val_check_interval": 0.25,
        "save_top_k": 3,
        "data_path": DATA_DIR,
        "use_deepspeed": False,
        "use_lora": False,
        "lora_r": 16,
        "lora_alpha": 32
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            default_config.update(loaded_config)
    
    return default_config

def save_training_config(config: Dict, config_path: str = "training_config.json"):
    """Save training configuration to JSON"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return f"Configuration saved to {config_path}"

def train_model(
    data_path: str,
    model_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    gradient_clip_val: float,
    accumulate_grad_batches: int,
    precision: int,
    early_stopping_patience: int,
    save_top_k: int,
    resume_from_checkpoint: Optional[str],
    use_deepspeed: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    progress=gr.Progress()
):
    """Training function with progress tracking"""
    try:
        # Validate dataset path
        if not os.path.exists(data_path):
            return f"❌ Dataset not found at {data_path}", gr.update(), ""
        
        progress(0, desc="Initializing training...")
        
        # Create config
        config = {
            "data_path": data_path,
            "model_name": model_name,
            "language": "am",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "gradient_clip_val": gradient_clip_val,
            "accumulate_grad_batches": accumulate_grad_batches,
            "precision": precision,
            "early_stopping_patience": early_stopping_patience,
            "save_top_k": save_top_k,
            "checkpoint_dir": CHECKPOINT_DIR,
            "use_deepspeed": use_deepspeed,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha
        }
        
        # Save config
        save_training_config(config)
        
        progress(0.1, desc="Loading dataset...")
        
        # Start training
        result = start_training(
            config=config,
            resume_checkpoint=resume_from_checkpoint,
            progress_callback=progress
        )
        
        progress(1.0, desc="Training complete!")
        
        # Get best checkpoint info
        checkpoints = get_available_checkpoints()
        best_ckpt = checkpoints[0] if checkpoints else "No checkpoint saved"
        
        return (
            f"✅ Training completed successfully!\n\nBest checkpoint: {best_ckpt}\n\nMetrics:\n{json.dumps(result.get('metrics', {}), indent=2)}",
            gr.update(choices=format_checkpoint_dropdown()),
            result.get('log', '')
        )
        
    except Exception as e:
        return f"❌ Training failed: {str(e)}", gr.update(), str(e)

def prepare_data(
    ljspeech_path: str,
    output_path: str,
    train_split: float,
    val_split: float,
    progress=gr.Progress()
):
    """Prepare LJSpeech dataset for training"""
    try:
        progress(0, desc="Loading metadata...")
        
        result = prepare_dataset(
            ljspeech_path=ljspeech_path,
            output_path=output_path,
            train_split=train_split,
            val_split=val_split,
            progress_callback=progress
        )
        
        progress(1.0, desc="Dataset preparation complete!")
        
        return f"✅ Dataset prepared successfully!\n\nTrain samples: {result['train_size']}\nValidation samples: {result['val_size']}\nTest samples: {result['test_size']}"
        
    except Exception as e:
        return f"❌ Dataset preparation failed: {str(e)}"

def transcribe(
    audio_file,
    model_path: str,
    language: str,
    task: str,
    beam_size: int,
    show_waveform: bool = True,
    show_spectrogram: bool = False,
    use_vad: bool = True,
    progress=gr.Progress()
):
    """Transcribe audio using trained model with enhanced visualizations"""
    try:
        progress(0, desc="Loading model...")
        
        if not audio_file:
            return "Please upload an audio file", None, None, None, ""
        
        # Load audio for visualization
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_file)
        
        # Create visualizations if enabled
        waveform_fig = None
        spectrogram_fig = None
        stats_text = ""
        
        if ENHANCED_FEATURES_AVAILABLE and show_waveform:
            progress(0.1, desc="Creating waveform...")
            
            # Detect speech segments with VAD if enabled
            speech_segments = None
            if use_vad and vad_processor:
                speech_segments = vad_processor.get_speech_segments(audio_data)
            
            waveform_fig = create_waveform_plot(
                audio_data,
                sample_rate,
                highlight_segments=speech_segments
            )
        
        if ENHANCED_FEATURES_AVAILABLE and show_spectrogram:
            progress(0.2, desc="Creating spectrogram...")
            spectrogram_fig = create_spectrogram_plot(audio_data, sample_rate)
        
        if ENHANCED_FEATURES_AVAILABLE:
            stats = create_audio_stats_display(audio_data, sample_rate)
            stats_text = "\n".join([f"**{k}**: {v}" for k, v in stats.items()])
        
        progress(0.3, desc="Transcribing...")
        
        result = transcribe_audio(
            audio_path=audio_file,
            model_path=model_path,
            language=language,
            task=task,
            beam_size=beam_size
        )
        
        # Post-process with Amharic normalization if available
        transcript = result['text']
        if ENHANCED_FEATURES_AVAILABLE and amharic_processor and language == 'am':
            transcript = amharic_processor.normalize_text(transcript)
        
        progress(1.0, desc="Transcription complete!")
        
        return transcript, waveform_fig, spectrogram_fig, result.get('segments', []), stats_text
        
    except Exception as e:
        return f"❌ Transcription failed: {str(e)}", None, None, None, ""

def set_checkpoint_dir(new_dir: str):
    """Set a new checkpoint directory"""
    global CHECKPOINT_DIR
    try:
        if not new_dir or not new_dir.strip():
            return f"❌ Invalid directory path", gr.update()
        
        # Expand paths like ~/... or environment variables
        new_dir = os.path.expanduser(new_dir)
        new_dir = os.path.abspath(new_dir)
        
        # Create directory if it doesn't exist
        os.makedirs(new_dir, exist_ok=True)
        
        # Update global variable
        CHECKPOINT_DIR = new_dir
        
        return (
            f"✅ Checkpoint directory set to: {CHECKPOINT_DIR}",
            gr.update(choices=format_checkpoint_dropdown())
        )
    except Exception as e:
        return f"❌ Failed to set checkpoint directory: {str(e)}", gr.update()

def delete_checkpoint(checkpoint_path: str):
    """Delete a checkpoint file"""
    try:
        if not checkpoint_path:
            return "❌ No checkpoint selected", gr.update()
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            return (
                f"✅ Checkpoint deleted: {os.path.basename(checkpoint_path)}",
                gr.update(choices=format_checkpoint_dropdown())
            )
        return "❌ Checkpoint not found", gr.update()
    except Exception as e:
        return f"❌ Failed to delete checkpoint: {str(e)}", gr.update()

def check_youtube_video(url: str):
    """Check if YouTube video has Amharic subtitles"""
    try:
        if not url or not url.strip():
            return "Please enter a YouTube URL"
        
        preparator = YouTubeDatasetPreparator()
        info = preparator.check_youtube_link(url)
        
        result = f"""✅ Video Information:

📹 **Title**: {info['title']}
⏱️  **Duration**: {info['duration'] // 60}:{info['duration'] % 60:02d}
👤 **Uploader**: {info['uploader']}
🆔 **Video ID**: {info['video_id']}

**Amharic Subtitles**:
  - Manual SRT: {'✅ Available' if info['amharic_manual'] else '❌ Not available'}
  - Auto-generated: {'✅ Available' if info['amharic_auto'] else '❌ Not available'}

**All Available Subtitles**: {', '.join(info['available_subtitles']) if info['available_subtitles'] else 'None'}
**Auto Captions**: {', '.join(info['available_auto_captions']) if info['available_auto_captions'] else 'None'}
"""
        return result
    except Exception as e:
        return f"❌ Error checking video: {str(e)}"

def process_youtube_video(
    url: str,
    output_dir: str,
    use_demucs: bool,
    min_duration: float,
    max_duration: float,
    cookies_browser: str,
    progress=gr.Progress()
):
    """Process a YouTube video to create dataset"""
    try:
        if not url or not url.strip():
            return "Please enter a YouTube URL", None, ""
        
        progress(0, desc="Initializing...")
        
        preparator = YouTubeDatasetPreparator(
            output_dir=output_dir,
            use_demucs=use_demucs,
            min_segment_duration=min_duration,
            max_segment_duration=max_duration,
            cookies_from_browser=cookies_browser if cookies_browser != "none" else None
        )
        
        result = preparator.process_youtube_video(
            url=url,
            progress_callback=progress
        )
        
        if not result['success']:
            return f"❌ Processing failed: {result['error']}", None, ""
        
        # Create summary
        summary = f"""✅ YouTube Dataset Created Successfully!

📹 **Video**: {result['video_title']}
🆔 **Video ID**: {result['video_id']}
📊 **Total Segments**: {result['total_segments']}
⏱️  **Total Duration**: {result['total_duration_minutes']:.1f} minutes ({result['total_duration_seconds']:.1f} seconds)

📁 **Output Files**:
  - Dataset Manifest: `{result['manifest_path']}`
  - Full Audio: `{result['audio_path']}`
  - Subtitles: `{result['srt_path']}`
  - Segments: `{result['output_dir']}/segments/`

🎯 **Next Steps**:
  1. Review the generated segments
  2. Use the manifest file for training
  3. Combine with other datasets if needed
"""
        
        # Prepare preview data
        preview_data = []
        for i, entry in enumerate(result['dataset_entries'][:10]):  # Show first 10
            preview_data.append({
                "Segment": i + 1,
                "Duration": f"{entry['duration']:.2f}s",
                "Text": entry['text'][:50] + "..." if len(entry['text']) > 50 else entry['text'],
                "Audio File": entry['audio_path']
            })
        
        return summary, preview_data, result['manifest_path']
    
    except Exception as e:
        return f"❌ Processing failed: {str(e)}", None, ""

def process_local_file_pair(
    audio_file,
    srt_file,
    output_dir: str,
    use_demucs: bool,
    min_duration: float,
    max_duration: float,
    progress=gr.Progress()
):
    """Process local audio + SRT file pair"""
    try:
        if not audio_file or not srt_file:
            return "Please upload both audio and SRT files", None, ""
        
        progress(0, desc="Initializing...")
        
        preparator = YouTubeDatasetPreparator(
            output_dir=output_dir,
            use_demucs=use_demucs,
            min_segment_duration=min_duration,
            max_segment_duration=max_duration
        )
        
        result = preparator.process_local_files(
            audio_path=audio_file.name if hasattr(audio_file, 'name') else audio_file,
            srt_path=srt_file.name if hasattr(srt_file, 'name') else srt_file,
            progress_callback=progress
        )
        
        if not result['success']:
            return f"❌ Processing failed: {result['error']}", None, ""
        
        # Create summary
        summary = f"""✅ Local Files Dataset Created Successfully!

📁 **Files**: {result['audio_filename']} + {result['srt_filename']}
🆔 **ID**: {result['file_id']}
📊 **Total Segments**: {result['total_segments']}
⏱️  **Total Duration**: {result['total_duration_minutes']:.1f} minutes ({result['total_duration_seconds']:.1f} seconds)

📁 **Output Files**:
  - Dataset Manifest: `{result['manifest_path']}`
  - Audio: `{result['audio_path']}`
  - Subtitles: `{result['srt_path']}`
  - Segments: `{result['output_dir']}/segments/`

🎯 **Next Steps**:
  1. Review the generated segments
  2. Use the manifest file for training
  3. Combine with other datasets if needed
"""
        
        # Prepare preview data
        preview_data = []
        for i, entry in enumerate(result['dataset_entries'][:10]):  # Show first 10
            preview_data.append({
                "Segment": i + 1,
                "Duration": f"{entry['duration']:.2f}s",
                "Text": entry['text'][:50] + "..." if len(entry['text']) > 50 else entry['text'],
                "Audio File": entry['audio_path']
            })
        
        return summary, preview_data, result['manifest_path']
    
    except Exception as e:
        return f"❌ Processing failed: {str(e)}", None, ""

def process_multiple_youtube_videos(
    urls_text: str,
    output_dir: str,
    use_demucs: bool,
    min_duration: float,
    max_duration: float,
    cookies_browser: str,
    progress=gr.Progress()
):
    """Process multiple YouTube videos from text input (one URL per line)"""
    try:
        if not urls_text or not urls_text.strip():
            return "Please enter YouTube URLs (one per line)", None, ""
        
        # Parse URLs
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        if not urls:
            return "No valid URLs found", None, ""
        
        progress(0, desc=f"Processing {len(urls)} videos...")
        
        preparator = YouTubeDatasetPreparator(
            output_dir=output_dir,
            use_demucs=use_demucs,
            min_segment_duration=min_duration,
            max_segment_duration=max_duration,
            cookies_from_browser=cookies_browser if cookies_browser != "none" else None
        )
        
        result = preparator.process_multiple_videos(
            urls=urls,
            progress_callback=progress
        )
        
        # Create summary
        summary = f"""✅ Batch Processing Complete!

📊 **Statistics**:
  - Total Videos: {result['total_videos']}
  - Successful: {result['successful']}
  - Failed: {result['failed']}
  - Total Segments: {result['total_segments']}
  - Total Duration: {result['total_duration_minutes']:.1f} minutes

📁 **Combined Manifest**: `{result['combined_manifest']}`

📹 **Individual Results**:
"""
        
        for i, res in enumerate(result['results'], 1):
            if res['success']:
                summary += f"\n  {i}. ✅ {res['video_title'][:40]}... ({res['total_segments']} segments)"
            else:
                summary += f"\n  {i}. ❌ Failed: {res['error']}"
        
        return summary, None, result['combined_manifest']
    
    except Exception as e:
        return f"❌ Batch processing failed: {str(e)}", None, ""

# Create custom theme with dark mode support
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_label_text_weight="600",
)

# Create Gradio Interface
with gr.Blocks(
    title="Amharic Whisper Fine-Tuning - SOTA Enhanced",
    theme=custom_theme,
    css="""
    .gradio-container {
        max-width: 1400px !important;
    }
    .tab-nav button {
        font-size: 16px;
        font-weight: 500;
    }
    """
) as demo:
    gr.Markdown("""
    # 🎙️ Amharic Whisper Fine-Tuning & Inference
    
    Fine-tune OpenAI's Whisper model for Amharic speech recognition using your LJSpeech dataset.
    Optimized for Lightning AI remote training.
    """)
    
    with gr.Tabs():
        # ===== TRAINING TAB =====
        with gr.Tab("🚀 Training"):
            gr.Markdown("### Dataset Preparation")
            
            with gr.Row():
                with gr.Column():
                    ljspeech_path = gr.Textbox(
                        label="LJSpeech Dataset Path",
                        placeholder="/path/to/ljspeech/",
                        info="Path to LJSpeech directory containing wavs/ and metadata.csv"
                    )
                    output_data_path = gr.Textbox(
                        label="Output Data Path",
                        value=DATA_DIR,
                        info="Where to save processed dataset"
                    )
                    
                with gr.Column():
                    train_split = gr.Slider(
                        minimum=0.5, maximum=0.95, value=0.9, step=0.05,
                        label="Train Split Ratio"
                    )
                    val_split = gr.Slider(
                        minimum=0.02, maximum=0.3, value=0.05, step=0.01,
                        label="Validation Split Ratio"
                    )
            
            prepare_btn = gr.Button("📦 Prepare Dataset", variant="primary")
            prepare_output = gr.Textbox(label="Preparation Status", lines=4)
            
            gr.Markdown("---")
            gr.Markdown("### Training Configuration")
            
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(
                        choices=["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", "openai/whisper-medium"],
                        value="openai/whisper-small",
                        label="Base Model"
                    )
                    data_path = gr.Textbox(
                        label="Training Data Path",
                        value=DATA_DIR,
                        info="Path to prepared dataset"
                    )
                    
                with gr.Column():
                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=1e-5,
                        precision=6
                    )
                    batch_size = gr.Slider(
                        minimum=1, maximum=32, value=16, step=1,
                        label="Batch Size"
                    )
            
            with gr.Row():
                num_epochs = gr.Slider(
                    minimum=1, maximum=100, value=20, step=1,
                    label="Number of Epochs"
                )
                precision = gr.Radio(
                    choices=[16, 32],
                    value=16,
                    label="Training Precision",
                    info="16-bit for faster training, 32-bit for higher accuracy"
                )
            
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    gradient_clip_val = gr.Number(
                        label="Gradient Clipping",
                        value=1.0
                    )
                    accumulate_grad_batches = gr.Slider(
                        minimum=1, maximum=16, value=4, step=1,
                        label="Gradient Accumulation Steps"
                    )
                
                with gr.Row():
                    early_stopping_patience = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="Early Stopping Patience"
                    )
                    save_top_k = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="Save Top K Checkpoints"
                    )
                
                with gr.Row():
                    use_deepspeed = gr.Checkbox(
                        label="Use DeepSpeed (for large models)",
                        value=False
                    )
                    use_lora = gr.Checkbox(
                        label="Use LoRA (Parameter-Efficient Fine-Tuning)",
                        value=False
                    )
                
                with gr.Row():
                    lora_r = gr.Slider(
                        minimum=4, maximum=64, value=16, step=4,
                        label="LoRA Rank (r)"
                    )
                    lora_alpha = gr.Slider(
                        minimum=8, maximum=128, value=32, step=8,
                        label="LoRA Alpha"
                    )
            
            gr.Markdown("### Checkpoint Management")
            
            with gr.Row():
                checkpoint_dir_input = gr.Textbox(
                    label="Checkpoint Directory",
                    value=CHECKPOINT_DIR,
                    info="For Google Colab: Use /content/drive/MyDrive/whisper_checkpoints (after mounting Drive)"
                )
                set_ckpt_dir_btn = gr.Button("📁 Set Checkpoint Dir", scale=0)
            
            ckpt_dir_status = gr.Textbox(label="Status", value=f"Current: {CHECKPOINT_DIR}", interactive=False)
            
            with gr.Row():
                checkpoint_dropdown = gr.Dropdown(
                    choices=format_checkpoint_dropdown(),
                    label="Resume from Checkpoint (optional)",
                    interactive=True
                )
                refresh_ckpt_btn = gr.Button("🔄 Refresh Checkpoints")
            
            with gr.Row():
                delete_ckpt_btn = gr.Button("🗑️ Delete Selected Checkpoint", variant="stop")
                delete_status = gr.Textbox(label="Delete Status", scale=2)
            
            gr.Markdown("---")
            
            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            
            with gr.Row():
                training_output = gr.Textbox(label="Training Status", lines=8, scale=2)
                training_log = gr.Textbox(label="Training Log", lines=8, scale=1)
            
            # Wire up training tab events
            prepare_btn.click(
                fn=prepare_data,
                inputs=[ljspeech_path, output_data_path, train_split, val_split],
                outputs=prepare_output
            )
            
            train_btn.click(
                fn=train_model,
                inputs=[
                    data_path, model_name, learning_rate, batch_size, num_epochs,
                    gradient_clip_val, accumulate_grad_batches, precision,
                    early_stopping_patience, save_top_k, checkpoint_dropdown,
                    use_deepspeed, use_lora, lora_r, lora_alpha
                ],
                outputs=[training_output, checkpoint_dropdown, training_log]
            )
            
            set_ckpt_dir_btn.click(
                fn=set_checkpoint_dir,
                inputs=checkpoint_dir_input,
                outputs=[ckpt_dir_status, checkpoint_dropdown]
            )
            
            refresh_ckpt_btn.click(
                fn=lambda: gr.update(choices=format_checkpoint_dropdown()),
                outputs=checkpoint_dropdown
            )
            
            delete_ckpt_btn.click(
                fn=delete_checkpoint,
                inputs=checkpoint_dropdown,
                outputs=[delete_status, checkpoint_dropdown]
            )
        
        # ===== INFERENCE TAB =====
        with gr.Tab("🎯 Inference"):
            gr.Markdown("### Model Selection")
            
            with gr.Row():
                model_type = gr.Radio(
                    choices=["Fine-tuned Model", "CTranslate2 Model", "Checkpoint"],
                    value="Fine-tuned Model",
                    label="Model Type"
                )
                inference_model_path = gr.Textbox(
                    label="Model Path",
                    value=MODEL_DIR,
                    info="Path to model directory or checkpoint file"
                )
            
            inference_checkpoint_dropdown = gr.Dropdown(
                choices=format_checkpoint_dropdown(),
                label="Or Select Checkpoint",
                visible=False
            )
            
            gr.Markdown("### Transcription Settings")
            
            with gr.Row():
                inference_language = gr.Dropdown(
                    choices=["am", "en", "auto"],
                    value="am",
                    label="Language"
                )
                task = gr.Radio(
                    choices=["transcribe", "translate"],
                    value="transcribe",
                    label="Task"
                )
                beam_size = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Beam Size"
                )
            
            with gr.Row():
                show_waveform = gr.Checkbox(
                    label="Show Waveform",
                    value=True
                )
                show_spectrogram = gr.Checkbox(
                    label="Show Spectrogram",
                    value=False
                )
                use_vad = gr.Checkbox(
                    label="Use Voice Activity Detection",
                    value=True
                )
            
            gr.Markdown("### Audio Input")
            
            with gr.Row():
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                microphone_input = gr.Audio(
                    label="Or Record from Microphone",
                    sources=["microphone"],
                    type="filepath"
                )
            
            transcribe_btn = gr.Button("🎙️ Transcribe", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=2):
                    transcription_output = gr.Textbox(
                        label="Transcription",
                        lines=8,
                        placeholder="Transcribed text will appear here..."
                    )
                    
                    audio_stats = gr.Markdown(
                        label="Audio Statistics",
                        value="Audio stats will appear here..."
                    )
                
                with gr.Column(scale=1):
                    segments_output = gr.JSON(
                        label="Segments (with timestamps)",
                        visible=True
                    )
            
            with gr.Row():
                waveform_plot = gr.Plot(
                    label="Waveform Visualization"
                )
            
            with gr.Row():
                spectrogram_plot = gr.Plot(
                    label="Spectrogram",
                    visible=False
                )
            
            # Wire up inference tab events
            model_type.change(
                fn=lambda t: gr.update(visible=(t == "Checkpoint")),
                inputs=model_type,
                outputs=inference_checkpoint_dropdown
            )
            
            show_spectrogram.change(
                fn=lambda x: gr.update(visible=x),
                inputs=show_spectrogram,
                outputs=spectrogram_plot
            )
            
            transcribe_btn.click(
                fn=lambda a, m, mp, lang, t, bs, sw, ss, vad: transcribe(
                    a if a else m, mp, lang, t, bs, sw, ss, vad
                ),
                inputs=[
                    audio_input, microphone_input, inference_model_path,
                    inference_language, task, beam_size,
                    show_waveform, show_spectrogram, use_vad
                ],
                outputs=[transcription_output, waveform_plot, spectrogram_plot, segments_output, audio_stats]
            )
    
        # ===== STREAMING TAB =====
        if ENHANCED_FEATURES_AVAILABLE:
            with gr.Tab("🎙️ Real-time Streaming"):
                gr.Markdown("""
                ### Real-time Streaming Transcription
                Speak into your microphone for live transcription with Voice Activity Detection.
                """)
                
                streaming_state = gr.State({})
                
                with gr.Row():
                    streaming_audio = gr.Audio(
                        label="Microphone (Streaming)",
                        sources=["microphone"],
                        streaming=True,
                        type="numpy"
                    )
                
                with gr.Row():
                    streaming_transcript = gr.Textbox(
                        label="Live Transcript",
                        lines=12,
                        interactive=False,
                        placeholder="Start speaking to see real-time transcription..."
                    )
                
                with gr.Row():
                    vad_status = gr.Textbox(
                        label="VAD Status",
                        value="Waiting for speech...",
                        interactive=False
                    )
                    confidence_display = gr.Number(
                        label="Speech Confidence",
                        value=0.0,
                        interactive=False
                    )
                
                with gr.Row():
                    clear_streaming = gr.Button("🗑️ Clear Transcript")
                    pause_streaming = gr.Button("⏸️ Pause")
                
                # Placeholder for streaming function - would need actual implementation
                gr.Markdown("""
                **Note**: Full streaming implementation requires WebSocket backend.
                This is a placeholder for the streaming interface.
                """)
                
                clear_streaming.click(
                    fn=lambda: ("", {}, "Cleared", 0.0),
                    outputs=[streaming_transcript, streaming_state, vad_status, confidence_display]
                )
        
        # ===== CHAT INTERFACE TAB =====
        if ENHANCED_FEATURES_AVAILABLE:
            with gr.Tab("💬 Chat Interface"):
                gr.Markdown("""
                ### Multimodal Chat Interface
                Interact using voice or text. Audio messages will be automatically transcribed.
                """)
                
                chat_history = gr.Chatbot(
                    label="Conversation",
                    height=450
                )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        chat_text_input = gr.Textbox(
                            label="Type a message",
                            placeholder="Type here or use microphone...",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        chat_audio_input = gr.Audio(
                            label="🎤 Record Message",
                            sources=["microphone"],
                            type="filepath"
                        )
                
                with gr.Row():
                    chat_clear_btn = gr.Button("🗑️ Clear Chat")
                    chat_send_btn = gr.Button("📤 Send", variant="primary")
                
                def handle_chat_text(text, history):
                    if not text.strip():
                        return history, ""
                    response = f"Echo: {text}"
                    history.append((text, response))
                    return history, ""
                
                def handle_chat_audio(audio, history):
                    if audio is None:
                        return history, None
                    try:
                        # Transcribe audio
                        result = transcribe_audio(
                            audio_path=audio,
                            model_path=MODEL_DIR,
                            language="am",
                            task="transcribe",
                            beam_size=5
                        )
                        transcript = result['text']
                        response = f"Transcribed: {transcript}"
                        history.append((f"🎤 {transcript}", response))
                        return history, None
                    except Exception as e:
                        history.append(("🎤 [Audio]", f"Error: {str(e)}"))
                        return history, None
                
                chat_text_input.submit(
                    fn=handle_chat_text,
                    inputs=[chat_text_input, chat_history],
                    outputs=[chat_history, chat_text_input]
                )
                
                chat_send_btn.click(
                    fn=handle_chat_text,
                    inputs=[chat_text_input, chat_history],
                    outputs=[chat_history, chat_text_input]
                )
                
                chat_audio_input.change(
                    fn=handle_chat_audio,
                    inputs=[chat_audio_input, chat_history],
                    outputs=[chat_history, chat_audio_input]
                )
                
                chat_clear_btn.click(
                    fn=lambda: ([], "", None),
                    outputs=[chat_history, chat_text_input, chat_audio_input]
                )
        
        # ===== YOUTUBE DATASET PREPARATION TAB =====
        with gr.Tab("🎬 YouTube Dataset"):
            gr.Markdown("""
            ### YouTube Dataset Preparation
            
            Create high-quality Amharic Whisper datasets from YouTube videos with subtitles.
            
            **Features**:
            - ✅ Automatic Amharic subtitle detection
            - ✅ Background music removal using Demucs
            - ✅ Precise audio segmentation
            - ✅ SOTA-level dataset quality
            - ✅ Batch processing support
            """)
            
            with gr.Tabs():
                # Single Video Processing
                with gr.Tab("Single Video"):
                    with gr.Row():
                        with gr.Column():
                            youtube_url = gr.Textbox(
                                label="YouTube URL",
                                placeholder="https://www.youtube.com/watch?v=...",
                                info="Enter a YouTube video URL with Amharic subtitles"
                            )
                            
                            check_btn = gr.Button("🔍 Check Video", variant="secondary")
                            check_output = gr.Textbox(
                                label="Video Information",
                                lines=10,
                                interactive=False
                            )
                    
                    gr.Markdown("### Processing Settings")
                    
                    with gr.Row():
                        with gr.Column():
                            youtube_output_dir = gr.Textbox(
                                label="Output Directory",
                                value="./data/youtube_amharic",
                                info="Where to save the processed dataset"
                            )
                            
                            use_demucs_checkbox = gr.Checkbox(
                                label="Remove Background Music (Demucs)",
                                value=True,
                                info="Extract vocals only - requires Demucs installed"
                            )
                            
                            cookies_browser_dropdown = gr.Dropdown(
                                choices=["none", "chrome", "firefox", "edge", "safari", "brave"],
                                value="none",
                                label="Extract Cookies From Browser (Optional)",
                                info="Leave as 'none' - cookies only needed if default bypass fails"
                            )
                        
                        with gr.Column():
                            min_seg_duration = gr.Slider(
                                minimum=0.5,
                                maximum=5.0,
                                value=1.0,
                                step=0.1,
                                label="Minimum Segment Duration (seconds)"
                            )
                            
                            max_seg_duration = gr.Slider(
                                minimum=10.0,
                                maximum=30.0,
                                value=25.0,
                                step=1.0,
                                label="Maximum Segment Duration (seconds)"
                            )
                    
                    process_single_btn = gr.Button(
                        "🚀 Process Video & Create Dataset",
                        variant="primary",
                        size="lg"
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            process_output = gr.Textbox(
                                label="Processing Results",
                                lines=12
                            )
                        
                        with gr.Column():
                            preview_dataframe = gr.Dataframe(
                                label="Dataset Preview (First 10 segments)",
                                headers=["Segment", "Duration", "Text", "Audio File"],
                                interactive=False
                            )
                    
                    manifest_path_output = gr.Textbox(
                        label="Manifest Path",
                        interactive=False,
                        visible=False
                    )
                    
                    # Wire up single video processing
                    check_btn.click(
                        fn=check_youtube_video,
                        inputs=youtube_url,
                        outputs=check_output
                    )
                    
                    process_single_btn.click(
                        fn=process_youtube_video,
                        inputs=[
                            youtube_url,
                            youtube_output_dir,
                            use_demucs_checkbox,
                            min_seg_duration,
                            max_seg_duration,
                            cookies_browser_dropdown
                        ],
                        outputs=[process_output, preview_dataframe, manifest_path_output]
                    )
                
                # Local File Upload
                with gr.Tab("📁 Local Files"):
                    gr.Markdown("""
                    ### Upload Local Audio + SRT Files
                    
                    Upload your own audio files with matching SRT subtitles.
                    **Perfect for when YouTube download doesn't work!**
                    
                    **Supported audio formats**: MP3, WAV, M4A, OGG, FLAC
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            local_audio_file = gr.File(
                                label="Audio File",
                                file_types=["audio"],
                                type="filepath"
                            )
                        
                        with gr.Column():
                            local_srt_file = gr.File(
                                label="SRT Subtitle File",
                                file_types=[".srt"],
                                type="filepath"
                            )
                    
                    gr.Markdown("### Processing Settings")
                    
                    with gr.Row():
                        with gr.Column():
                            local_output_dir = gr.Textbox(
                                label="Output Directory",
                                value="./data/local_amharic",
                                info="Where to save the processed dataset"
                            )
                            
                            local_use_demucs = gr.Checkbox(
                                label="Remove Background Music (Demucs)",
                                value=False,
                                info="Extract vocals only - usually not needed for clean audio"
                            )
                        
                        with gr.Column():
                            local_min_duration = gr.Slider(
                                minimum=0.5,
                                maximum=5.0,
                                value=1.0,
                                step=0.1,
                                label="Minimum Segment Duration (seconds)"
                            )
                            
                            local_max_duration = gr.Slider(
                                minimum=10.0,
                                maximum=30.0,
                                value=25.0,
                                step=1.0,
                                label="Maximum Segment Duration (seconds)"
                            )
                    
                    process_local_btn = gr.Button(
                        "🚀 Process Files & Create Dataset",
                        variant="primary",
                        size="lg"
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            local_process_output = gr.Textbox(
                                label="Processing Results",
                                lines=12
                            )
                        
                        with gr.Column():
                            local_preview_dataframe = gr.Dataframe(
                                label="Dataset Preview (First 10 segments)",
                                headers=["Segment", "Duration", "Text", "Audio File"],
                                interactive=False
                            )
                    
                    local_manifest_output = gr.Textbox(
                        label="Manifest Path",
                        interactive=False,
                        visible=False
                    )
                    
                    # Wire up local file processing
                    process_local_btn.click(
                        fn=process_local_file_pair,
                        inputs=[
                            local_audio_file,
                            local_srt_file,
                            local_output_dir,
                            local_use_demucs,
                            local_min_duration,
                            local_max_duration
                        ],
                        outputs=[local_process_output, local_preview_dataframe, local_manifest_output]
                    )
                
                # Batch Processing
                with gr.Tab("Batch Processing"):
                    gr.Markdown("""
                    ### Process Multiple Videos
                    
                    Enter multiple YouTube URLs (one per line) to create a large dataset.
                    """)
                    
                    with gr.Row():
                        batch_urls = gr.Textbox(
                            label="YouTube URLs (one per line)",
                            lines=10,
                            placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
                            info="Enter YouTube video URLs, one per line"
                        )
                    
                    with gr.Row():
                        with gr.Column():
                            batch_output_dir = gr.Textbox(
                                label="Output Directory",
                                value="./data/youtube_amharic",
                                info="Where to save the combined dataset"
                            )
                            
                            batch_use_demucs = gr.Checkbox(
                                label="Remove Background Music (Demucs)",
                                value=True
                            )
                            
                            batch_cookies_browser = gr.Dropdown(
                                choices=["none", "chrome", "firefox", "edge", "safari", "brave"],
                                value="none",
                                label="Extract Cookies From Browser (Optional)",
                                info="Leave as 'none' - cookies only needed if default bypass fails"
                            )
                        
                        with gr.Column():
                            batch_min_duration = gr.Slider(
                                minimum=0.5,
                                maximum=5.0,
                                value=1.0,
                                step=0.1,
                                label="Minimum Segment Duration (seconds)"
                            )
                            
                            batch_max_duration = gr.Slider(
                                minimum=10.0,
                                maximum=30.0,
                                value=25.0,
                                step=1.0,
                                label="Maximum Segment Duration (seconds)"
                            )
                    
                    process_batch_btn = gr.Button(
                        "🚀 Process All Videos",
                        variant="primary",
                        size="lg"
                    )
                    
                    batch_output = gr.Textbox(
                        label="Batch Processing Results",
                        lines=15
                    )
                    
                    batch_manifest_output = gr.Textbox(
                        label="Combined Manifest Path",
                        interactive=False,
                        visible=False
                    )
                    
                    # Wire up batch processing
                    process_batch_btn.click(
                        fn=process_multiple_youtube_videos,
                        inputs=[
                            batch_urls,
                            batch_output_dir,
                            batch_use_demucs,
                            batch_min_duration,
                            batch_max_duration,
                            batch_cookies_browser
                        ],
                        outputs=[batch_output, preview_dataframe, batch_manifest_output]
                    )
            
            gr.Markdown("""
            ---
            ### 💡 Tips
            
            1. **Finding Videos**: Look for Amharic YouTube videos with closed captions (CC)
            2. **Quality**: Manual subtitles are preferred over auto-generated
            3. **Demucs**: Improves quality but requires installation and is slower
            4. **Duration**: Whisper works best with 1-25 second segments
            5. **Batch Processing**: Process multiple videos to create larger datasets
            
            ### 📦 Requirements
            
            ```bash
            pip install yt-dlp demucs
            ```
            """)
        
        # ===== METRICS DASHBOARD TAB =====
        if ENHANCED_FEATURES_AVAILABLE:
            with gr.Tab("📊 Metrics Dashboard"):
                gr.Markdown("""
                ### Training Metrics Dashboard
                Monitor training progress with live metrics visualization.
                """)
                
                with gr.Row():
                    refresh_metrics_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
                
                metrics_plot = gr.Plot(
                    label="Training Metrics",
                    value=create_training_metrics_plot(training_metrics)
                )
                
                with gr.Row():
                    with gr.Column():
                        current_loss = gr.Number(
                            label="Current Loss",
                            value=0.0,
                            interactive=False
                        )
                        current_wer = gr.Number(
                            label="Current WER (%)",
                            value=0.0,
                            interactive=False
                        )
                    with gr.Column():
                        best_loss = gr.Number(
                            label="Best Loss",
                            value=0.0,
                            interactive=False
                        )
                        best_wer = gr.Number(
                            label="Best WER (%)",
                            value=0.0,
                            interactive=False
                        )
                
                realtime_loss_plot = gr.Plot(
                    label="Real-time Loss (Last 100 steps)"
                )
                
                def refresh_metrics():
                    fig = create_training_metrics_plot(training_metrics)
                    
                    curr_loss = training_metrics['loss'][-1] if training_metrics['loss'] else 0.0
                    curr_wer = training_metrics['wer'][-1] if training_metrics['wer'] else 0.0
                    b_loss = min(training_metrics['loss']) if training_metrics['loss'] else 0.0
                    b_wer = min(training_metrics['wer']) if training_metrics['wer'] else 0.0
                    
                    loss_fig = create_realtime_loss_plot(training_metrics['loss'])
                    
                    return fig, curr_loss, curr_wer, b_loss, b_wer, loss_fig
                
                refresh_metrics_btn.click(
                    fn=refresh_metrics,
                    outputs=[metrics_plot, current_loss, current_wer, best_loss, best_wer, realtime_loss_plot]
                )
    
    gr.Markdown("""
    ---
    ### 📝 Notes
    - **Training**: Start by preparing your LJSpeech dataset, then configure and start training
    - **YouTube Dataset**: Create datasets from YouTube videos with Amharic subtitles
    - **Checkpoints**: Best checkpoints are automatically saved during training
    - **Resume**: Select a checkpoint to resume training from where you left off
    - **Inference**: Use trained models or checkpoints to transcribe audio with enhanced visualizations
    - **Streaming**: Real-time transcription with Voice Activity Detection (VAD)
    - **Chat**: Multimodal interface supporting both text and voice input
    - **Metrics**: Live training metrics dashboard for monitoring progress
    - **Lightning AI**: This interface is optimized for remote training on Lightning AI Studio
    """)

if __name__ == "__main__":
    import sys
    
    # Check if running in Colab or if --share flag is passed
    is_colab = 'google.colab' in sys.modules
    share = is_colab or '--share' in sys.argv
    
    demo.queue(max_size=50, default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        show_error=True
    )
