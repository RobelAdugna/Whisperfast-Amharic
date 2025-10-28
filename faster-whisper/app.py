import gradio as gr
import os
import json
import torch
from pathlib import Path
import lightning as L
from datetime import datetime
from typing import Dict, List, Optional
import glob

# Import training and inference modules
from train_whisper_lightning import WhisperLightningModule, AmharicDataModule, start_training
from inference_utils import load_model_for_inference, transcribe_audio
from prepare_ljspeech_dataset import prepare_dataset

# Global state for managing checkpoints and models
CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "./data"
MODEL_DIR = "./whisper_finetuned"
CT2_MODEL_DIR = "./whisper_ct2_model"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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
    progress=gr.Progress()
):
    """Transcribe audio using trained model"""
    try:
        progress(0, desc="Loading model...")
        
        if not audio_file:
            return "Please upload an audio file"
        
        progress(0.3, desc="Transcribing...")
        
        result = transcribe_audio(
            audio_path=audio_file,
            model_path=model_path,
            language=language,
            task=task,
            beam_size=beam_size
        )
        
        progress(1.0, desc="Transcription complete!")
        
        return result['text']
        
    except Exception as e:
        return f"❌ Transcription failed: {str(e)}"

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

# Create Gradio Interface
with gr.Blocks(title="Amharic Whisper Fine-Tuning", theme=gr.themes.Soft()) as demo:
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
            transcription_output = gr.Textbox(
                label="Transcription",
                lines=10,
                placeholder="Transcribed text will appear here..."
            )
            
            # Wire up inference tab events
            model_type.change(
                fn=lambda t: gr.update(visible=(t == "Checkpoint")),
                inputs=model_type,
                outputs=inference_checkpoint_dropdown
            )
            
            def get_audio_source(audio, mic):
                return audio if audio else mic
            
            transcribe_btn.click(
                fn=lambda a, m, mp, lang, t, bs: transcribe(
                    a if a else m, mp, lang, t, bs
                ),
                inputs=[
                    audio_input, microphone_input, inference_model_path,
                    inference_language, task, beam_size
                ],
                outputs=transcription_output
            )
    
    gr.Markdown("""
    ---
    ### 📝 Notes
    - **Training**: Start by preparing your LJSpeech dataset, then configure and start training
    - **Checkpoints**: Best checkpoints are automatically saved during training
    - **Resume**: Select a checkpoint to resume training from where you left off
    - **Inference**: Use trained models or checkpoints to transcribe audio
    - **Lightning AI**: This interface is optimized for remote training on Lightning AI Studio
    """)

if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )