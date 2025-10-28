import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from faster_whisper import WhisperModel
import torchaudio
import os
from typing import Dict, Optional

def load_model_for_inference(model_path: str, model_type: str = "huggingface"):
    """Load model for inference
    
    Args:
        model_path: Path to model directory or checkpoint
        model_type: Type of model ('huggingface', 'ctranslate2', 'checkpoint')
    """
    if model_type == "ctranslate2":
        # Load CTranslate2 model
        model = WhisperModel(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        return model, None
    
    elif model_type == "checkpoint":
        # Load from Lightning checkpoint
        from train_whisper_lightning import WhisperLightningModule
        lightning_model = WhisperLightningModule.load_from_checkpoint(model_path)
        model = lightning_model.model
        processor = lightning_model.processor
        return model, processor
    
    else:
        # Load Hugging Face model
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        return model, processor

def transcribe_audio(
    audio_path: str,
    model_path: str,
    language: str = "am",
    task: str = "transcribe",
    beam_size: int = 5,
    model_type: str = "auto"
) -> Dict:
    """Transcribe audio file
    
    Args:
        audio_path: Path to audio file
        model_path: Path to model
        language: Language code (e.g., 'am' for Amharic)
        task: 'transcribe' or 'translate'
        beam_size: Beam search size
        model_type: Type of model ('auto', 'huggingface', 'ctranslate2', 'checkpoint')
    
    Returns:
        Dict with transcription results
    """
    
    # Load audio
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Validate model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Detect model type from path if auto
    if model_type == "auto":
        if os.path.isfile(model_path) and model_path.endswith(".ckpt"):
            model_type = "checkpoint"
        elif "ct2" in model_path or "ctranslate" in model_path.lower():
            model_type = "ctranslate2"
        else:
            model_type = "huggingface"
    
    # Load model
    model, processor = load_model_for_inference(model_path, model_type)
    
    if model_type == "ctranslate2":
        # Use faster-whisper for CTranslate2 models
        segments, info = model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            task=task,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        transcription = " ".join([segment.text for segment in segments])
        
        return {
            "text": transcription,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        }
    
    else:
        # Use Hugging Face model
        # Load and resample audio to 16kHz
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process audio
        input_features = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
        
        # Set language tokens
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language if language != "auto" else None,
            task=task
        )
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=beam_size,
                max_length=448
            )
        
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return {
            "text": transcription,
            "language": language,
            "duration": waveform.shape[1] / 16000
        }

def convert_to_ctranslate2(
    model_path: str,
    output_path: str,
    quantization: str = "int8"
):
    """Convert Hugging Face model to CTranslate2 format
    
    Args:
        model_path: Path to Hugging Face model
        output_path: Path to save CTranslate2 model
        quantization: Quantization type ('int8', 'float16', 'float32')
    """
    try:
        import ctranslate2
        
        converter = ctranslate2.converters.TransformersConverter(model_path)
        converter.convert(
            output_path,
            quantization=quantization,
            force=True
        )
        
        print(f"Model converted successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return False