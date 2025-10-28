import pandas as pd
import os
from pathlib import Path
from datasets import Dataset, Audio, DatasetDict
import torchaudio
from tqdm import tqdm
from typing import Dict, Optional, Callable
import unicodedata

def normalize_amharic_text(text: str) -> str:
    """Normalize Amharic text
    
    Args:
        text: Input Amharic text
    
    Returns:
        Normalized text
    """
    # Unicode NFC normalization for Ge'ez script
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase (if applicable)
    # Note: Amharic doesn't have case, but keeping for consistency
    
    return text.strip()

def load_ljspeech_metadata(ljspeech_path: str) -> pd.DataFrame:
    """Load and parse LJSpeech metadata
    
    Args:
        ljspeech_path: Path to LJSpeech directory
    
    Returns:
        DataFrame with id, transcript, and audio_path
    """
    # Validate paths
    if not os.path.exists(ljspeech_path):
        raise FileNotFoundError(f"LJSpeech directory not found: {ljspeech_path}")
    
    metadata_path = os.path.join(ljspeech_path, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {ljspeech_path}")
    
    wav_dir = Path(ljspeech_path) / "wavs"
    if not wav_dir.exists():
        raise FileNotFoundError(f"wavs directory not found in {ljspeech_path}")
    
    # Read metadata (pipe-separated: ID|transcript)
    metadata = pd.read_csv(
        metadata_path,
        sep="|",
        header=None,
        names=["id", "transcript"],
        quoting=3  # Ignore quotes
    )
    
    # Clean ID (remove .wav if present)
    metadata["id"] = metadata["id"].str.replace(".wav", "", regex=False)
    
    # Add audio paths (wav_dir already validated above)
    metadata["audio_path"] = metadata["id"].apply(
        lambda x: str(wav_dir / f"{x}.wav")
    )
    
    # Filter out non-existent files
    metadata = metadata[metadata["audio_path"].apply(os.path.exists)]
    
    # Normalize text
    metadata["transcript"] = metadata["transcript"].apply(normalize_amharic_text)
    
    return metadata

def resample_audio(audio_path: str, target_sr: int = 16000):
    """Load and resample audio to target sampling rate
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate
    
    Returns:
        Resampled audio array and sampling rate
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform.squeeze().numpy(), target_sr

def create_hf_dataset(
    metadata: pd.DataFrame,
    progress_callback: Optional[Callable] = None
) -> Dataset:
    """Create Hugging Face Dataset from metadata
    
    Args:
        metadata: DataFrame with audio paths and transcripts
        progress_callback: Optional callback for progress updates
    
    Returns:
        Hugging Face Dataset
    """
    # Create dataset from pandas
    dataset = Dataset.from_pandas(
        metadata[["audio_path", "transcript"]].rename(
            columns={"audio_path": "audio", "transcript": "text"}
        )
    )
    
    # Cast audio column (this will handle resampling)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return dataset

def prepare_dataset(
    ljspeech_path: str,
    output_path: str,
    train_split: float = 0.9,
    val_split: float = 0.05,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """Prepare LJSpeech dataset for Whisper training
    
    Args:
        ljspeech_path: Path to LJSpeech directory
        output_path: Where to save processed dataset
        train_split: Proportion for training set
        val_split: Proportion for validation set
        progress_callback: Optional Gradio progress callback
    
    Returns:
        Dict with dataset statistics
    """
    if progress_callback:
        progress_callback(0.1, desc="Loading metadata...")
    
    # Load metadata
    print(f"Loading metadata from {ljspeech_path}...")
    metadata = load_ljspeech_metadata(ljspeech_path)
    print(f"Found {len(metadata)} audio files")
    
    if progress_callback:
        progress_callback(0.3, desc="Creating dataset...")
    
    # Create dataset
    dataset = create_hf_dataset(metadata, progress_callback)
    
    if progress_callback:
        progress_callback(0.6, desc="Splitting dataset...")
    
    # Split dataset
    test_split = 1.0 - train_split - val_split
    
    # First split: train vs (val + test)
    train_testval = dataset.train_test_split(
        test_size=(val_split + test_split),
        seed=42
    )
    
    # Second split: val vs test
    val_test = train_testval["test"].train_test_split(
        test_size=test_split / (val_split + test_split),
        seed=42
    )
    
    # Create final dataset dict
    dataset_dict = DatasetDict({
        "train": train_testval["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    
    if progress_callback:
        progress_callback(0.9, desc="Saving dataset...")
    
    # Save to disk
    os.makedirs(output_path, exist_ok=True)
    dataset_dict.save_to_disk(output_path)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Train: {len(dataset_dict['train'])} samples")
    print(f"Validation: {len(dataset_dict['validation'])} samples")
    print(f"Test: {len(dataset_dict['test'])} samples")
    
    if progress_callback:
        progress_callback(1.0, desc="Dataset preparation complete!")
    
    return {
        "train_size": len(dataset_dict["train"]),
        "val_size": len(dataset_dict["validation"]),
        "test_size": len(dataset_dict["test"]),
        "total_size": len(dataset)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare LJSpeech dataset for Whisper")
    parser.add_argument("--ljspeech_path", type=str, required=True, help="Path to LJSpeech directory")
    parser.add_argument("--output_path", type=str, default="./data", help="Output path")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split ratio")
    
    args = parser.parse_args()
    
    prepare_dataset(
        ljspeech_path=args.ljspeech_path,
        output_path=args.output_path,
        train_split=args.train_split,
        val_split=args.val_split
    )