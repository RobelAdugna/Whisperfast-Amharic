#!/usr/bin/env python3
"""
Convert LJSpeech format Amharic TTS dataset to Whisper training format.
"""

import argparse
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil


def load_ljspeech_metadata(metadata_path: str) -> List[Tuple[str, str]]:
    """
    Load LJSpeech metadata.csv file.
    Format: filename|transcript or filename|transcript|normalized_transcript
    """
    data = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) >= 2:
                filename = row[0]
                transcript = row[1]  # Use raw transcript
                data.append((filename, transcript))
    return data


def validate_audio(audio_path: str, sample_rate: int = 16000, max_duration: float = 30.0) -> Tuple[bool, str]:
    """
    Validate audio file meets Whisper requirements.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        if duration > max_duration:
            return False, f"Audio too long: {duration:.2f}s > {max_duration}s"
        
        if duration < 0.1:
            return False, f"Audio too short: {duration:.2f}s"
        
        if sr != sample_rate:
            # Resample if needed
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sf.write(audio_path, audio, sample_rate)
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def normalize_amharic_text(text: str, remove_punctuation: bool = False) -> str:
    """
    Normalize Amharic text for Whisper training.
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Optional: remove punctuation (configure based on your needs)
    if remove_punctuation:
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text.strip()


def split_dataset(data: List[Tuple[str, str]], train_ratio: float = 0.9, val_ratio: float = 0.05) -> Dict[str, List]:
    """
    Split dataset into train/val/test sets.
    """
    import random
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }


def create_whisper_dataset(input_dir: str, output_dir: str, sample_rate: int = 16000):
    """
    Main function to convert LJSpeech to Whisper format.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Locate audio directory and metadata
    wavs_dir = input_path / 'wavs'
    metadata_path = input_path / 'metadata.csv'
    
    if not wavs_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {wavs_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = load_ljspeech_metadata(str(metadata_path))
    print(f"Found {len(metadata)} entries")
    
    # Split dataset
    print("Splitting dataset...")
    splits = split_dataset(metadata)
    
    # Process each split
    for split_name, split_data in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_data)} samples)...")
        
        split_output_dir = output_path / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_output_dir = split_output_dir / 'audio'
        audio_output_dir.mkdir(exist_ok=True)
        
        valid_samples = []
        
        for filename, transcript in tqdm(split_data, desc=f"Converting {split_name}"):
            # Construct audio path
            if filename.endswith('.wav'):
                audio_name = filename
            else:
                audio_name = f"{filename}.wav"
            
            audio_path = wavs_dir / audio_name
            
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}")
                continue
            
            # Validate and process audio
            output_audio_path = audio_output_dir / audio_name
            shutil.copy2(audio_path, output_audio_path)
            
            is_valid, message = validate_audio(str(output_audio_path), sample_rate)
            
            if not is_valid:
                print(f"Warning: Skipping {audio_name}: {message}")
                output_audio_path.unlink()
                continue
            
            # Normalize transcript
            normalized_transcript = normalize_amharic_text(transcript)
            
            valid_samples.append({
                'audio': str(output_audio_path.relative_to(split_output_dir)),
                'text': normalized_transcript,
                'duration': librosa.get_duration(path=str(output_audio_path))
            })
        
        # Save metadata as JSON
        metadata_output_path = split_output_dir / 'metadata.json'
        with open(metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(valid_samples, f, ensure_ascii=False, indent=2)
        
        print(f"{split_name}: {len(valid_samples)}/{len(split_data)} valid samples")
        total_duration = sum(s['duration'] for s in valid_samples)
        print(f"Total duration: {total_duration/3600:.2f} hours")
    
    print(f"\nDataset conversion complete! Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LJSpeech Amharic dataset to Whisper format")
    parser.add_argument("--input_dir", required=True, help="Path to LJSpeech format dataset")
    parser.add_argument("--output_dir", default="./data/amharic_whisper", help="Output directory")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    
    args = parser.parse_args()
    
    create_whisper_dataset(args.input_dir, args.output_dir, args.sample_rate)
