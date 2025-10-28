"""Amharic Dataset Processor optimized for large-scale fine-tuning"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

try:
    from utils.amharic_tokenizer import AmharicTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

class AmharicDatasetProcessor:
    """Process and validate Amharic ASR dataset (optimized for 150h+)"""
    
    def __init__(self, 
                 data_dir: str,
                 sample_rate: int = 16000,
                 min_duration: float = 0.5,
                 max_duration: float = 30.0,
                 quality_threshold: float = 0.8):
        """
        Initialize dataset processor
        
        Args:
            data_dir: Root directory containing audio files and transcripts
            sample_rate: Target sample rate (Whisper uses 16kHz)
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds  
            quality_threshold: Minimum Amharic text quality score (0-1)
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.quality_threshold = quality_threshold
        
        if TOKENIZER_AVAILABLE:
            self.tokenizer = AmharicTokenizer(
                normalize_punctuation=True,
                normalize_numerals=True,
                preserve_diacritics=True
            )
        else:
            self.tokenizer = None
    
    def analyze_dataset(self, manifest_path: str) -> Dict:
        """
        Comprehensive dataset analysis for 150-hour corpus
        
        Args:
            manifest_path: Path to dataset manifest (JSON/CSV)
        
        Returns:
            Dictionary with detailed statistics
        """
        print("🔍 Analyzing Amharic dataset...")
        
        # Load manifest
        if manifest_path.endswith('.json'):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif manifest_path.endswith('.csv'):
            data = pd.read_csv(manifest_path).to_dict('records')
        else:
            raise ValueError("Manifest must be JSON or CSV")
        
        stats = {
            'total_samples': len(data),
            'total_duration_hours': 0,
            'duration_distribution': {'0-5s': 0, '5-10s': 0, '10-20s': 0, '20-30s': 0, '>30s': 0},
            'avg_duration': 0,
            'avg_text_length': 0,
            'total_characters': 0,
            'total_words': 0,
            'code_switching_count': 0,
            'quality_issues': 0,
            'dialect_distribution': {},
            'speaker_stats': {},
        }
        
        durations = []
        text_lengths = []
        
        for item in tqdm(data, desc="Analyzing samples"):
            # Get audio duration
            audio_path = self.data_dir / item.get('audio_path', item.get('path', ''))
            if audio_path.exists():
                try:
                    info = sf.info(str(audio_path))
                    duration = info.duration
                    durations.append(duration)
                    
                    # Duration distribution
                    if duration < 5:
                        stats['duration_distribution']['0-5s'] += 1
                    elif duration < 10:
                        stats['duration_distribution']['5-10s'] += 1
                    elif duration < 20:
                        stats['duration_distribution']['10-20s'] += 1
                    elif duration < 30:
                        stats['duration_distribution']['20-30s'] += 1
                    else:
                        stats['duration_distribution']['>30s'] += 1
                        
                except Exception as e:
                    print(f"Error reading {audio_path}: {e}")
                    continue
            
            # Analyze text
            text = item.get('text', item.get('transcript', ''))
            if text and self.tokenizer:
                # Normalize text
                normalized_text = self.tokenizer.prepare_for_whisper(text)
                text_lengths.append(len(normalized_text))
                stats['total_characters'] += len(normalized_text)
                
                # Word count
                words = self.tokenizer.tokenize_words(normalized_text)
                stats['total_words'] += len(words)
                
                # Check code-switching
                has_cs, _ = self.tokenizer.detect_code_switching(normalized_text)
                if has_cs:
                    stats['code_switching_count'] += 1
                
                # Quality check
                quality = self.tokenizer.validate_text_quality(normalized_text)
                if quality['quality_score'] < self.quality_threshold:
                    stats['quality_issues'] += 1
                
                # Dialect detection
                dialects = self.tokenizer.detect_dialect_markers(normalized_text)
                for dialect, present in dialects.items():
                    if present:
                        stats['dialect_distribution'][dialect] = stats['dialect_distribution'].get(dialect, 0) + 1
            
            # Speaker statistics
            speaker = item.get('speaker_id', 'unknown')
            stats['speaker_stats'][speaker] = stats['speaker_stats'].get(speaker, 0) + 1
        
        # Calculate averages
        stats['total_duration_hours'] = sum(durations) / 3600
        stats['avg_duration'] = np.mean(durations) if durations else 0
        stats['avg_text_length'] = np.mean(text_lengths) if text_lengths else 0
        stats['median_duration'] = np.median(durations) if durations else 0
        stats['std_duration'] = np.std(durations) if durations else 0
        
        # Quality metrics
        stats['code_switching_rate'] = stats['code_switching_count'] / stats['total_samples']
        stats['quality_issue_rate'] = stats['quality_issues'] / stats['total_samples']
        stats['num_speakers'] = len(stats['speaker_stats'])
        
        return stats
    
    def filter_dataset(self, 
                      manifest_path: str,
                      output_path: str,
                      remove_code_switching: bool = False,
                      remove_quality_issues: bool = True) -> Dict:
        """
        Filter dataset based on quality criteria
        
        Args:
            manifest_path: Input manifest path
            output_path: Output filtered manifest path
            remove_code_switching: Remove samples with code-switching
            remove_quality_issues: Remove low-quality samples
        
        Returns:
            Filtering statistics
        """
        print("🔧 Filtering Amharic dataset...")
        
        # Load manifest
        if manifest_path.endswith('.json'):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = pd.read_csv(manifest_path).to_dict('records')
        
        filtered_data = []
        filter_stats = {
            'original_count': len(data),
            'filtered_count': 0,
            'removed_duration': 0,
            'removed_code_switching': 0,
            'removed_quality': 0,
            'removed_other': 0
        }
        
        for item in tqdm(data, desc="Filtering samples"):
            keep = True
            
            # Check audio duration
            audio_path = self.data_dir / item.get('audio_path', item.get('path', ''))
            if audio_path.exists():
                try:
                    info = sf.info(str(audio_path))
                    duration = info.duration
                    
                    if duration < self.min_duration or duration > self.max_duration:
                        keep = False
                        filter_stats['removed_duration'] += 1
                        continue
                except:
                    keep = False
                    filter_stats['removed_other'] += 1
                    continue
            else:
                keep = False
                filter_stats['removed_other'] += 1
                continue
            
            # Check text quality
            text = item.get('text', item.get('transcript', ''))
            if text and self.tokenizer:
                # Code-switching check
                if remove_code_switching:
                    has_cs, _ = self.tokenizer.detect_code_switching(text)
                    if has_cs:
                        keep = False
                        filter_stats['removed_code_switching'] += 1
                        continue
                
                # Quality check
                if remove_quality_issues:
                    quality = self.tokenizer.validate_text_quality(text)
                    if quality['quality_score'] < self.quality_threshold:
                        keep = False
                        filter_stats['removed_quality'] += 1
                        continue
                
                # Normalize text in the item
                item['text'] = self.tokenizer.prepare_for_whisper(text)
            
            if keep:
                filtered_data.append(item)
        
        filter_stats['filtered_count'] = len(filtered_data)
        filter_stats['retention_rate'] = filter_stats['filtered_count'] / filter_stats['original_count']
        
        # Save filtered manifest
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        else:
            pd.DataFrame(filtered_data).to_csv(output_path, index=False)
        
        print(f"✅ Filtered dataset: {filter_stats['filtered_count']}/{filter_stats['original_count']} samples retained")
        
        return filter_stats
    
    def create_balanced_splits(self,
                              manifest_path: str,
                              output_dir: str,
                              train_ratio: float = 0.9,
                              val_ratio: float = 0.05,
                              test_ratio: float = 0.05,
                              stratify_by: str = 'speaker_id') -> Dict:
        """
        Create balanced train/val/test splits for 150h dataset
        
        Args:
            manifest_path: Input manifest
            output_dir: Output directory for splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify_by: Column to stratify by ('speaker_id' or 'duration_bin')
        
        Returns:
            Split statistics
        """
        print(f"📊 Creating balanced splits (stratified by {stratify_by})...")
        
        # Load data
        if manifest_path.endswith('.json'):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = pd.read_csv(manifest_path).to_dict('records')
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Add duration bins if stratifying by duration
        if stratify_by == 'duration_bin':
            durations = []
            for _, row in df.iterrows():
                audio_path = self.data_dir / row.get('audio_path', row.get('path', ''))
                if audio_path.exists():
                    info = sf.info(str(audio_path))
                    durations.append(info.duration)
                else:
                    durations.append(0)
            
            df['duration'] = durations
            df['duration_bin'] = pd.cut(df['duration'], bins=[0, 5, 10, 20, 30], labels=['short', 'medium', 'long', 'very_long'])
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        # First split: train + (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df[stratify_by] if stratify_by in df.columns else None,
            random_state=42
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df[stratify_by] if stratify_by in temp_df.columns else None,
            random_state=42
        )
        
        # Save splits
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            output_path = output_dir / f"{name}_manifest.json"
            split_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        
        # Calculate statistics
        stats = {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_hours': train_df['duration'].sum() / 3600 if 'duration' in train_df.columns else 0,
            'val_hours': val_df['duration'].sum() / 3600 if 'duration' in val_df.columns else 0,
            'test_hours': test_df['duration'].sum() / 3600 if 'duration' in test_df.columns else 0,
        }
        
        print(f"✅ Created splits: Train={stats['train_samples']}, Val={stats['val_samples']}, Test={stats['test_samples']}")
        print(f"   Hours: Train={stats['train_hours']:.1f}h, Val={stats['val_hours']:.1f}h, Test={stats['test_hours']:.1f}h")
        
        return stats
