#!/usr/bin/env python3
"""YouTube Dataset Preparation for Whisper Fine-tuning

This module provides functionality to:
1. Download YouTube videos with Amharic subtitles
2. Extract and validate SRT files
3. Remove background music using Demucs
4. Create precise, SOTA-level datasets for Whisper training
"""

import os
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import timedelta
import tempfile
import shutil

try:
    import yt_dlp
    from yt_dlp.utils import DownloadError, ExtractorError
except ImportError:
    yt_dlp = None

try:
    import soundfile as sf
    import librosa
    import numpy as np
except ImportError:
    librosa = None

try:
    from .amharic_processing import AmharicTextProcessor
except ImportError:
    AmharicTextProcessor = None


class YouTubeDatasetPreparator:
    """Prepare high-quality Whisper datasets from YouTube videos with Amharic subtitles"""
    
    def __init__(
        self,
        output_dir: str = "./data/youtube_amharic",
        temp_dir: Optional[str] = None,
        use_demucs: bool = True,
        min_segment_duration: float = 1.0,
        max_segment_duration: float = 25.0,
        target_sample_rate: int = 16000,
        cookies_from_browser: Optional[str] = None,
        append_mode: bool = True
    ):
        """
        Initialize YouTube Dataset Preparator
        
        Args:
            output_dir: Directory to save processed dataset
            temp_dir: Temporary directory for processing (default: system temp)
            use_demucs: Whether to use Demucs for background music removal
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            target_sample_rate: Target sample rate for audio (Whisper uses 16kHz)
            cookies_from_browser: Browser to extract cookies from (chrome, firefox, edge, etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "youtube_whisper"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_demucs = use_demucs
        self.min_duration = min_segment_duration
        self.max_duration = max_segment_duration
        self.target_sr = target_sample_rate
        self.cookies_from_browser = cookies_from_browser
        self.append_mode = append_mode
        
        # Create subdirectories
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        self.srt_dir = self.output_dir / "subtitles"
        self.srt_dir.mkdir(exist_ok=True)
        
        self.segments_dir = self.output_dir / "segments"
        self.segments_dir.mkdir(exist_ok=True)
        
        # Master manifest for incremental building
        self.master_manifest_path = self.output_dir / "master_manifest.json"
        
        # Counter file for sequential naming
        self.counter_file = self.output_dir / ".audio_counter.txt"
        
        # Initialize Amharic processor if available
        self.amharic_processor = AmharicTextProcessor() if AmharicTextProcessor else None
    
    def _get_next_audio_counter(self) -> int:
        """Get the next audio counter for sequential naming"""
        if self.append_mode and self.counter_file.exists():
            try:
                with open(self.counter_file, 'r') as f:
                    return int(f.read().strip())
            except:
                return 1
        return 1
    
    def _save_audio_counter(self, counter: int):
        """Save the current audio counter"""
        with open(self.counter_file, 'w') as f:
            f.write(str(counter))
    
    def _load_existing_manifest(self) -> List[Dict]:
        """Load existing master manifest if in append mode"""
        if self.append_mode and self.master_manifest_path.exists():
            try:
                with open(self.master_manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing manifest: {e}")
                return []
        return []
    
    def _save_master_manifest(self, new_entries: List[Dict]) -> Dict:
        """Save or append to master manifest"""
        if self.append_mode:
            # Load existing entries
            existing_entries = self._load_existing_manifest()
            previous_count = len(existing_entries)
            
            # Append new entries
            all_entries = existing_entries + new_entries
            
            # Save combined manifest
            with open(self.master_manifest_path, 'w', encoding='utf-8') as f:
                json.dump(all_entries, f, ensure_ascii=False, indent=2)
            
            return {
                'previous_segments': previous_count,
                'new_segments': len(new_entries),
                'total_segments': len(all_entries),
                'master_manifest_path': str(self.master_manifest_path)
            }
        else:
            # Fresh mode: just save new entries
            with open(self.master_manifest_path, 'w', encoding='utf-8') as f:
                json.dump(new_entries, f, ensure_ascii=False, indent=2)
            
            return {
                'previous_segments': 0,
                'new_segments': len(new_entries),
                'total_segments': len(new_entries),
                'master_manifest_path': str(self.master_manifest_path)
            }
    
    def check_youtube_link(self, url: str) -> Dict:
        """
        Check if YouTube video has Amharic subtitles
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dict with video info and subtitle availability
        """
        if yt_dlp is None:
            raise ImportError("yt-dlp is not installed. Install with: pip install yt-dlp")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            # Bot bypass strategies - use multiple clients for maximum compatibility
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android', 'mweb'],  # iOS is most reliable
                    'player_skip': ['webpage', 'js', 'configs'],  # Skip all detection points
                    'skip': ['dash', 'hls'],  # Prefer direct formats
                }
            },
            'http_headers': {
                'User-Agent': 'com.google.ios.youtube/19.29.1 (iPhone16,2; U; CPU iOS 17_5_1 like Mac OS X;)',  # iOS YouTube app
                'Accept-Language': 'en-US,en;q=0.9',
            },
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Check for subtitles
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                has_amharic_manual = 'am' in subtitles
                has_amharic_auto = 'am' in automatic_captions
                
                result = {
                    'video_id': info.get('id'),
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'uploader': info.get('uploader'),
                    'has_amharic_srt': has_amharic_manual or has_amharic_auto,
                    'amharic_manual': has_amharic_manual,
                    'amharic_auto': has_amharic_auto,
                    'available_subtitles': list(subtitles.keys()),
                    'available_auto_captions': list(automatic_captions.keys()),
                    'url': url
                }
                
                return result
        
        except (DownloadError, ExtractorError) as e:
            raise ValueError(f"Failed to extract video info: {str(e)}")
    
    def download_video_and_subtitles(
        self,
        url: str,
        prefer_manual: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Path, Path, Dict]:
        """
        Download YouTube video and Amharic subtitles
        
        Args:
            url: YouTube video URL
            prefer_manual: Prefer manual subtitles over auto-generated
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (audio_path, subtitle_path, video_info)
        """
        if progress_callback:
            progress_callback(0.1, "Checking video and subtitles...")
        
        # First check if video has Amharic subtitles
        info = self.check_youtube_link(url)
        
        if not info['has_amharic_srt']:
            raise ValueError(
                f"No Amharic subtitles found for video: {info['title']}\n"
                f"Available subtitles: {', '.join(info['available_subtitles'])}"
            )
        
        video_id = info['video_id']
        
        if progress_callback:
            progress_callback(0.2, f"Downloading: {info['title'][:50]}...")
        
        # Download audio and subtitles
        audio_output = self.temp_dir / f"{video_id}.wav"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.temp_dir / f"{video_id}.%(ext)s"),
            'writesubtitles': True,
            'writeautomaticsub': not prefer_manual,
            'subtitleslangs': ['am'],
            'subtitlesformat': 'srt',
            'quiet': False,
            # Bot bypass strategies - iOS client is most reliable
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android', 'mweb'],  # iOS first, then Android, mobile web
                    'player_skip': ['webpage', 'js', 'configs'],  # Skip all detection points
                    'skip': ['dash', 'hls'],  # Prefer direct formats
                }
            },
            # iOS YouTube app headers - most reliable
            'http_headers': {
                'User-Agent': 'com.google.ios.youtube/19.29.1 (iPhone16,2; U; CPU iOS 17_5_1 like Mac OS X;)',
                'Accept-Language': 'en-US,en;q=0.9',
            },
        }
        
        # Add cookie support to bypass bot detection (optional)
        if self.cookies_from_browser:
            ydl_opts['cookiesfrombrowser'] = (self.cookies_from_browser,)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Find the downloaded subtitle file
        srt_path = self.temp_dir / f"{video_id}.am.srt"
        if not srt_path.exists():
            # Try without language code
            srt_path = self.temp_dir / f"{video_id}.srt"
        
        if not srt_path.exists():
            raise FileNotFoundError("Subtitle file not downloaded successfully")
        
        if not audio_output.exists():
            raise FileNotFoundError("Audio file not downloaded successfully")
        
        if progress_callback:
            progress_callback(0.4, "Download complete!")
        
        return audio_output, srt_path, info
    
    def remove_background_music(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable] = None
    ) -> Path:
        """
        Remove background music using Demucs to extract vocals only
        
        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to vocals-only audio file
        """
        if not self.use_demucs:
            return audio_path
        
        if progress_callback:
            progress_callback(0.5, "Removing background music with Demucs...")
        
        try:
            # Check if demucs is available
            result = subprocess.run(
                ['demucs', '--help'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("⚠️  Demucs not available, skipping music removal")
                return audio_path
        
        except FileNotFoundError:
            print("⚠️  Demucs not installed, skipping music removal")
            return audio_path
        
        # Run Demucs to separate vocals
        output_dir = self.temp_dir / "demucs_output"
        
        cmd = [
            'demucs',
            '--two-stems=vocals',
            '-n', 'htdemucs',
            '--out', str(output_dir),
            str(audio_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Find the vocals file
            vocals_path = output_dir / 'htdemucs' / audio_path.stem / 'vocals.wav'
            
            if vocals_path.exists():
                if progress_callback:
                    progress_callback(0.6, "Background music removed!")
                return vocals_path
            else:
                print("⚠️  Vocals file not found, using original audio")
                return audio_path
        
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Demucs processing failed: {e}, using original audio")
            return audio_path
    
    def parse_srt(self, srt_path: Path) -> List[Dict]:
        """
        Parse SRT subtitle file
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            List of subtitle segments with timestamps and text
        """
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newline (subtitle blocks)
        blocks = re.split(r'\n\n+', content.strip())
        
        segments = []
        
        for block in blocks:
            lines = block.strip().split('\n')
            
            if len(lines) < 3:
                continue
            
            # Parse timestamp line (format: 00:00:01,000 --> 00:00:03,500)
            timestamp_line = lines[1]
            match = re.match(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                timestamp_line
            )
            
            if not match:
                continue
            
            # Convert to seconds
            start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
            end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
            
            start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
            end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
            
            # Get text (everything after timestamp line)
            text = ' '.join(lines[2:]).strip()
            
            # Basic Amharic validation
            if self._contains_amharic(text):
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'text': text
                })
        
        return segments
    
    def _contains_amharic(self, text: str) -> bool:
        """Check if text contains Amharic characters"""
        # Amharic Unicode range: U+1200 to U+137F
        amharic_pattern = re.compile(r'[\u1200-\u137F]')
        return bool(amharic_pattern.search(text))
    
    def _calculate_quality_score(self, text: str, duration: float, audio_segment: np.ndarray = None, sr: int = 16000) -> Dict:
        """
        Calculate quality score for a dataset segment (Amharic-specific)
        
        Args:
            text: Transcript text
            duration: Audio duration in seconds
            audio_segment: Audio samples (optional, for audio quality checks)
            sr: Sample rate
            
        Returns:
            Dictionary with quality metrics and overall score
        """
        scores = {}
        
        # 1. Amharic purity check
        if self.amharic_processor:
            lang_scores = self.amharic_processor.detect_language(text)
            amharic_ratio = lang_scores.get('amharic', 0)
            scores['amharic_purity'] = amharic_ratio
        else:
            # Fallback: simple character count
            amharic_pattern = re.compile(r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]')
            amharic_chars = len(amharic_pattern.findall(text))
            total_chars = len([c for c in text if c.strip()])
            scores['amharic_purity'] = amharic_chars / total_chars if total_chars > 0 else 0
        
        # 2. Text/Audio ratio check (characters per second)
        char_count = len(text.strip())
        chars_per_second = char_count / duration if duration > 0 else 0
        # Typical Amharic speech: 8-15 characters per second
        # Score: 1.0 if in range [8, 15], decay outside
        if 8 <= chars_per_second <= 15:
            scores['text_audio_ratio'] = 1.0
        elif chars_per_second < 8:
            scores['text_audio_ratio'] = max(0, chars_per_second / 8)
        else:
            scores['text_audio_ratio'] = max(0, 1.0 - (chars_per_second - 15) / 20)
        
        # 3. Text length check
        if char_count < 10:
            scores['text_length'] = 0.3  # Too short
        elif char_count > 200:
            scores['text_length'] = 0.8  # Very long
        else:
            scores['text_length'] = 1.0  # Good length
        
        # 4. Punctuation presence (indicates proper transcription)
        has_punctuation = bool(re.search(r'[።፣፤፥፧,.!?;:]', text))
        scores['has_punctuation'] = 1.0 if has_punctuation else 0.7
        
        # 5. Audio quality check (if audio provided)
        if audio_segment is not None and len(audio_segment) > 0:
            # Silence ratio
            silence_threshold = 0.01  # Amplitude threshold
            silence_ratio = np.sum(np.abs(audio_segment) < silence_threshold) / len(audio_segment)
            scores['silence_ratio'] = max(0, 1.0 - silence_ratio)  # Less silence = better
            
            # Signal-to-noise ratio estimate (simple version)
            signal_power = np.mean(audio_segment ** 2)
            if signal_power > 0:
                # Estimate noise as quietest 10% of frames
                sorted_power = np.sort(audio_segment ** 2)
                noise_power = np.mean(sorted_power[:len(sorted_power)//10])
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                # SNR > 20dB is good, < 10dB is bad
                scores['audio_quality'] = min(1.0, max(0, (snr - 10) / 20))
            else:
                scores['audio_quality'] = 0.0
        else:
            scores['silence_ratio'] = 1.0  # Assume OK if no audio
            scores['audio_quality'] = 1.0
        
        # 6. Overall quality score (weighted average)
        weights = {
            'amharic_purity': 0.25,
            'text_audio_ratio': 0.20,
            'text_length': 0.15,
            'has_punctuation': 0.10,
            'silence_ratio': 0.15,
            'audio_quality': 0.15
        }
        
        overall_score = sum(scores[k] * weights[k] for k in weights.keys())
        scores['overall_score'] = overall_score
        
        return scores
    
    def _normalize_amharic_text(self, text: str) -> str:
        """
        Normalize Amharic text using AmharicTextProcessor
        
        Args:
            text: Raw Amharic text
            
        Returns:
            Normalized text
        """
        if self.amharic_processor:
            return self.amharic_processor.normalize_text(text)
        return text.strip()
    
    def create_dataset_segments(
        self,
        audio_path: Path,
        segments: List[Dict],
        video_id: str,
        progress_callback: Optional[Callable] = None,
        quality_threshold: float = 0.6,
        start_counter: Optional[int] = None
    ) -> Tuple[List[Dict], Dict, int]:
        """
        Create individual audio segments from subtitles with Amharic quality scoring
        
        Args:
            audio_path: Path to processed audio file
            segments: List of subtitle segments
            video_id: YouTube video ID (not used for naming anymore)
            progress_callback: Optional callback for progress updates
            quality_threshold: Minimum quality score (0-1) to include segment
            start_counter: Starting counter for sequential naming (auto-determined if None)
            
        Returns:
            Tuple of (dataset entries, quality statistics, next_counter)
        """
        if librosa is None:
            raise ImportError("librosa is not installed. Install with: pip install librosa")
        
        if progress_callback:
            progress_callback(0.7, "Creating dataset segments with quality scoring...")
        
        # Get starting counter for sequential naming
        if start_counter is None:
            audio_counter = self._get_next_audio_counter()
        else:
            audio_counter = start_counter
        
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        dataset_entries = []
        quality_stats = {
            'total_segments': 0,
            'passed_quality': 0,
            'failed_duration': 0,
            'failed_quality': 0,
            'avg_quality_score': 0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        }
        
        total_quality = 0
        
        for idx, segment in enumerate(segments):
            quality_stats['total_segments'] += 1
            
            # Validate segment duration
            if segment['duration'] < self.min_duration or segment['duration'] > self.max_duration:
                quality_stats['failed_duration'] += 1
                continue
            
            # Extract audio segment
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            audio_segment = audio[start_sample:end_sample]
            
            # Skip if segment is too short after extraction
            if len(audio_segment) < sr * self.min_duration:
                quality_stats['failed_duration'] += 1
                continue
            
            # Normalize text
            normalized_text = self._normalize_amharic_text(segment['text'])
            
            # Calculate quality score
            quality_scores = self._calculate_quality_score(
                normalized_text,
                segment['duration'],
                audio_segment,
                sr
            )
            
            overall_score = quality_scores['overall_score']
            total_quality += overall_score
            
            # Classify quality
            if overall_score >= 0.85:
                quality_stats['quality_distribution']['excellent'] += 1
            elif overall_score >= 0.75:
                quality_stats['quality_distribution']['good'] += 1
            elif overall_score >= 0.65:
                quality_stats['quality_distribution']['fair'] += 1
            else:
                quality_stats['quality_distribution']['poor'] += 1
            
            # Filter by quality threshold
            if overall_score < quality_threshold:
                quality_stats['failed_quality'] += 1
                continue
            
            # Sequential naming: audio_00001.wav, audio_00002.wav, etc.
            segment_filename = f"audio_{audio_counter:05d}.wav"
            segment_path = self.segments_dir / segment_filename
            
            sf.write(segment_path, audio_segment, sr)
            
            # Create dataset entry with quality metrics
            dataset_entries.append({
                'audio_path': str(segment_path.relative_to(self.output_dir)),
                'text': normalized_text,
                'duration': segment['duration'],
                'start': segment['start'],
                'end': segment['end'],
                'source_id': video_id,  # Track source but don't use for naming
                'segment_id': idx,
                'audio_id': audio_counter,  # Sequential ID
                'quality_score': round(overall_score, 3),
                'quality_metrics': {
                    'amharic_purity': round(quality_scores['amharic_purity'], 3),
                    'text_audio_ratio': round(quality_scores['text_audio_ratio'], 3),
                    'audio_quality': round(quality_scores['audio_quality'], 3)
                }
            })
            
            quality_stats['passed_quality'] += 1
            audio_counter += 1  # Increment for next segment
            
            if progress_callback and idx % 10 == 0:
                progress = 0.7 + (idx / len(segments)) * 0.2
                progress_callback(progress, f"Processing segment {idx + 1}/{len(segments)}")
        
        # Calculate average quality
        if quality_stats['total_segments'] > 0:
            quality_stats['avg_quality_score'] = round(total_quality / quality_stats['total_segments'], 3)
        
        # Save the updated counter
        self._save_audio_counter(audio_counter)
        
        return dataset_entries, quality_stats, audio_counter
    
    def process_youtube_video(
        self,
        url: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Complete pipeline: download, process, and create dataset from YouTube video
        
        Args:
            url: YouTube video URL
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            # Step 1: Download video and subtitles
            audio_path, srt_path, info = self.download_video_and_subtitles(
                url, progress_callback=progress_callback
            )
            
            # Step 2: Remove background music
            clean_audio_path = self.remove_background_music(
                audio_path, progress_callback=progress_callback
            )
            
            # Step 3: Parse subtitles
            if progress_callback:
                progress_callback(0.65, "Parsing subtitles...")
            
            segments = self.parse_srt(srt_path)
            
            if not segments:
                raise ValueError("No valid Amharic segments found in subtitles")
            
            # Step 4: Create dataset segments with quality scoring and sequential naming
            dataset_entries, quality_stats, _ = self.create_dataset_segments(
                clean_audio_path,
                segments,
                info['video_id'],
                progress_callback=progress_callback
            )
            
            if progress_callback:
                progress_callback(0.9, "Saving dataset manifest...")
            
            # Step 5: Save individual manifest
            manifest_path = self.output_dir / f"{info['video_id']}_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_entries, f, ensure_ascii=False, indent=2)
            
            # Step 6: Save to master manifest (incremental)
            master_info = self._save_master_manifest(dataset_entries)
            
            # Copy cleaned audio and subtitles to output
            final_audio_path = self.audio_dir / f"{info['video_id']}.wav"
            shutil.copy2(clean_audio_path, final_audio_path)
            
            final_srt_path = self.srt_dir / f"{info['video_id']}.srt"
            shutil.copy2(srt_path, final_srt_path)
            
            if progress_callback:
                progress_callback(1.0, "Processing complete!")
            
            # Calculate statistics
            total_duration = sum(entry['duration'] for entry in dataset_entries)
            
            result = {
                'success': True,
                'video_id': info['video_id'],
                'video_title': info['title'],
                'total_segments': len(dataset_entries),
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
                'manifest_path': str(manifest_path),
                'audio_path': str(final_audio_path),
                'srt_path': str(final_srt_path),
                'output_dir': str(self.output_dir),
                'dataset_entries': dataset_entries,
                'quality_stats': quality_stats,
                'master_manifest': master_info
            }
            
            return result
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        
        finally:
            # Cleanup temp files
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except:
                    pass
    
    def process_local_files(
        self,
        audio_path: str,
        srt_path: str,
        file_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process local audio + SRT file pair to create dataset
        
        Args:
            audio_path: Path to audio file (mp3, wav, etc.)
            srt_path: Path to SRT subtitle file
            file_id: Optional identifier for the file pair (default: audio filename)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results and statistics
        """
        from pathlib import Path
        
        try:
            audio_file = Path(audio_path)
            srt_file = Path(srt_path)
            
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not srt_file.exists():
                raise FileNotFoundError(f"SRT file not found: {srt_path}")
            
            # Use filename as ID if not provided
            if not file_id:
                file_id = audio_file.stem
            
            if progress_callback:
                progress_callback(0.1, "Loading audio file...")
            
            # Convert audio to standard format if needed
            if audio_file.suffix.lower() != '.wav':
                if progress_callback:
                    progress_callback(0.2, "Converting audio to WAV...")
                
                if librosa is None:
                    raise ImportError("librosa is required for audio conversion")
                
                # Load and convert
                audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
                
                # Save as WAV
                temp_wav = self.temp_dir / f"{file_id}_converted.wav"
                import soundfile as sf
                sf.write(temp_wav, audio, sr)
                audio_file = temp_wav
            
            if progress_callback:
                progress_callback(0.3, "Parsing SRT file...")
            
            # Parse SRT
            segments = self.parse_srt(srt_file)
            
            if not segments:
                raise ValueError("No valid Amharic segments found in SRT file")
            
            if progress_callback:
                progress_callback(0.4, f"Found {len(segments)} segments...")
            
            # Create dataset segments with quality scoring and sequential naming
            dataset_entries, quality_stats, _ = self.create_dataset_segments(
                audio_file,
                segments,
                file_id,
                progress_callback=progress_callback
            )
            
            if progress_callback:
                progress_callback(0.9, "Saving dataset manifest...")
            
            # Save individual manifest
            manifest_path = self.output_dir / f"{file_id}_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_entries, f, ensure_ascii=False, indent=2)
            
            # Save to master manifest (incremental)
            master_info = self._save_master_manifest(dataset_entries)
            
            # Copy files to output
            final_audio_path = self.audio_dir / f"{file_id}{audio_file.suffix}"
            shutil.copy2(audio_file, final_audio_path)
            
            final_srt_path = self.srt_dir / f"{file_id}.srt"
            shutil.copy2(srt_file, final_srt_path)
            
            if progress_callback:
                progress_callback(1.0, "Processing complete!")
            
            # Calculate statistics
            total_duration = sum(entry['duration'] for entry in dataset_entries)
            
            return {
                'success': True,
                'file_id': file_id,
                'audio_filename': audio_file.name,
                'srt_filename': srt_file.name,
                'total_segments': len(dataset_entries),
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
                'manifest_path': str(manifest_path),
                'audio_path': str(final_audio_path),
                'srt_path': str(final_srt_path),
                'output_dir': str(self.output_dir),
                'dataset_entries': dataset_entries,
                'quality_stats': quality_stats,
                'master_manifest': master_info
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_multiple_local_files(
        self,
        file_pairs: List[Tuple[str, str]],
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process multiple local audio + SRT file pairs
        
        Args:
            file_pairs: List of (audio_path, srt_path) tuples
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with combined results
        """
        results = []
        all_entries = []
        
        for idx, (audio_path, srt_path) in enumerate(file_pairs):
            if progress_callback:
                overall_progress = idx / len(file_pairs)
                progress_callback(overall_progress, f"Processing file {idx + 1}/{len(file_pairs)}")
            
            result = self.process_local_files(audio_path, srt_path, progress_callback=None)
            results.append(result)
            
            if result['success']:
                all_entries.extend(result['dataset_entries'])
        
        # Save combined manifest and master manifest
        combined_manifest_path = self.output_dir / "combined_manifest.json"
        with open(combined_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)
        
        # Save to master manifest (incremental)
        master_info = self._save_master_manifest(all_entries)
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_duration = sum(
            r.get('total_duration_seconds', 0) for r in results if r['success']
        )
        
        return {
            'total_files': len(file_pairs),
            'successful': successful,
            'failed': failed,
            'total_segments': len(all_entries),
            'total_duration_minutes': total_duration / 60,
            'combined_manifest': str(combined_manifest_path),
            'results': results
        }
    
    def process_multiple_videos(
        self,
        urls: List[str],
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process multiple YouTube videos
        
        Args:
            urls: List of YouTube URLs
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with combined results
        """
        results = []
        all_entries = []
        
        for idx, url in enumerate(urls):
            if progress_callback:
                overall_progress = idx / len(urls)
                progress_callback(overall_progress, f"Processing video {idx + 1}/{len(urls)}")
            
            result = self.process_youtube_video(url, progress_callback=None)
            results.append(result)
            
            if result['success']:
                all_entries.extend(result['dataset_entries'])
        
        # Save combined manifest
        combined_manifest_path = self.output_dir / "combined_manifest.json"
        with open(combined_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_duration = sum(
            r.get('total_duration_seconds', 0) for r in results if r['success']
        )
        
        return {
            'total_videos': len(urls),
            'successful': successful,
            'failed': failed,
            'total_segments': len(all_entries),
            'total_duration_minutes': total_duration / 60,
            'combined_manifest': str(combined_manifest_path),
            'results': results
        }
