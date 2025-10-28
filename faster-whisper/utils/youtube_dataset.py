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
        cookies_from_browser: Optional[str] = None
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
        
        # Create subdirectories
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        self.srt_dir = self.output_dir / "subtitles"
        self.srt_dir.mkdir(exist_ok=True)
        
        self.segments_dir = self.output_dir / "segments"
        self.segments_dir.mkdir(exist_ok=True)
    
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
    
    def create_dataset_segments(
        self,
        audio_path: Path,
        segments: List[Dict],
        video_id: str,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Create individual audio segments from subtitles
        
        Args:
            audio_path: Path to processed audio file
            segments: List of subtitle segments
            video_id: YouTube video ID
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of dataset entries
        """
        if librosa is None:
            raise ImportError("librosa is not installed. Install with: pip install librosa")
        
        if progress_callback:
            progress_callback(0.7, "Creating dataset segments...")
        
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        dataset_entries = []
        
        for idx, segment in enumerate(segments):
            # Validate segment duration
            if segment['duration'] < self.min_duration or segment['duration'] > self.max_duration:
                continue
            
            # Extract audio segment
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            audio_segment = audio[start_sample:end_sample]
            
            # Skip if segment is too short after extraction
            if len(audio_segment) < sr * self.min_duration:
                continue
            
            # Save segment
            segment_filename = f"{video_id}_seg_{idx:04d}.wav"
            segment_path = self.segments_dir / segment_filename
            
            sf.write(segment_path, audio_segment, sr)
            
            # Create dataset entry
            dataset_entries.append({
                'audio_path': str(segment_path.relative_to(self.output_dir)),
                'text': segment['text'],
                'duration': segment['duration'],
                'start': segment['start'],
                'end': segment['end'],
                'video_id': video_id,
                'segment_id': idx
            })
            
            if progress_callback and idx % 10 == 0:
                progress = 0.7 + (idx / len(segments)) * 0.2
                progress_callback(progress, f"Processing segment {idx + 1}/{len(segments)}")
        
        return dataset_entries
    
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
            
            # Step 4: Create dataset segments
            dataset_entries = self.create_dataset_segments(
                clean_audio_path,
                segments,
                info['video_id'],
                progress_callback=progress_callback
            )
            
            if progress_callback:
                progress_callback(0.9, "Saving dataset manifest...")
            
            # Step 5: Save manifest
            manifest_path = self.output_dir / f"{info['video_id']}_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_entries, f, ensure_ascii=False, indent=2)
            
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
                'dataset_entries': dataset_entries
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
