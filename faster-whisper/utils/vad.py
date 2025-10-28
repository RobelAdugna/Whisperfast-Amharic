"""Voice Activity Detection utilities using Silero VAD"""

import torch
import numpy as np
from typing import List, Dict, Tuple

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

class VADProcessor:
    """Voice Activity Detection processor using Silero VAD"""
    
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        """
        Initialize VAD processor
        
        Args:
            sample_rate: Audio sample rate (default: 16000)
            threshold: VAD threshold (default: 0.5)
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model"""
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            self.save_audio = utils[1]
            self.read_audio = utils[2]
            self.VADIterator = utils[3]
            self.collect_chunks = utils[4]
        except Exception as e:
            print(f"Warning: Could not load Silero VAD model: {e}")
            self.model = None
    
    def get_speech_segments(
        self, 
        audio: np.ndarray,
        return_seconds: bool = True
    ) -> List[Dict[str, float]]:
        """
        Detect speech segments in audio
        
        Args:
            audio: Audio array (numpy array)
            return_seconds: Return timestamps in seconds (default: True)
        
        Returns:
            List of speech segments with 'start' and 'end' timestamps
        """
        if self.model is None:
            return [{'start': 0, 'end': len(audio) / self.sample_rate}]
        
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
        
        # Convert to seconds if requested
        if return_seconds:
            segments = [
                {
                    'start': seg['start'] / self.sample_rate,
                    'end': seg['end'] / self.sample_rate
                }
                for seg in speech_timestamps
            ]
        else:
            segments = speech_timestamps
        
        return segments
    
    def is_speech(
        self,
        audio_chunk: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if audio chunk contains speech
        
        Args:
            audio_chunk: Audio chunk (numpy array)
        
        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        if self.model is None:
            return True, 1.0
        
        # Convert to torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk
        
        # Get speech probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        return speech_prob > self.threshold, speech_prob
    
    def filter_non_speech(
        self,
        audio: np.ndarray
    ) -> np.ndarray:
        """
        Filter out non-speech segments from audio
        
        Args:
            audio: Audio array
        
        Returns:
            Filtered audio with only speech segments
        """
        if self.model is None:
            return audio
        
        # Get speech segments
        segments = self.get_speech_segments(audio, return_seconds=False)
        
        # Collect speech chunks
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
        
        speech_audio = self.collect_chunks(segments, audio_tensor)
        
        # Convert back to numpy if needed
        if isinstance(audio, np.ndarray):
            return speech_audio.numpy()
        return speech_audio
