"""Audio augmentation utilities for training"""

import numpy as np
from typing import Optional, Union

try:
    from audiomentations import (
        Compose, AddGaussianNoise, TimeStretch, PitchShift,
        Shift, Normalize, AddBackgroundNoise, RoomSimulator
    )
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False
    print("Warning: audiomentations not available. Install with: pip install audiomentations")

class AudioAugmentor:
    """Audio augmentation pipeline for training"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        augmentation_prob: float = 0.5
    ):
        """
        Initialize audio augmentor
        
        Args:
            sample_rate: Audio sample rate
            augmentation_prob: Probability of applying augmentation
        """
        self.sample_rate = sample_rate
        self.augmentation_prob = augmentation_prob
        
        if AUDIOMENTATIONS_AVAILABLE:
            self.augment = Compose([
                AddGaussianNoise(
                    min_amplitude=0.001,
                    max_amplitude=0.015,
                    p=0.3
                ),
                TimeStretch(
                    min_rate=0.9,
                    max_rate=1.1,
                    p=0.3
                ),
                PitchShift(
                    min_semitones=-2,
                    max_semitones=2,
                    p=0.3
                ),
                Shift(
                    min_shift=-0.5,
                    max_shift=0.5,
                    p=0.3
                ),
            ])
        else:
            self.augment = None
    
    def __call__(
        self,
        audio: np.ndarray,
        apply_augmentation: bool = True
    ) -> np.ndarray:
        """
        Apply augmentation to audio
        
        Args:
            audio: Input audio array
            apply_augmentation: Whether to apply augmentation
        
        Returns:
            Augmented audio array
        """
        if not apply_augmentation or self.augment is None:
            return audio
        
        if np.random.random() < self.augmentation_prob:
            return self.augment(samples=audio, sample_rate=self.sample_rate)
        
        return audio
    
    def add_noise(
        self,
        audio: np.ndarray,
        noise_level: float = 0.005
    ) -> np.ndarray:
        """Add gaussian noise to audio"""
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise
    
    def time_stretch(
        self,
        audio: np.ndarray,
        rate: float = 1.0
    ) -> np.ndarray:
        """Time stretch audio (simple implementation)"""
        if rate == 1.0:
            return audio
        
        # Simple resampling-based time stretch
        indices = np.arange(0, len(audio), rate)
        return np.interp(indices, np.arange(len(audio)), audio)
    
    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: float = 0
    ) -> np.ndarray:
        """Pitch shift audio (requires librosa)"""
        if n_steps == 0:
            return audio
        
        try:
            import librosa
            return librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=n_steps
            )
        except ImportError:
            print("Warning: librosa not available for pitch shifting")
            return audio


class SpecAugment:
    """SpecAugment for spectrogram augmentation"""
    
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        """
        Initialize SpecAugment
        
        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to spectrogram
        
        Args:
            spec: Input spectrogram (freq, time)
        
        Returns:
            Augmented spectrogram
        """
        spec = spec.copy()
        n_freqs, n_frames = spec.shape
        
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, n_freqs - f)
            spec[f0:f0 + f, :] = 0
        
        # Apply time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, n_frames - t)
            spec[:, t0:t0 + t] = 0
        
        return spec
