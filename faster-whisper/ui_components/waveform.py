"""Waveform visualization component for Gradio"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple


def create_waveform_plot(
    audio: np.ndarray,
    sample_rate: int = 16000,
    title: str = "Audio Waveform",
    highlight_segments: Optional[list] = None,
    show_grid: bool = True
) -> go.Figure:
    """
    Create interactive waveform visualization
    
    Args:
        audio: Audio array
        sample_rate: Sample rate of audio
        title: Plot title
        highlight_segments: List of (start, end) tuples to highlight
        show_grid: Whether to show grid
    
    Returns:
        Plotly figure object
    """
    # Generate time axis
    duration = len(audio) / sample_rate
    time = np.linspace(0, duration, len(audio))
    
    # Create figure
    fig = go.Figure()
    
    # Add waveform trace
    fig.add_trace(go.Scatter(
        x=time,
        y=audio,
        mode='lines',
        name='Waveform',
        line=dict(color='#3b82f6', width=1),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'
    ))
    
    # Add highlighted segments if provided
    if highlight_segments:
        for i, (start, end) in enumerate(highlight_segments):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor='rgba(255, 0, 0, 0.2)',
                layer='below',
                line_width=0,
                annotation_text=f'Segment {i+1}',
                annotation_position='top left'
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    if show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def create_spectrogram_plot(
    audio: np.ndarray,
    sample_rate: int = 16000,
    title: str = "Spectrogram",
    n_fft: int = 2048,
    hop_length: int = 512
) -> go.Figure:
    """
    Create spectrogram visualization
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        title: Plot title
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        Plotly figure object
    """
    try:
        import librosa
        import librosa.display
    except ImportError:
        # Return empty figure if librosa not available
        fig = go.Figure()
        fig.add_annotation(
            text="Librosa not available for spectrogram generation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Compute spectrogram
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Create time and frequency axes
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]),
        sr=sample_rate,
        hop_length=hop_length
    )
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        x=times,
        y=frequencies,
        colorscale='Viridis',
        colorbar=dict(title='dB'),
        hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.0f}Hz<br>Magnitude: %{z:.1f}dB<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Frequency (Hz)',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_audio_stats_display(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> dict:
    """
    Calculate and return audio statistics
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
    
    Returns:
        Dictionary of audio statistics
    """
    duration = len(audio) / sample_rate
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    
    # Estimate SNR (simple approximation)
    # Assume last 10% is noise floor
    noise_floor = np.std(audio[int(len(audio) * 0.9):])
    signal_power = rms
    snr_estimate = 20 * np.log10(signal_power / (noise_floor + 1e-10))
    
    return {
        'duration': f"{duration:.2f} seconds",
        'sample_rate': f"{sample_rate} Hz",
        'channels': '1 (Mono)',
        'rms_amplitude': f"{rms:.4f}",
        'peak_amplitude': f"{peak:.4f}",
        'estimated_snr': f"{snr_estimate:.1f} dB",
        'samples': len(audio)
    }
