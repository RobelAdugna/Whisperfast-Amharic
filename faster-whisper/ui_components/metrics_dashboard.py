"""Metrics dashboard component for training visualization"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional


def create_training_metrics_plot(
    metrics_history: Dict[str, List[float]],
    title: str = "Training Metrics"
) -> go.Figure:
    """
    Create interactive training metrics visualization
    
    Args:
        metrics_history: Dictionary with 'loss', 'wer', 'lr' lists
        title: Plot title
    
    Returns:
        Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Loss',
            'Word Error Rate (WER)',
            'Learning Rate',
            'Gradient Norm'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}]
        ]
    )
    
    # Get epochs
    epochs = list(range(len(metrics_history.get('loss', []))))
    
    # Training loss
    if 'loss' in metrics_history and metrics_history['loss']:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=metrics_history['loss'],
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    if 'val_loss' in metrics_history and metrics_history['val_loss']:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=metrics_history['val_loss'],
                mode='lines+markers',
                name='Val Loss',
                line=dict(color='#f97316', width=2, dash='dash'),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # WER
    if 'wer' in metrics_history and metrics_history['wer']:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=metrics_history['wer'],
                mode='lines+markers',
                name='WER',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
    
    # Learning rate
    if 'lr' in metrics_history and metrics_history['lr']:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=metrics_history['lr'],
                mode='lines',
                name='Learning Rate',
                line=dict(color='#10b981', width=2),
            ),
            row=2, col=1
        )
    
    # Gradient norm
    if 'grad_norm' in metrics_history and metrics_history['grad_norm']:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=metrics_history['grad_norm'],
                mode='lines',
                name='Gradient Norm',
                line=dict(color='#8b5cf6', width=2),
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="WER (%)", row=1, col=2)
    fig.update_yaxes(title_text="LR", row=2, col=1, type="log")
    fig.update_yaxes(title_text="Norm", row=2, col=2)
    
    fig.update_layout(
        title_text=title,
        showlegend=True,
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_realtime_loss_plot(
    loss_values: List[float],
    window_size: int = 100
) -> go.Figure:
    """
    Create real-time loss plot with moving window
    
    Args:
        loss_values: List of loss values
        window_size: Number of recent values to display
    
    Returns:
        Plotly figure
    """
    # Take last window_size values
    recent_losses = loss_values[-window_size:] if len(loss_values) > window_size else loss_values
    steps = list(range(len(loss_values) - len(recent_losses), len(loss_values)))
    
    fig = go.Figure()
    
    # Add loss trace
    fig.add_trace(go.Scatter(
        x=steps,
        y=recent_losses,
        mode='lines',
        name='Loss',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Add moving average
    if len(recent_losses) > 10:
        window = min(20, len(recent_losses) // 4)
        moving_avg = np.convolve(
            recent_losses,
            np.ones(window) / window,
            mode='valid'
        )
        ma_steps = steps[window-1:]
        
        fig.add_trace(go.Scatter(
            x=ma_steps,
            y=moving_avg,
            mode='lines',
            name=f'MA({window})',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='Real-time Training Loss',
        xaxis_title='Step',
        yaxis_title='Loss',
        template='plotly_white',
        hovermode='x unified',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_comparison_plot(
    model_metrics: Dict[str, Dict[str, List[float]]],
    metric_name: str = 'wer'
) -> go.Figure:
    """
    Create comparison plot for multiple models
    
    Args:
        model_metrics: Dictionary of {model_name: {metric: values}}
        metric_name: Name of metric to compare
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
    
    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        if metric_name in metrics:
            epochs = list(range(len(metrics[metric_name])))
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=metrics[metric_name],
                mode='lines+markers',
                name=model_name,
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=f'Model Comparison: {metric_name.upper()}',
        xaxis_title='Epoch',
        yaxis_title=metric_name.upper(),
        template='plotly_white',
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig


def create_confusion_matrix(
    predictions: List[str],
    references: List[str],
    top_n: int = 20
) -> go.Figure:
    """
    Create confusion matrix for character-level errors
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        top_n: Number of top confused characters to show
    
    Returns:
        Plotly figure
    """
    # Simple character-level confusion tracking
    confusion_counts = {}
    
    for pred, ref in zip(predictions, references):
        for p_char, r_char in zip(pred, ref):
            if p_char != r_char:
                key = (r_char, p_char)
                confusion_counts[key] = confusion_counts.get(key, 0) + 1
    
    # Get top N confusions
    top_confusions = sorted(
        confusion_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    if not top_confusions:
        fig = go.Figure()
        fig.add_annotation(
            text="No confusions to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create bar chart
    labels = [f"'{ref}' → '{pred}'" for (ref, pred), _ in top_confusions]
    counts = [count for _, count in top_confusions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts,
            y=labels,
            orientation='h',
            marker=dict(color='#3b82f6')
        )
    ])
    
    fig.update_layout(
        title='Top Character Confusions',
        xaxis_title='Count',
        yaxis_title='Confusion (Reference → Prediction)',
        template='plotly_white',
        height=max(400, len(labels) * 25),
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig
