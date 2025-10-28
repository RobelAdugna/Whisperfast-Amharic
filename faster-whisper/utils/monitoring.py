"""Monitoring and metrics utilities"""

import time
from typing import Optional, Dict, Any
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available")


class MetricsCollector:
    """Collect and expose metrics for monitoring"""
    
    def __init__(self, app_name: str = "whisper_amharic"):
        """
        Initialize metrics collector
        
        Args:
            app_name: Application name for metrics
        """
        self.app_name = app_name
        
        if PROMETHEUS_AVAILABLE:
            # Training metrics
            self.training_loss = Gauge(
                f'{app_name}_training_loss',
                'Current training loss'
            )
            self.validation_wer = Gauge(
                f'{app_name}_validation_wer',
                'Current validation WER'
            )
            self.epoch = Gauge(
                f'{app_name}_epoch',
                'Current training epoch'
            )
            
            # Inference metrics
            self.inference_duration = Histogram(
                f'{app_name}_inference_duration_seconds',
                'Time spent on inference',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            )
            self.inference_count = Counter(
                f'{app_name}_inference_total',
                'Total number of inferences'
            )
            self.inference_errors = Counter(
                f'{app_name}_inference_errors_total',
                'Total number of inference errors'
            )
            
            # Audio metrics
            self.audio_duration = Histogram(
                f'{app_name}_audio_duration_seconds',
                'Duration of processed audio',
                buckets=[1, 5, 10, 30, 60, 120, 300]
            )
            
            # Model info
            self.model_info = Info(
                f'{app_name}_model',
                'Model information'
            )
        else:
            self.training_loss = None
            self.validation_wer = None
            self.epoch = None
            self.inference_duration = None
            self.inference_count = None
            self.inference_errors = None
            self.audio_duration = None
            self.model_info = None
    
    def record_training_loss(self, loss: float):
        """Record training loss"""
        if self.training_loss:
            self.training_loss.set(loss)
    
    def record_validation_wer(self, wer: float):
        """Record validation WER"""
        if self.validation_wer:
            self.validation_wer.set(wer)
    
    def record_epoch(self, epoch: int):
        """Record current epoch"""
        if self.epoch:
            self.epoch.set(epoch)
    
    def record_inference(self, duration: float, success: bool = True):
        """Record inference metrics"""
        if self.inference_duration:
            self.inference_duration.observe(duration)
        if self.inference_count:
            self.inference_count.inc()
        if not success and self.inference_errors:
            self.inference_errors.inc()
    
    def record_audio_duration(self, duration: float):
        """Record audio duration"""
        if self.audio_duration:
            self.audio_duration.observe(duration)
    
    def set_model_info(self, info: Dict[str, str]):
        """Set model information"""
        if self.model_info:
            self.model_info.info(info)


def timing_decorator(metrics_collector: Optional[MetricsCollector] = None):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                if metrics_collector:
                    metrics_collector.record_inference(duration, success)
        return wrapper
    return decorator


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.metrics[name] = {'start': time.time()}
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        if name in self.metrics and 'start' in self.metrics[name]:
            duration = time.time() - self.metrics[name]['start']
            self.metrics[name]['duration'] = duration
            return duration
        return 0.0
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metric by name"""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return self.metrics.copy()
    
    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()
