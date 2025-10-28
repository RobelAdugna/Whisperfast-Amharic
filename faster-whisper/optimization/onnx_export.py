"""ONNX export utilities for Whisper model"""

import torch
import onnx
from pathlib import Path
from typing import Optional, Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))

def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 14,
    dynamic_axes: bool = True,
    optimize: bool = True
) -> Dict[str, str]:
    """
    Export Whisper model to ONNX format
    
    Args:
        model_path: Path to PyTorch model
        output_path: Output path for ONNX model
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch/sequence axes
        optimize: Apply ONNX optimization
    
    Returns:
        Dictionary with export info
    """
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        return {"error": "transformers not available"}
    
    try:
        # Load model
        print(f"Loading model from {model_path}...")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        
        model.eval()
        
        # Prepare dummy input
        dummy_input_features = torch.randn(1, 80, 3000)  # (batch, n_mels, sequence)
        dummy_decoder_input_ids = torch.ones((1, 1), dtype=torch.long)
        
        # Define input/output names
        input_names = ['input_features', 'decoder_input_ids']
        output_names = ['logits']
        
        # Define dynamic axes if enabled
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_features': {0: 'batch', 2: 'sequence'},
                'decoder_input_ids': {0: 'batch', 1: 'decoder_sequence'},
                'logits': {0: 'batch', 1: 'decoder_sequence'}
            }
        else:
            dynamic_axes_dict = {}
        
        # Export to ONNX
        print(f"Exporting to ONNX...")
        torch.onnx.export(
            model,
            (dummy_input_features, dummy_decoder_input_ids),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True
        )
        
        # Verify ONNX model
        print("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Optimize if requested
        if optimize:
            print("Optimizing ONNX model...")
            try:
                from onnxruntime.transformers import optimizer
                optimized_model = optimizer.optimize_model(
                    output_path,
                    model_type='bert',  # Use bert optimizations
                    num_heads=model.config.encoder_attention_heads,
                    hidden_size=model.config.d_model
                )
                optimized_path = output_path.replace('.onnx', '_optimized.onnx')
                optimized_model.save_model_to_file(optimized_path)
                print(f"Optimized model saved to {optimized_path}")
            except Exception as e:
                print(f"Optimization failed: {e}")
        
        return {
            "status": "success",
            "output_path": output_path,
            "opset_version": opset_version,
            "dynamic_axes": dynamic_axes
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def quantize_onnx(
    onnx_path: str,
    output_path: str,
    quantization_mode: str = "dynamic"
) -> Dict[str, str]:
    """
    Quantize ONNX model to INT8
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output path for quantized model
        quantization_mode: 'dynamic' or 'static'
    
    Returns:
        Dictionary with quantization info
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        return {"error": "onnxruntime not available"}
    
    try:
        print(f"Quantizing model: {onnx_path}")
        
        if quantization_mode == "dynamic":
            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QInt8
            )
        else:
            # Static quantization would require calibration data
            return {"error": "Static quantization not yet implemented"}
        
        # Check file size reduction
        import os
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        reduction = (1 - quantized_size / original_size) * 100
        
        return {
            "status": "success",
            "output_path": output_path,
            "original_size_mb": f"{original_size:.2f}",
            "quantized_size_mb": f"{quantized_size:.2f}",
            "size_reduction_percent": f"{reduction:.1f}"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
