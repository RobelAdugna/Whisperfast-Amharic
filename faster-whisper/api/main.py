"""FastAPI server for Whisper inference API"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import tempfile
import os
import asyncio
from pathlib import Path

# Import inference utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from inference_utils import load_model_for_inference, transcribe_audio

# Create FastAPI app
app = FastAPI(
    title="Amharic Whisper API",
    description="REST API for Amharic speech recognition using fine-tuned Whisper",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TranscriptionRequest(BaseModel):
    language: str = "am"
    task: str = "transcribe"
    beam_size: int = 5
    model_path: Optional[str] = "./whisper_finetuned"

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str]
    segments: Optional[List[Dict]]
    duration: Optional[float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# Global model cache
model_cache = {}

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Amharic Whisper API starting...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Amharic Whisper API shutting down...")
    model_cache.clear()

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=len(model_cache) > 0,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=len(model_cache) > 0,
        version="1.0.0"
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = "am",
    task: str = "transcribe",
    beam_size: int = 5,
    model_path: str = "./whisper_finetuned"
):
    """
    Transcribe an audio file
    
    - **file**: Audio file (WAV, MP3, etc.)
    - **language**: Language code (default: 'am' for Amharic)
    - **task**: 'transcribe' or 'translate'
    - **beam_size**: Beam search size (1-10)
    - **model_path**: Path to model directory
    """
    # Validate file type
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg']
    if file.content_type not in allowed_types:
        # Allow anyway if content type is not set
        if file.content_type and not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {allowed_types}"
            )
    
    # Save uploaded file to temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Transcribe
        result = transcribe_audio(
            audio_path=temp_path,
            model_path=model_path,
            language=language,
            task=task,
            beam_size=beam_size
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return TranscriptionResponse(
            text=result['text'],
            language=result.get('language'),
            segments=result.get('segments'),
            duration=result.get('duration')
        )
    
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@app.post("/transcribe/batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    language: str = "am",
    task: str = "transcribe",
    beam_size: int = 5,
    model_path: str = "./whisper_finetuned"
):
    """
    Transcribe multiple audio files
    
    - **files**: List of audio files
    - **language**: Language code (default: 'am' for Amharic)
    - **task**: 'transcribe' or 'translate'
    - **beam_size**: Beam search size (1-10)
    - **model_path**: Path to model directory
    """
    results = []
    
    for file in files:
        try:
            # Use the single transcribe endpoint logic
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
            
            result = transcribe_audio(
                audio_path=temp_path,
                model_path=model_path,
                language=language,
                task=task,
                beam_size=beam_size
            )
            
            os.unlink(temp_path)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "text": result['text'],
                "language": result.get('language'),
                "duration": result.get('duration')
            })
        
        except Exception as e:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}

@app.get("/models")
async def list_models():
    """List available models"""
    models = []
    
    # Check for common model directories
    model_dirs = ["./whisper_finetuned", "./whisper_ct2_model", "./checkpoints"]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            models.append({
                "path": model_dir,
                "type": "directory",
                "exists": True
            })
    
    return {"models": models}

# WebSocket endpoint for real-time streaming
from fastapi import WebSocket
from api.websocket import websocket_endpoint

@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription"""
    await websocket_endpoint(websocket)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
