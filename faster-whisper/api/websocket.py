"""WebSocket handler for real-time streaming transcription"""

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Import utilities
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.vad import VADProcessor
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.vad_processor = VADProcessor() if VAD_AVAILABLE else None
    
    async def connect(self, websocket: WebSocket):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Remove disconnected clients
                self.disconnect(connection)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription
    
    Protocol:
    - Client sends: {"type": "audio", "data": [...], "sample_rate": 16000}
    - Server sends: {"type": "transcript", "text": "...", "partial": true/false}
    - Server sends: {"type": "vad", "is_speech": true/false, "confidence": 0.9}
    """
    await manager.connect(websocket)
    
    audio_buffer = []
    
    try:
        # Send welcome message
        await manager.send_message({
            "type": "connection",
            "status": "connected",
            "vad_available": VAD_AVAILABLE
        }, websocket)
        
        while True:
            # Receive audio data
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                # Process audio chunk
                audio_chunk = np.array(data.get("data", []), dtype=np.float32)
                sample_rate = data.get("sample_rate", 16000)
                
                if len(audio_chunk) == 0:
                    continue
                
                # VAD check if available
                if manager.vad_processor:
                    is_speech, confidence = manager.vad_processor.is_speech(audio_chunk)
                    
                    # Send VAD status
                    await manager.send_message({
                        "type": "vad",
                        "is_speech": bool(is_speech),
                        "confidence": float(confidence)
                    }, websocket)
                    
                    if is_speech:
                        audio_buffer.append(audio_chunk)
                    else:
                        # End of speech - transcribe if buffer has data
                        if len(audio_buffer) > 0:
                            # Placeholder for actual transcription
                            # In production, this would call the transcription model
                            full_audio = np.concatenate(audio_buffer)
                            
                            # Send transcript (placeholder)
                            await manager.send_message({
                                "type": "transcript",
                                "text": f"[Transcribed {len(full_audio)} samples]",
                                "partial": False,
                                "duration": len(full_audio) / sample_rate
                            }, websocket)
                            
                            # Clear buffer
                            audio_buffer = []
                else:
                    # No VAD - accumulate and transcribe periodically
                    audio_buffer.append(audio_chunk)
                    
                    # Transcribe every ~2 seconds worth of audio
                    total_samples = sum(len(chunk) for chunk in audio_buffer)
                    if total_samples >= sample_rate * 2:
                        full_audio = np.concatenate(audio_buffer)
                        
                        # Send transcript (placeholder)
                        await manager.send_message({
                            "type": "transcript",
                            "text": f"[Transcribed {len(full_audio)} samples]",
                            "partial": False,
                            "duration": len(full_audio) / sample_rate
                        }, websocket)
                        
                        # Clear buffer
                        audio_buffer = []
            
            elif data.get("type") == "reset":
                # Clear audio buffer
                audio_buffer = []
                await manager.send_message({
                    "type": "reset",
                    "status": "buffer_cleared"
                }, websocket)
            
            elif data.get("type") == "ping":
                # Respond to ping
                await manager.send_message({
                    "type": "pong"
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client disconnected")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)
        try:
            await manager.send_message({
                "type": "error",
                "message": str(e)
            }, websocket)
        except:
            pass
