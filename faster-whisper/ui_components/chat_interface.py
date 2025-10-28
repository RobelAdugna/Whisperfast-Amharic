"""Multimodal chat interface for audio transcription"""

import gradio as gr
from typing import List, Tuple, Optional, Callable
import time


def create_chat_interface(
    transcribe_fn: Callable,
    process_text_fn: Optional[Callable] = None,
    enable_llm: bool = False
) -> gr.Blocks:
    """
    Create multimodal chat interface
    
    Args:
        transcribe_fn: Function to transcribe audio
        process_text_fn: Optional function to process text with LLM
        enable_llm: Whether to enable LLM post-processing
    
    Returns:
        Gradio Blocks interface
    """
    
    def handle_audio_message(
        audio,
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Handle audio input in chat"""
        if audio is None:
            return history, ""
        
        # Transcribe audio
        try:
            transcript = transcribe_fn(audio)
            
            # Process with LLM if enabled
            if enable_llm and process_text_fn:
                response = process_text_fn(transcript)
            else:
                response = f"Transcribed: {transcript}"
            
            # Add to history
            history.append((f"🎤 {transcript}", response))
            
            return history, ""
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append(("🎤 [Audio]", error_msg))
            return history, ""
    
    def handle_text_message(
        text: str,
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Handle text input in chat"""
        if not text.strip():
            return history, ""
        
        try:
            # Process with LLM if enabled
            if enable_llm and process_text_fn:
                response = process_text_fn(text)
            else:
                response = f"Received: {text}"
            
            # Add to history
            history.append((text, response))
            
            return history, ""
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append((text, error_msg))
            return history, ""
    
    with gr.Blocks() as chat_interface:
        gr.Markdown("### 💬 Multimodal Chat Interface")
        gr.Markdown("Interact using voice or text. Audio will be transcribed automatically.")
        
        chatbot = gr.Chatbot(
            label="Conversation",
            height=400,
            type="tuples"
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Type a message",
                    placeholder="Type here or use microphone...",
                    lines=1
                )
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="🎤 Record",
                    sources=["microphone"],
                    type="filepath"
                )
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat")
            submit_btn = gr.Button("📤 Send", variant="primary")
        
        # Event handlers
        text_input.submit(
            fn=handle_text_message,
            inputs=[text_input, chatbot],
            outputs=[chatbot, text_input]
        )
        
        submit_btn.click(
            fn=handle_text_message,
            inputs=[text_input, chatbot],
            outputs=[chatbot, text_input]
        )
        
        audio_input.change(
            fn=handle_audio_message,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, audio_input]
        )
        
        clear_btn.click(
            fn=lambda: ([], "", None),
            outputs=[chatbot, text_input, audio_input]
        )
    
    return chat_interface


def create_streaming_chat(
    streaming_transcribe_fn: Callable,
    vad_processor: Optional[object] = None
) -> gr.Blocks:
    """
    Create streaming chat interface with real-time transcription
    
    Args:
        streaming_transcribe_fn: Function for streaming transcription
        vad_processor: Optional VAD processor
    
    Returns:
        Gradio Blocks interface
    """
    
    def process_streaming_audio(
        audio_stream,
        state: dict
    ) -> Tuple[str, dict]:
        """Process streaming audio chunks"""
        if audio_stream is None:
            return state.get('transcript', ''), state
        
        # Accumulate audio
        if 'buffer' not in state:
            state['buffer'] = []
        
        state['buffer'].append(audio_stream)
        
        # Check for speech with VAD if available
        if vad_processor:
            is_speech, confidence = vad_processor.is_speech(audio_stream)
            
            if not is_speech and len(state['buffer']) > 0:
                # End of speech detected, transcribe buffer
                try:
                    transcript = streaming_transcribe_fn(state['buffer'])
                    state['transcript'] = state.get('transcript', '') + ' ' + transcript
                    state['buffer'] = []
                except Exception as e:
                    print(f"Transcription error: {e}")
        else:
            # No VAD, transcribe periodically
            if len(state['buffer']) >= 10:  # Every 10 chunks
                try:
                    transcript = streaming_transcribe_fn(state['buffer'])
                    state['transcript'] = state.get('transcript', '') + ' ' + transcript
                    state['buffer'] = []
                except Exception as e:
                    print(f"Transcription error: {e}")
        
        return state.get('transcript', ''), state
    
    with gr.Blocks() as streaming_interface:
        gr.Markdown("### 🎙️ Real-time Streaming Transcription")
        gr.Markdown("Speak into your microphone for live transcription.")
        
        state = gr.State({})
        
        with gr.Row():
            audio_stream = gr.Audio(
                label="Microphone Stream",
                sources=["microphone"],
                streaming=True,
                type="numpy"
            )
        
        transcript_output = gr.Textbox(
            label="Live Transcript",
            lines=10,
            interactive=False
        )
        
        with gr.Row():
            clear_transcript = gr.Button("🗑️ Clear Transcript")
        
        # Stream processing
        audio_stream.stream(
            fn=process_streaming_audio,
            inputs=[audio_stream, state],
            outputs=[transcript_output, state]
        )
        
        clear_transcript.click(
            fn=lambda: ("", {}),
            outputs=[transcript_output, state]
        )
    
    return streaming_interface
