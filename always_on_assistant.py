#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Always-On Voice Assistant with Gemini Live API
Features:
- Voice input and output
- Core memory management
- Context window management with clearing
- Full interaction recording
- Automatic reconnection on connection failure with exponential backoff
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
import wave
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import traceback
from asyncio.exceptions import CancelledError

try:
    import pyaudio
except ImportError:
    pyaudio = None

from google import genai
from google.genai import types

# Import new memory system
from memory_system import IntegratedMemorySystem

# Configuration
MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"  # Voice model
MEMORY_PROCESSOR_MODEL = "gemini-flash-latest"  # Memory processing model
MEMORY_FILE = "assistant_memories.json"
INTERACTION_LOG_DIR = "interaction_logs"
AUDIO_RECORDINGS_DIR = "audio_recordings"

# Audio settings
CHANNELS = 1
RATE = 24000
CHUNK = 1024
FORMAT = pyaudio.paInt16 if pyaudio else None
SAMPLE_WIDTH = 2

# Context management
MAX_TURNS_BEFORE_SUMMARIZE = 20  # Summarize and clear after this many turns
WARNING_TURNS = 15  # Start warning about context at this point

# Memory processing settings
STM_MAX_TURNS = 20  # Maximum turns to keep in short-term memory
MEMORY_PROCESS_THRESHOLD = 8  # Process STM to LTM after this many turns

# Connection management
MAX_RECONNECT_ATTEMPTS = 5  # Maximum number of reconnection attempts
INITIAL_RECONNECT_DELAY = 2  # Initial delay in seconds before reconnecting
MAX_RECONNECT_DELAY = 60  # Maximum delay in seconds between reconnection attempts


class MemoryManager:
    """Manages persistent core memories for the assistant"""
    
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = Path(memory_file)
        self.memories: Dict[str, Any] = self._load_memories()
        
    def _load_memories(self) -> Dict[str, Any]:
        """Load memories from JSON file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not parse {self.memory_file}, starting fresh")
                return self._default_memories()
        return self._default_memories()
    
    def _default_memories(self) -> Dict[str, Any]:
        """Default memory structure"""
        return {
            "user_info": {},
            "preferences": {},
            "important_facts": [],
            "conversation_summaries": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def save_memories(self):
        """Save memories to JSON file"""
        self.memories["last_updated"] = datetime.now().isoformat()
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)
        logging.info("Memories saved")
    
    def add_fact(self, fact: str):
        """Add an important fact to memory"""
        self.memories["important_facts"].append({
            "fact": fact,
            "timestamp": datetime.now().isoformat()
        })
        self.save_memories()
    
    def add_conversation_summary(self, summary: str):
        """Add a conversation summary when context is cleared"""
        self.memories["conversation_summaries"].append({
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 10 summaries
        if len(self.memories["conversation_summaries"]) > 10:
            self.memories["conversation_summaries"] = self.memories["conversation_summaries"][-10:]
        self.save_memories()
    
    def get_memory_context(self) -> str:
        """Generate a context string from memories to inject into conversations"""
        context_parts = ["=== CORE MEMORIES ==="]
        
        if self.memories["user_info"]:
            context_parts.append("\nUser Information:")
            for key, value in self.memories["user_info"].items():
                context_parts.append(f"- {key}: {value}")
        
        if self.memories["preferences"]:
            context_parts.append("\nUser Preferences:")
            for key, value in self.memories["preferences"].items():
                context_parts.append(f"- {key}: {value}")
        
        if self.memories["important_facts"]:
            context_parts.append("\nImportant Facts:")
            for fact in self.memories["important_facts"][-10:]:  # Last 10 facts
                context_parts.append(f"- {fact['fact']}")
        
        if self.memories["conversation_summaries"]:
            context_parts.append("\nRecent Conversation Summaries:")
            for summary in self.memories["conversation_summaries"][-3:]:  # Last 3 summaries
                context_parts.append(f"- {summary['summary']}")
        
        context_parts.append("=== END CORE MEMORIES ===\n")
        return "\n".join(context_parts)


class InteractionLogger:
    """Logs all interactions to files"""
    
    def __init__(self, log_dir: str = INTERACTION_LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_session_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
    def log_interaction(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Log a single interaction"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        with open(self.current_session_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


class AudioRecorder:
    """Handles audio recording from microphone"""
    
    def __init__(self):
        if pyaudio is None:
            raise ImportError("PyAudio is required for audio recording. Install with: pip install pyaudio")
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.recordings_dir = Path(AUDIO_RECORDINGS_DIR)
        self.recordings_dir.mkdir(exist_ok=True)
        
    def start_recording(self):
        """Start recording from microphone"""
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.is_recording = True
        logging.info("Microphone recording started")
    
    def stop_recording(self):
        """Stop recording"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.is_recording = False
        logging.info("Microphone recording stopped")
    
    def read_chunk(self) -> bytes:
        """Read a chunk of audio data"""
        if self.stream and self.is_recording:
            return self.stream.read(CHUNK, exception_on_overflow=False)
        return b''
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        self.audio.terminate()
    
    def save_audio_to_file(self, audio_data: bytes, filename: str):
        """Save raw audio data to WAV file"""
        filepath = self.recordings_dir / filename
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        return filepath


class AudioPlayer:
    """Handles audio playback to speakers"""
    
    def __init__(self):
        if pyaudio is None:
            raise ImportError("PyAudio is required for audio playback. Install with: pip install pyaudio")
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def start_playback(self):
        """Start playback stream"""
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
        )
        logging.info("Audio playback started")
    
    def play_chunk(self, data: bytes):
        """Play a chunk of audio"""
        if self.stream:
            self.stream.write(data)
    
    def stop_playback(self):
        """Stop playback"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        logging.info("Audio playback stopped")
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_playback()
        self.audio.terminate()


@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    """Context manager for writing WAV files"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf


class AlwaysOnAssistant:
    """Main always-on voice assistant"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Setup
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # New integrated memory system (replaces old MemoryManager)
        self.memory_system = IntegratedMemorySystem(
            api_key=self.api_key,
            stm_max_turns=STM_MAX_TURNS,
            process_threshold=MEMORY_PROCESS_THRESHOLD
        )
        
        # Keep old memory manager for backwards compatibility (can be removed later)
        self.memory_manager = MemoryManager()
        self.interaction_logger = InteractionLogger()
        
        # Audio components (optional if PyAudio not available)
        self.use_audio = pyaudio is not None
        if self.use_audio:
            self.audio_recorder = AudioRecorder()
            self.audio_player = AudioPlayer()
        else:
            logging.warning("PyAudio not available, running in text-only mode")
            self.audio_recorder = None
            self.audio_player = None
        
        # Session state
        self.session = None
        self.turn_count = 0
        self.send_queue = asyncio.Queue()
        self.is_running = False
        self.connection_active = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = MAX_RECONNECT_ATTEMPTS
        self.initial_reconnect_delay = INITIAL_RECONNECT_DELAY
        
        # Background memory processing task
        self.memory_processing_task = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AlwaysOnAssistant')
    
    async def run(self):
        """Main run loop with automatic reconnection"""
        self.is_running = True
        self.logger.info("Starting Always-On Assistant...")
        
        try:
            while self.is_running:
                try:
                    await self._run_session()
                    # If we exit cleanly, stop the loop
                    break
                except (ConnectionError, Exception) as e:
                    if not self.is_running:
                        # User requested shutdown
                        break
                    
                    self.reconnect_attempts += 1
                    if self.reconnect_attempts >= self.max_reconnect_attempts:
                        self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Giving up.")
                        break
                    
                    # Exponential backoff with max delay cap
                    delay = min(
                        self.initial_reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
                        MAX_RECONNECT_DELAY
                    )
                    self.logger.warning(f"Connection lost: {e}")
                    self.logger.info(f"Reconnecting in {delay:.1f} seconds... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                    await asyncio.sleep(delay)
                    
        except KeyboardInterrupt:
            self.logger.info("Assistant shutting down...")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            traceback.print_exc()
        finally:
            await self._cleanup()
    
    async def _run_session(self):
        """Run a single session (can be reconnected)"""
        # Build system instruction with current state
        system_instruction = self._build_system_instruction()
        
        config = {
            "response_modalities": ["AUDIO"] if self.use_audio else ["TEXT"],
            "system_instruction": {"parts": [{"text": system_instruction}]}
        }
        
        self.logger.info("Establishing connection to Gemini Live API...")
        
        try:
            async with (
                self.client.aio.live.connect(model=MODEL, config=config) as session,
                asyncio.TaskGroup() as tg
            ):
                self.session = session
                self.connection_active = True
                self.reconnect_attempts = 0  # Reset on successful connection
                self.logger.info("Connected successfully!")
                
                # Start tasks
                recv_task = tg.create_task(self._receive_loop())
                send_task = tg.create_task(self._send_loop())
                
                if self.use_audio:
                    audio_input_task = tg.create_task(self._audio_input_loop())
                else:
                    text_input_task = tg.create_task(self._text_input_loop())
                
                # Wait for completion (runs indefinitely)
                await send_task
                
        except CancelledError:
            self.logger.info("Session cancelled")
            raise
        except Exception as e:
            self.connection_active = False
            # Re-raise to trigger reconnection
            raise
    
    def _build_system_instruction(self) -> str:
        """Build system instruction with memories"""
        # Use new memory system for context
        memory_context = self.memory_system.get_context_for_model()
        
        instruction = f"""You are an always-on voice assistant. You have access to long-term memories about the user and recent conversation context.

{memory_context}

When the user shares important information about themselves, remember it naturally as part of our conversation. Your memories will be automatically updated in the background."""
        return instruction
    
    async def _text_input_loop(self):
        """Text input loop (fallback when no audio)"""
        self.logger.info("Text mode active. Type your messages (or 'quit' to exit)")
        while self.is_running and self.connection_active:
            try:
                text = await asyncio.to_thread(input, "\nYou: ")
                if text.lower() in ['quit', 'exit', 'q']:
                    self.is_running = False
                    break
                
                if text.strip() and self.connection_active:
                    await self.send_queue.put({"text": text})
                    self.interaction_logger.log_interaction("user", text, {"mode": "text"})
                    
                    # Add to memory system (no audio file for text input)
                    self.memory_system.add_interaction("user", text, {"mode": "text"}, audio_file_path=None)
                    
            except CancelledError:
                # Task was cancelled (e.g., during reconnection), this is expected
                self.logger.info("Text input loop cancelled")
            except Exception as e:
                self.logger.error(f"Error in text input: {e}")
                break
    
    async def _audio_input_loop(self):
        """Audio input loop - records and sends audio to API"""
        self.logger.info("Voice mode active. Speak to interact (Ctrl+C to exit)")
        
        # Start recording if not already recording
        if not self.audio_recorder.is_recording:
            self.audio_recorder.start_recording()
        
        # Note: This is a simplified version. In production, you'd want:
        # - Voice Activity Detection (VAD)
        # - Push-to-talk or wake word
        # - Better chunking logic
        
        try:
            while self.is_running and self.connection_active:
                # Collect audio in chunks
                audio_chunks = []
                
                # Simple: record for 3 seconds at a time
                # TODO: Implement VAD for better experience
                self.logger.info("Listening... (speak now)")
                
                start_time = time.time()
                while time.time() - start_time < 3.0:
                    chunk = self.audio_recorder.read_chunk()
                    audio_chunks.append(chunk)
                    await asyncio.sleep(0.01)
                
                # Combine chunks
                audio_data = b''.join(audio_chunks)
                
                # Save recording
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"user_input_{timestamp}.wav"
                filepath = self.audio_recorder.save_audio_to_file(audio_data, filename)
                
                # Send to API - only if connection is active
                if self.connection_active:
                    await self.send_queue.put({
                        "audio": audio_data,
                        "filepath": str(filepath)
                    })
                    
                    self.interaction_logger.log_interaction(
                        "user",
                        f"[Audio input: {filename}]",
                        {"mode": "audio", "filepath": str(filepath)}
                    )
                    
                    # Add user audio to memory system with file path
                    self.memory_system.add_interaction(
                        "user",
                        f"[Audio input: {filename}]",
                        {"mode": "audio", "filepath": str(filepath)},
                        audio_file_path=str(filepath)
                    )
                
        except CancelledError:
            # Task was cancelled (e.g., during reconnection), this is expected
            self.logger.info("Audio input loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in audio input loop: {e}")
            traceback.print_exc()
    
    async def _send_loop(self):
        """Send loop - sends messages to API"""
        while self.is_running and self.connection_active:
            try:
                msg = await asyncio.wait_for(self.send_queue.get(), timeout=1.0)
                
                if "text" in msg:
                    await self.session.send_client_content(
                        turns={"role": "user", "parts": [{"text": msg["text"]}]},
                        turn_complete=True
                    )
                    self.logger.info(f"Sent text: {msg['text'][:50]}...")
                    
                elif "audio" in msg:
                    # Send audio data using send_realtime_input (correct method for audio)
                    await self.session.send_realtime_input(
                        audio=types.Blob(
                            data=msg["audio"], 
                            mime_type=f"audio/pcm;rate={RATE}"
                        )
                    )
                    self.logger.info(f"Sent audio: {msg.get('filepath', 'unknown')}")
                
                self.turn_count += 1
                
                # Check if we need to summarize and clear context
                if self.turn_count >= MAX_TURNS_BEFORE_SUMMARIZE:
                    await self._summarize_and_clear_context()
                    
            except asyncio.TimeoutError:
                # No message to send, continue
                continue
            except Exception as e:
                # Connection closed or other error - mark connection as inactive and re-raise
                self.connection_active = False
                self.logger.error(f"Error in send loop: {e}")
                traceback.print_exc()
                raise
    
    async def _receive_loop(self):
        """Receive loop - receives responses from API"""
        audio_index = 0
        
        try:
            while self.is_running and self.connection_active:
                # Receive model responses
                turn = self.session.receive()
                
                text_response = []
                audio_chunks = []
                
                # Start audio playback if available
                if self.use_audio and self.audio_player:
                    self.audio_player.start_playback()
                
                async for response in turn:
                    # Handle text response
                    if response.text:
                        text_response.append(response.text)
                        if not self.use_audio:
                            print(response.text, end='', flush=True)
                    
                    # Handle audio response
                    if response.data:
                        audio_chunks.append(response.data)
                        if self.use_audio and self.audio_player:
                            self.audio_player.play_chunk(response.data)
                
                # Stop playback
                if self.use_audio and self.audio_player:
                    self.audio_player.stop_playback()
                
                # Log the interaction
                full_text = ''.join(text_response)
                
                # Save audio if we got any
                audio_path = None
                if audio_chunks:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    audio_filename = f"assistant_response_{timestamp}.wav"
                    audio_path = Path(AUDIO_RECORDINGS_DIR) / audio_filename
                    
                    with wave_file(str(audio_path)) as wf:
                        for chunk in audio_chunks:
                            wf.writeframes(chunk)
                    
                    self.interaction_logger.log_interaction(
                        "assistant",
                        full_text if full_text else "[Audio response]",
                        {"mode": "audio", "filepath": str(audio_path)}
                    )
                    
                    self.logger.info(f"Assistant responded (audio saved to {audio_filename})")
                else:
                    self.interaction_logger.log_interaction(
                        "assistant",
                        full_text,
                        {"mode": "text"}
                    )
                    self.logger.info(f"Assistant responded: {full_text[:50]}...")
                
                # Add assistant response to memory system (with audio file if available)
                if full_text.strip() or audio_path:
                    content = full_text if full_text else "[Audio response]"
                    self.memory_system.add_interaction(
                        "assistant",
                        content,
                        {"mode": "audio" if audio_chunks else "text"},
                        audio_file_path=str(audio_path) if audio_path else None
                    )
                    
                    # Process memories in background (non-blocking)
                    self._schedule_memory_processing()
                
                if not self.use_audio:
                    print()  # New line after response
                    
        except CancelledError:
            pass
        except Exception as e:
            # Connection closed or other error - mark connection as inactive and re-raise
            self.connection_active = False
            self.logger.error(f"Error in receive loop: {e}")
            traceback.print_exc()
            raise
    
    def _schedule_memory_processing(self):
        """Schedule memory processing as a background task (non-blocking)"""
        # Cancel any existing processing task
        if self.memory_processing_task and not self.memory_processing_task.done():
            self.logger.debug("Memory processing already in progress, skipping")
            return
        
        # Create new background task
        self.memory_processing_task = asyncio.create_task(self._process_memories_background())
    
    async def _process_memories_background(self):
        """Process memories in the background without blocking responses"""
        try:
            self.logger.debug("Starting background memory processing...")
            processed = await self.memory_system.process_if_ready()
            
            if processed:
                stats = self.memory_system.get_statistics()
                self.logger.info(f"Background memory processing complete. Stats: {stats}")
        except Exception as e:
            self.logger.error(f"Error in background memory processing: {e}")
            # Don't crash the assistant if memory processing fails
    
    async def _summarize_and_clear_context(self):
        """Summarize conversation and clear context to manage window size"""
        self.logger.info("Context window getting full, processing memories...")
        
        # Force process all pending memories to long-term storage (blocking this time)
        await self.memory_system.process_if_ready(force=True)
        
        # Get memory statistics
        stats = self.memory_system.get_statistics()
        self.logger.info(f"Memory stats: {stats}")
        
        # Reset turn counter
        self.turn_count = 0
        
        # Optionally restart session with fresh context
        # (This would require closing and reopening the session)
        # For now, just note that memories have been processed
        self.logger.info("Context processed, memories updated in long-term storage")
    
    async def _cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        self.is_running = False
        
        if self.use_audio:
            if self.audio_recorder:
                self.audio_recorder.cleanup()
            if self.audio_player:
                self.audio_player.cleanup()
        
        # Cancel any ongoing background memory processing
        if self.memory_processing_task and not self.memory_processing_task.done():
            self.logger.info("Waiting for background memory processing to complete...")
            try:
                await asyncio.wait_for(self.memory_processing_task, timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.warning("Background memory processing timed out")
                self.memory_processing_task.cancel()
        
        # Save memories one last time (old system)
        self.memory_manager.save_memories()
        
        # Process and save any remaining short-term memories
        try:
            await self.memory_system.process_if_ready(force=True)
            self.logger.info("Final memory processing complete")
        except Exception as e:
            self.logger.error(f"Error in final memory processing: {e}")
        
        self.logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    assistant = AlwaysOnAssistant()
    
    try:
        await assistant.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

