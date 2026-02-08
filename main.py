from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import websockets
from websockets.legacy.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed
import json
import base64
import os
from typing import Optional
import numpy as np
from scipy import signal
from dotenv import load_dotenv
import tempfile
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Telnyx + ElevenLabs Voice Agent")

# Store active call sessions
active_calls = {}

# Create temp directory for audio files
TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "telnyx_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# Configuration
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
TELNYX_APP_ID = os.getenv("TELNYX_APP_ID")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")

# Validate required environment variables
if not TELNYX_API_KEY:
    print("❌ ERROR: TELNYX_API_KEY not set in .env file")
if not TELNYX_APP_ID:
    print("❌ ERROR: TELNYX_APP_ID not set in .env file")
if not ELEVENLABS_API_KEY:
    print("❌ ERROR: ELEVENLABS_API_KEY not set in .env file")
if not ELEVENLABS_AGENT_ID:
    print("❌ ERROR: ELEVENLABS_AGENT_ID not set in .env file")


class CallSession:
    """Manages a single call session bridging Telnyx and ElevenLabs"""
    
    def __init__(self, call_control_id: str):
        self.call_control_id = call_control_id
        self.telnyx_ws: Optional[WebSocket] = None
        self.elevenlabs_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_active = True
        self.streaming_ready = False  # Wait for streaming.started before sending audio
    
    @staticmethod
    def mulaw_to_pcm(mulaw_data: bytes) -> np.ndarray:
        """Convert μ-law encoded audio to linear PCM16"""
        # μ-law decoding lookup table (simplified)
        mulaw_array = np.frombuffer(mulaw_data, dtype=np.uint8)
        # Standard μ-law to linear conversion
        mulaw_array = mulaw_array.astype(np.int16)
        sign = (mulaw_array & 0x80) >> 7
        exponent = (mulaw_array & 0x70) >> 4
        mantissa = mulaw_array & 0x0F
        
        # Convert to linear
        linear = ((mantissa << 3) + 0x84) << exponent
        linear = np.where(sign == 0, linear, -linear)
        return linear.astype(np.int16)
    
    @staticmethod
    def pcm_to_mulaw(pcm_data: np.ndarray) -> bytes:
        """Convert linear PCM16 to μ-law encoding"""
        # Ensure input is int16
        pcm_data = pcm_data.astype(np.int16)
        
        # Get sign
        sign = (pcm_data < 0).astype(np.uint8) << 7
        pcm_data = np.abs(pcm_data)
        
        # Add bias
        pcm_data = np.clip(pcm_data + 0x84, 0, 0x7FFF)
        
        # Find exponent and mantissa
        exponent = np.zeros_like(pcm_data, dtype=np.uint8)
        for i in range(7, -1, -1):
            mask = pcm_data >= (0x84 << i)
            exponent = np.where(mask & (exponent == 0), i, exponent)
        
        mantissa = ((pcm_data >> (exponent + 3)) & 0x0F).astype(np.uint8)
        
        # Combine
        mulaw = sign | (exponent << 4) | mantissa
        mulaw = ~mulaw & 0xFF  # Invert bits
        
        return mulaw.astype(np.uint8).tobytes()
    
    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio from orig_sr to target_sr using scipy"""
        if orig_sr == target_sr:
            return audio
        
        # Calculate number of samples in target sample rate
        num_samples = int(len(audio) * target_sr / orig_sr)
        
        # Use scipy's resample
        resampled = signal.resample(audio, num_samples)
        
        return resampled.astype(np.int16)
        
    async def connect_elevenlabs(self):
        """Connect to ElevenLabs Conversational AI WebSocket"""
        url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={ELEVENLABS_AGENT_ID}"
        
        # Use legacy websockets client which supports extra_headers
        self.elevenlabs_ws = await ws_connect(
            url,
            extra_headers={"xi-api-key": ELEVENLABS_API_KEY}
        )
        print(f"✅ Connected to ElevenLabs for call {self.call_control_id}")
        
        # Wait for initial metadata from ElevenLabs
        initial_msg = await self.elevenlabs_ws.recv()
        print(f"📩 Initial message from ElevenLabs: {initial_msg[:200]}...")
        
        # Initialize the conversation with μ-law 8000 Hz for both input AND output
        print(f"📣 Initializing ElevenLabs conversation with μ-law 8kHz (input + output)...")
        init_message = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {
                "agent": {
                    "prompt": {
                        "prompt": ""  # Use agent's default prompt
                    }
                },
                "tts": {
                    "model_id": "eleven_turbo_v2_5"
                }
            },
            "audio_interface": {
                "input": {
                    "sample_rate": 8000,
                    "encoding": "ulaw"
                },
                "output": {
                    "sample_rate": 8000,
                    "encoding": "ulaw"
                }
            }
        }
        
        await self.elevenlabs_ws.send(json.dumps(init_message))
        print(f"✅ Conversation initialization sent (μ-law 8kHz input + output)")
        
        # Note: ElevenLabs may send audio immediately after init
        # We'll handle the metadata in the forwarding loop
        print(f"⚠️  Note: ElevenLabs is sending PCM16 output (will convert to μ-law)")
        
    async def bridge_audio(self):
        """Bridge audio between Telnyx and ElevenLabs"""
        try:
            # Start two concurrent tasks: Telnyx->ElevenLabs and ElevenLabs->Telnyx
            await asyncio.gather(
                self.forward_telnyx_to_elevenlabs(),
                self.forward_elevenlabs_to_telnyx()
            )
        except Exception as e:
            print(f"❌ Bridge error: {e}")
        finally:
            await self.cleanup()
    
    async def forward_telnyx_to_elevenlabs(self):
        """Forward audio from Telnyx (caller) to ElevenLabs (AI agent)"""
        print(f"🎤 Starting Telnyx->ElevenLabs audio forwarding (μ-law passthrough)...")
        while self.is_active:
            try:
                # Receive audio from Telnyx WebSocket
                data = await self.telnyx_ws.receive_text()
                message = json.loads(data)
                
                event_type = message.get("event")
                
                if event_type == "media":
                    # Extract audio payload (base64 encoded μ-law)
                    audio_b64 = message.get("media", {}).get("payload", "")
                    if audio_b64:
                        # Send directly to ElevenLabs - no conversion needed!
                        # Both are μ-law 8000 Hz
                        message_to_send = {
                            "user_audio_chunk": audio_b64  # Send base64 μ-law directly
                        }
                        
                        await self.elevenlabs_ws.send(json.dumps(message_to_send))
                
                elif event_type == "start":
                    print(f"🎬 Telnyx stream started")
                        
                elif event_type == "stop":
                    print("📞 Call ended by Telnyx")
                    self.is_active = False
                    break
                    
            except Exception as e:
                print(f"❌ Error forwarding to ElevenLabs: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"🛑 Stopped Telnyx->ElevenLabs forwarding")
    
    async def forward_elevenlabs_to_telnyx(self):
        """Forward audio from ElevenLabs (AI agent) back to Telnyx (caller)"""
        print(f"🔊 Starting ElevenLabs->Telnyx audio forwarding (μ-law passthrough)...")
        while self.is_active:
            try:
                # Receive from ElevenLabs
                response = await self.elevenlabs_ws.recv()
                data = json.loads(response)
                
                message_type = data.get("type")
                print(f"📩 ElevenLabs message: {message_type or list(data.keys())[:3]}")
                
                # Handle different ElevenLabs message types
                message_type = data.get("type")
                
                # Check for audio in audio_event
                if "audio_event" in data:
                    audio_event = data["audio_event"]
                    audio_b64 = audio_event.get("audio_base_64", "")
                    
                    if audio_b64:
                        print(f"🤖 AI speaking... (received {len(audio_b64)} chars of base64)")
                        
                        # Decode from base64
                        audio_pcm = base64.b64decode(audio_b64)
                        
                        # Convert bytes to numpy array (PCM16)
                        audio_array = np.frombuffer(audio_pcm, dtype=np.int16)
                        
                        # Resample from 16kHz to 8kHz
                        audio_array_8k = self.resample_audio(audio_array, 16000, 8000)
                        
                        # Convert PCM16 to μ-law
                        audio_mulaw = self.pcm_to_mulaw(audio_array_8k)
                        
                        # Save audio to temporary file
                        import time
                        filename = f"audio_{self.call_control_id}_{int(time.time() * 1000)}.ul"
                        file_path = TEMP_AUDIO_DIR / filename
                        
                        with open(file_path, 'wb') as f:
                            f.write(audio_mulaw)
                        
                        # Get public URL
                        webhook_url = os.getenv("PUBLIC_WEBHOOK_URL", "https://your-server.com")
                        audio_url = f"{webhook_url}/audio/{filename}"
                        
                        print(f"💾 Saved audio to {filename}")
                        
                        # Use Telnyx Call Control API to play audio from URL
                        import httpx
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                f"https://api.telnyx.com/v2/calls/{self.call_control_id}/actions/playback_start",
                                headers={
                                    "Authorization": f"Bearer {TELNYX_API_KEY}",
                                    "Content-Type": "application/json"
                                },
                                json={
                                    "audio_url": audio_url,
                                    "overlay": False
                                }
                            )
                            if response.status_code == 200:
                                print(f"➡️  Playing AI audio from URL: {audio_url}")
                            else:
                                print(f"❌ Failed to play audio: {response.status_code} - {response.text}")
                
                # Legacy format check (just in case)
                elif "audio" in data:
                    audio_b64 = data["audio"]
                    print(f"🤖 AI speaking (legacy format)... (received {len(audio_b64)} chars)")
                    
                    # Same conversion as above
                    audio_pcm = base64.b64decode(audio_b64)
                    audio_array = np.frombuffer(audio_pcm, dtype=np.int16)
                    audio_array_8k = self.resample_audio(audio_array, 16000, 8000)
                    audio_mulaw = self.pcm_to_mulaw(audio_array_8k)
                    audio_b64_mulaw = base64.b64encode(audio_mulaw).decode('utf-8')
                    
                    await self.telnyx_ws.send_text(json.dumps({
                        "event": "media",
                        "stream_id": self.call_control_id,
                        "media": {
                            "track": "outbound",
                            "chunk": "1",
                            "timestamp": "0",
                            "payload": audio_b64_mulaw
                        }
                    }))
                    print(f"➡️  Sent AI audio to Telnyx")
                    
                elif message_type == "ping":
                    # Respond to ping with pong
                    await self.elevenlabs_ws.send(json.dumps({
                        "type": "pong",
                        "event_id": data.get("event_id", 0)
                    }))
                    
                elif message_type == "interruption":
                    print("🎤 User interrupted the agent")
                    
                elif message_type == "agent_response":
                    print(f"💭 Agent response: {data.get('agent_response', '')[:100]}...")
                    
                elif message_type == "user_transcript":
                    print(f"👤 User said: {data.get('user_transcript', '')}")
                    
                elif message_type == "agent_response_done":
                    print("✅ Agent finished speaking")
                
                elif message_type == "conversation_initiation_metadata":
                    print("🎬 ElevenLabs conversation started")
                    agent_output_audio_format = data.get("agent_output_audio_format", {})
                    print(f"   Audio format: {agent_output_audio_format}")
                    
            except ConnectionClosed:
                print("📞 ElevenLabs connection closed")
                self.is_active = False
                break
            except Exception as e:
                print(f"❌ Error forwarding to Telnyx: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"🛑 Stopped ElevenLabs->Telnyx forwarding")
    
    async def cleanup(self):
        """Clean up connections"""
        self.is_active = False
        if self.elevenlabs_ws:
            await self.elevenlabs_ws.close()
        print(f"🧹 Cleaned up call {self.call_control_id}")


@app.post("/make-call")
async def make_call(to_number: str, from_number: str):
    """
    Initiate an outbound call
    
    Args:
        to_number: Phone number to call (E.164 format, e.g., +1234567890)
        from_number: Your Telnyx number (E.164 format)
    """
    import httpx
    
    # Make call via Telnyx Call Control API
    url = "https://api.telnyx.com/v2/calls"
    headers = {
        "Authorization": f"Bearer {TELNYX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Your server's public URL for webhooks
    webhook_url = os.getenv("PUBLIC_WEBHOOK_URL", "https://your-server.com")
    
    payload = {
        "connection_id": TELNYX_APP_ID,
        "to": to_number,
        "from": from_number,
        "webhook_url": f"{webhook_url}/webhooks/telnyx",
        # Don't include stream_url here - we'll start streaming after call is answered
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        
    if response.status_code == 200:
        call_data = response.json()
        return {
            "status": "success",
            "call_control_id": call_data["data"]["call_control_id"],
            "message": "Call initiated"
        }
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)


@app.post("/webhooks/telnyx")
async def telnyx_webhook(request: dict):
    """
    Webhook endpoint for Telnyx call events
    Handles: call.initiated, call.answered, call.hangup, etc.
    """
    event_type = request.get("data", {}).get("event_type")
    call_control_id = request.get("data", {}).get("payload", {}).get("call_control_id")
    
    print(f"📞 Telnyx event: {event_type} for call {call_control_id}")
    
    if event_type == "call.answered":
        # Person answered! Start streaming audio
        print(f"✅ Call answered: {call_control_id}")
        
        # Get the public WebSocket URL
        webhook_url = os.getenv("PUBLIC_WEBHOOK_URL", "https://your-server.com")
        # Convert https:// to wss://
        ws_url = webhook_url.replace("https://", "wss://").replace("http://", "ws://")
        stream_url = f"{ws_url}/ws/telnyx/{call_control_id}"
        
        print(f"🔌 Starting stream to: {stream_url}")
        
        # Send command to start streaming audio to our WebSocket
        import httpx
        url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/streaming_start"
        headers = {
            "Authorization": f"Bearer {TELNYX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # This tells Telnyx to stream audio to our WebSocket endpoint
        payload = {
            "stream_url": stream_url,
            "stream_track": "both_tracks",  # both inbound and outbound audio
            "client_state": base64.b64encode(call_control_id.encode()).decode()  # Pass call_control_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                print(f"❌ Failed to start streaming: {response.status_code} - {response.text}")
            else:
                print(f"✅ Streaming started successfully")
            
    elif event_type == "streaming.failed":
        print(f"❌ Streaming failed for call {call_control_id}")
        # Log the error details
        error = request.get("data", {}).get("payload", {})
        print(f"   Error details: {error}")
    
    elif event_type == "streaming.started":
        print(f"✅ Streaming is ready for call {call_control_id}")
        # Mark streaming as ready
        if call_control_id in active_calls:
            active_calls[call_control_id].streaming_ready = True
            
    elif event_type == "call.hangup":
        print(f"📞 Call ended: {call_control_id}")
        if call_control_id in active_calls:
            active_calls[call_control_id].is_active = False
            del active_calls[call_control_id]
    
    return {"status": "received"}


@app.websocket("/ws/telnyx/{call_control_id}")
async def telnyx_websocket(websocket: WebSocket, call_control_id: str):
    """
    WebSocket endpoint for Telnyx audio streaming
    This is where the real-time audio flows
    """
    await websocket.accept()
    print(f"🔌 Telnyx WebSocket connected for call {call_control_id}")
    
    # Create call session
    session = CallSession(call_control_id)
    session.telnyx_ws = websocket
    active_calls[call_control_id] = session
    
    try:
        # Connect to ElevenLabs
        print(f"🔗 Connecting to ElevenLabs for call {call_control_id}...")
        await session.connect_elevenlabs()
        
        # Start the audio bridge
        print(f"🌉 Starting audio bridge for call {call_control_id}...")
        await session.bridge_audio()
        
    except Exception as e:
        print(f"❌ WebSocket error for call {call_control_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"🔚 WebSocket closed for call {call_control_id}")
        if call_control_id in active_calls:
            del active_calls[call_control_id]


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve temporary audio files for Telnyx playback"""
    file_path = TEMP_AUDIO_DIR / filename
    if file_path.exists():
        return FileResponse(
            file_path,
            media_type="audio/basic",  # μ-law audio
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    raise HTTPException(status_code=404, detail="Audio file not found")


@app.get("/")
async def root():
    return {
        "message": "Telnyx + ElevenLabs Voice Agent API",
        "active_calls": len(active_calls),
        "endpoints": {
            "make_call": "POST /make-call",
            "webhook": "POST /webhooks/telnyx",
            "websocket": "WS /ws/telnyx/{call_control_id}"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "active_calls": len(active_calls)}


if __name__ == "__main__":
    import uvicorn
    # Use import string format to enable reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)