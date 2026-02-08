import os
import time
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json
import base64
import requests

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

TELNYX_API_KEY = os.getenv("TELNYX_API_KEY", "your_api_key")
TELNYX_PHONE_NUMBER = os.getenv("TELNYX_PHONE_NUMBER", "+1234567890")
TELNYX_CONNECTION_ID = os.getenv("TELNYX_CONNECTION_ID", "your_connection_id")
TELNYX_API_BASE = "https://api.telnyx.com/v2"

def get_telnyx_headers():
    """Helper function to get Telnyx API headers"""
    return {
        "Authorization": f"Bearer {TELNYX_API_KEY}",
        "Content-Type": "application/json"
    }

@app.post("/make-call")
async def make_call(request: Request):
    """
    Initiate an outbound call to a specified number.
    Body: {"to": "+15551234567"}
    """
    body = await request.json()
    to_number = body.get("to")
    
    if not to_number:
        return {"error": "Missing 'to' phone number"}
    
    base_url = os.getenv("BASE_URL", "https://your-server-url.com")
    stream_url = f"wss://{base_url.replace('https://', '').replace('http://', '')}/media-stream"
    
    try:
        # Make direct API call to create call WITH streaming
        response = requests.post(
            f"{TELNYX_API_BASE}/calls",
            headers=get_telnyx_headers(),
            json={
                "connection_id": TELNYX_CONNECTION_ID,
                "to": to_number,
                "from": TELNYX_PHONE_NUMBER,
                "webhook_url": f"{base_url}/webhooks/answer",
                "webhook_url_method": "POST",
                "stream_url": stream_url,
                "stream_track": "inbound_track"
            }
        )
        
        response.raise_for_status()
        call_data = response.json()
        
        print(f"Call initiated with stream_url: {stream_url}")
        
        return {
            "success": True,
            "call_control_id": call_data.get("data", {}).get("call_control_id"),
            "call_leg_id": call_data.get("data", {}).get("call_leg_id"),
            "to": to_number,
            "stream_url": stream_url
        }
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "details": e.response.text if hasattr(e, 'response') else None}

# Handle ALL Telnyx webhooks (they're going to root)
@app.post("/")
async def root_post(request: Request):
    """Handle Telnyx webhooks - main handler"""
    return await handle_webhook(request)

@app.post("/webhooks/answer")
async def webhook_answer(request: Request):
    """Handle Telnyx webhooks - alternative endpoint"""
    return await handle_webhook(request)

async def handle_webhook(request: Request):
    """
    Unified webhook handler for all Telnyx call events.
    """
    body = await request.json()
    
    event_type = body.get("data", {}).get("event_type")
    payload = body.get("data", {}).get("payload", {})
    call_control_id = payload.get("call_control_id")
    
    print(f"\n{'='*60}")
    print(f"Webhook: {event_type}")
    print(f"Call Control ID: {call_control_id}")
    print(f"{'='*60}\n")
    
    try:
        if event_type == "call.initiated":
            print("✅ Call initiated - waiting for answer...")
            # Don't answer outbound calls - they auto-connect
        
        elif event_type == "call.answered":
            print("✅ Call answered!")
            
            # Make sure streaming is started
            base_url = os.getenv("BASE_URL", "https://your-server-url.com")
            stream_url = f"wss://{base_url.replace('https://', '').replace('http://', '')}/media-stream"
            
            stream_response = requests.post(
                f"{TELNYX_API_BASE}/calls/{call_control_id}/actions/streaming_start",
                headers=get_telnyx_headers(),
                json={
                    "stream_url": stream_url,
                    "stream_track": "inbound_track"
                }
            )
            print(f"✅ Streaming command sent: {stream_response.status_code}")
            if stream_response.status_code != 200:
                print(f"Streaming error: {stream_response.text}")
            
        elif event_type == "call.hangup":
            print(f"📞 Call ended - hangup_cause: {payload.get('hangup_cause')}")
            
        elif event_type == "call.streaming.started":
            print("🎙️ Audio streaming STARTED")
            
        elif event_type == "call.streaming.stopped":
            print("🎙️ Audio streaming STOPPED")
            
    except Exception as e:
        print(f"❌ Error in webhook handler: {e}")
        import traceback
        traceback.print_exc()
    
    return JSONResponse(content={"status": "ok"})

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for receiving Telnyx media streams.
    """
    await websocket.accept()
    print("\n" + "="*60)
    print("🔌 WebSocket connection ESTABLISHED")
    print("="*60 + "\n")
    
    call_audio = []
    call_id = None
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                event = message.get("event")
                
                if event == "connected":
                    print("✅ Media stream connected")
                    print(f"Message: {json.dumps(message, indent=2)}")
                    
                elif event == "start":
                    # Extract call ID from various possible fields
                    call_id = (
                        message.get("call_control_id") or 
                        message.get("call_id") or
                        message.get("streamSid") or
                        message.get("start", {}).get("callSid") or
                        message.get("start", {}).get("streamSid") or
                        message.get("start", {}).get("call_control_id")
                    )
                    
                    print(f"🎬 Stream STARTED")
                    print(f"Call ID: {call_id}")
                    print(f"Full start message: {json.dumps(message, indent=2)}")
                    call_audio = []
                    
                elif event == "media":
                    payload = message.get("media", {}).get("payload")
                    if payload:
                        call_audio.append(payload)
                        if len(call_audio) == 1:
                            print("🎤 First audio chunk received!")
                        if len(call_audio) % 100 == 0:
                            print(f"📊 Received {len(call_audio)} audio chunks")
                    
                elif event == "stop":
                    print(f"\n🛑 Stream STOPPED")
                    print(f"Call ID: {call_id}")
                    print(f"Total chunks: {len(call_audio)}")
                    
                    if call_audio:
                        # Use fallback ID if needed
                        save_id = call_id or f"call_{int(time.time() * 1000)}"
                        save_audio(call_audio, save_id)
                    else:
                        print("⚠️ No audio chunks to save!")
                    break
                    
            except WebSocketDisconnect:
                print("WebSocket disconnected by client")
                break
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing WebSocket...")
        try:
            await websocket.close()
        except:
            pass

def save_audio(audio_payloads: list, call_id: str):
    """
    Save the collected audio chunks to a file.
    """
    if not audio_payloads:
        print("⚠️ No audio to save")
        return
    
    try:
        print(f"\n{'='*60}")
        print(f"💾 SAVING AUDIO")
        print(f"{'='*60}")
        
        audio_data = b"".join([base64.b64decode(payload) for payload in audio_payloads])
        
        os.makedirs("recordings", exist_ok=True)
        
        # Save raw file
        raw_filename = f"recordings/{call_id}.raw"
        with open(raw_filename, "wb") as f:
            f.write(audio_data)
        
        print(f"✅ Raw audio saved: {raw_filename}")
        print(f"   Size: {len(audio_data)} bytes ({len(audio_data)/1024:.2f} KB)")
        print(f"   Chunks: {len(audio_payloads)}")
        
        # Try to convert to WAV
        try:
            import subprocess
            wav_filename = f"recordings/{call_id}.wav"
            
            result = subprocess.run([
                "ffmpeg", "-y", 
                "-f", "mulaw",      # Format: mu-law (PCMU codec)
                "-ar", "8000",      # Sample rate: 8kHz
                "-ac", "1",         # Channels: mono
                "-i", raw_filename,  # Input file
                wav_filename        # Output file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                wav_size = os.path.getsize(wav_filename)
                print(f"✅ WAV converted: {wav_filename}")
                print(f"   Size: {wav_size} bytes ({wav_size/1024:.2f} KB)")
            else:
                print(f"⚠️ FFmpeg error: {result.stderr}")
                
        except FileNotFoundError:
            print("⚠️ FFmpeg not installed - install with: brew install ffmpeg")
        except Exception as e:
            print(f"⚠️ WAV conversion failed: {e}")
        
        print(f"{'='*60}\n")
            
    except Exception as e:
        print(f"❌ Error saving audio: {e}")
        import traceback
        traceback.print_exc()

@app.get("/")
async def root():
    return {"message": "Cabby is running", "recordings_endpoint": "/recordings"}

@app.get("/recordings")
async def list_recordings():
    """List all recordings"""
    try:
        recordings = []
        if os.path.exists("recordings"):
            for file in os.listdir("recordings"):
                filepath = os.path.join("recordings", file)
                size = os.path.getsize(filepath)
                recordings.append({
                    "filename": file,
                    "size_bytes": size,
                    "size_kb": f"{size/1024:.2f} KB",
                    "path": filepath
                })
        return {
            "recordings": recordings, 
            "count": len(recordings),
            "directory": os.path.abspath("recordings")
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=1604,
        reload=True,
        log_level="info",
    )