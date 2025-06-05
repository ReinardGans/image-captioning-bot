import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import requests
import tempfile
import base64
import uuid
from pathlib import Path

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.responses import FileResponse

app = FastAPI()


# Get absolute path to project root
PROJECT_ROOT = Path(__file__).parent.resolve()
# Serve static files (Vue app) at /static
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")

# Serve index.html for root and unknown routes (SPA fallback)
@app.get("/")
async def serve_vue_index():
    return FileResponse("frontend/index.html")

@app.get("/{full_path:path}")
async def serve_vue_spa(full_path: str):
    # Only serve index.html for GET requests that are not API endpoints
    if full_path.startswith("caption") or full_path.startswith("audio"):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse("frontend/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/caption")
def caption_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        # Get caption from OpenAI

        with open(tmp_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode()
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a witty, funny assistant who roasts the person in the image. Your captions should be humorous and cheeky, and mean-spirited or offensive."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe the co-worker of mine in this image in a witty manner."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]}
                ],
                max_tokens=100
            )
            caption = response.choices[0].message.content.strip()


        # Generate speech audio using ElevenLabs
        # Get ElevenLabs audio as base64 (do not save to disk)
        eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")
        if not eleven_api_key:
            raise Exception("ELEVENLABS_API_KEY not set in environment or .env file.")
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": eleven_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": caption,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        tts_response = requests.post(tts_url, headers=headers, json=payload)
        if tts_response.status_code != 200:
            raise Exception(f"ElevenLabs TTS failed: {tts_response.text}")
        audio_b64 = base64.b64encode(tts_response.content).decode()

        # Clean up image file
        os.remove(tmp_path)

        # Return JSON with caption and audio as base64 data URL
        audio_url = f"data:audio/mpeg;base64,{audio_b64}"
        return JSONResponse({"caption": caption, "audio_url": audio_url})
    except Exception as e:
        import traceback
        print("Error in /caption endpoint:", traceback.format_exc())
        return HTMLResponse(content=f"<h2>Internal Server Error</h2><pre>{str(e)}</pre>", status_code=500)

@app.get("/audio/{filename}")
def get_audio(filename: str):
    audio_path = PROJECT_ROOT / "audio" / filename
    print(f"[AUDIO SERVE] Looking for: {audio_path} Exists: {audio_path.exists()}")
    if not audio_path.exists():
        return JSONResponse({"detail": f"Audio file not found: {audio_path}"}, status_code=404)
    return FileResponse(str(audio_path), media_type="audio/mpeg", filename="caption.mp3")
