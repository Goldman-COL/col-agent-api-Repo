import os
import logging
from dotenv import load_dotenv
import uuid
import traceback
import tempfile
import subprocess
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict

load_dotenv()

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("video-verification")

# class NNPackFilter(logging.Filter):
#     def filter(self, record):
#         if isinstance(record.msg, str):
#             return not ("NNPACK" in record.msg and "Unsupported hardware" in record.msg)
#         return True

# logging.getLogger().addFilter(NNPackFilter())

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from app.generate_challenge_phrase import generate_challenge_phrase
# from app.liveness_cnn import calculate_liveness_score
# from app.face_verification import calculate_face_similarity
from app.whisper import calculate_speech_score
from app.whisper import transcribe_audio
from app.infra.azure_blob import upload_blob, download_blob

app = FastAPI(title="Video Verification API")

# Mount the static directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for challenge phrases
# In production, use Redis or another caching solution
CHALLENGE_PHRASES: Dict[str, str] = {}

async def process_verification(video_file: UploadFile, user_id: str):
    logger.info(f"Starting verification process for user {user_id}")
    expected_phrase = CHALLENGE_PHRASES.get(user_id, "")
    if not expected_phrase:
        logger.warning(f"No challenge phrase found for user {user_id}")
        return {
            "ok": False,
            "speech": 0.0,
            "face": 0.0,
            "liveness": 0.0,
            "error": "No challenge phrase found for this user"
        }
    video_blob_url = None
    try:
        # Read video bytes directly from UploadFile
        video_bytes = await video_file.read()
        video_blob_name = f"{user_id}_{uuid.uuid4().hex}.webm"
        video_blob_url = upload_blob(
            os.environ["AZURE_RECORDINGS_CONTAINER"],
            video_blob_name,
            video_bytes,
            "video/webm"
        )
        
        # Create temporary files for video and audio
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as video_temp:
            video_temp.write(video_bytes)
            video_temp_path = video_temp.name

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_temp:
            audio_temp_path = audio_temp.name

        try:
            # Use ffmpeg directly to extract audio
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_temp_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                audio_temp_path
            ]
            
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise Exception(f"Failed to extract audio: {stderr.decode()}")

            # Read the audio file
            with open(audio_temp_path, 'rb') as audio_file:
                # Create a new UploadFile for the audio
                audio_upload = UploadFile(
                    filename="audio.wav",
                    file=audio_file,
                    content_type="audio/wav"
                )
                
                # Transcribe the audio
                transcription = await transcribe_audio(audio_upload)
                logger.info(f"Transcription: {transcription}")
                
                # Calculate speech score
                speech_score = calculate_speech_score(expected_phrase, transcription)
                logger.info(f"Speech score: {speech_score}, expected: '{expected_phrase}', got: '{transcription}'")
                
                # For now, return response with speech verification only
                speech_passed = speech_score >= 0.60
                return {
                    "ok": bool(speech_passed),
                    "speech": float(speech_score),
                    "face": 1.0,    # Placeholder
                    "liveness": 1.0, # Placeholder
                    "video_url": video_blob_url,
                    "transcription": transcription  # Added for debugging
                }
        finally:
            # Clean up temporary files
            try:
                os.unlink(video_temp_path)
                os.unlink(audio_temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")

    except Exception as e:
        logger.error(f"Error processing verification: {e}\n{traceback.format_exc()}")
        return {
            "ok": False,
            "speech": 0.0,
            "face": 0.0,
            "liveness": 0.0,
            "error": str(e),
            "video_url": video_blob_url
        }

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/challenge")
async def get_challenge(user_id: str):
    phrase = generate_challenge_phrase()
    CHALLENGE_PHRASES[user_id] = phrase
    logger.info(f"Generated challenge phrase for user {user_id}: {phrase}")
    return {"phrase": phrase}

@app.post("/verify")
async def verify_video(
        file: UploadFile = File(...),
        user_id: str = Form(...)
):
    logger.info(f"Received verification request for user {user_id}")
    result = await process_verification(file, user_id)
    return result

@app.post("/upload_profile")
async def upload_profile_photo(
        file: UploadFile = File(...),
        user_id: str = Form(...)
):
    logger.info(f"Received profile photo upload for user {user_id}")

    content_type = file.content_type
    if not content_type.startswith("image/"):
        return {"ok": False, "error": "Uploaded file is not an image"}

    extension = "jpg"
    if content_type == "image/png":
        extension = "png"
    elif content_type == "image/jpeg":
        extension = "jpg"

    blob_name = f"{user_id}.{extension}"
    try:
        content = await file.read()
        url = upload_blob(
            os.environ["AZURE_PROFILE_CONTAINER"],
            blob_name,
            content,
            content_type
        )
        logger.info(f"Profile photo saved for user {user_id} at {url}")
        return {"ok": True, "profile_url": url}
    except Exception as e:
        logger.error(f"Error saving profile photo: {e}")
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info",
                access_log=True, use_colors=True)
