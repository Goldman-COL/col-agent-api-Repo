import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import time
import uuid

load_dotenv()

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("video-verification")

class NNPackFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            return not ("NNPACK" in record.msg and "Unsupported hardware" in record.msg)
        return True

logging.getLogger().addFilter(NNPackFilter())

from fastapi.staticfiles import StaticFiles
import tempfile
import string
import shutil
import traceback
import warnings
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import random
from typing import List, Optional, Dict
from fastapi.responses import FileResponse


from app.generate_challenge_phrase import generate_challenge_phrase
#from app.liveness_cnn import calculate_liveness_score
#from app.face_verification import calculate_face_similarity
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

def safe_delete(filepath, retries=5, delay=0.2):
    """Attempt to delete a file, retrying if Windows locks it temporarily."""
    for _ in range(retries):
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
            return
        except PermissionError:
            time.sleep(delay)
    # Final attempt, let exception propagate if it fails
    if os.path.exists(filepath):
        os.unlink(filepath)

async def extract_audio_from_video(video_file: UploadFile) -> str:
    temp_video_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
            temp_video_path = temp_video.name
            content = await video_file.read()
            temp_video.write(content)
        return temp_video_path
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise e

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
    temp_file_path = None
    video_blob_url = None
    try:
        temp_file_path = await extract_audio_from_video(video_file)
        # Upload the video to Azure Blob Storage
        with open(temp_file_path, "rb") as f:
            video_bytes = f.read()
        video_blob_name = f"{user_id}_{uuid.uuid4().hex}.webm"
        video_blob_url = upload_blob(
            os.environ["AZURE_RECORDINGS_CONTAINER"],
            video_blob_name,
            video_bytes,
            "video/webm"
        )
        # Download profile image from Azure Blob Storage
        profile_blob_name = f"{user_id}.jpg"  # or .png if you want to check both
        try:
            profile_image_bytes = download_blob(os.environ["AZURE_PROFILE_CONTAINER"], profile_blob_name)
        except Exception as e:
            logger.error(f"Could not download profile image for user {user_id}: {e}")
            profile_image_bytes = None
        await video_file.seek(0)
        transcription = await transcribe_audio(video_file)
        logger.info(f"Transcription: {transcription}")
        speech_score = calculate_speech_score(expected_phrase, transcription)
        logger.info(f"Speech score: {speech_score}, expected: '{expected_phrase}', got: '{transcription}'")
        face_score = await calculate_face_similarity(temp_file_path, user_id, profile_image_bytes)
        logger.info(f"Face score: {face_score}")
        liveness_score = await calculate_liveness_score(temp_file_path)
        logger.info(f"Liveness score: {liveness_score}")
        speech_passed = speech_score >= 0.60
        face_passed = face_score >= 0.80
        liveness_passed = liveness_score >= 0.50
        passed = speech_passed and face_passed and liveness_passed
        CHALLENGE_PHRASES.pop(user_id, None)
        logger.info(f"Verification for user {user_id} completed: passed={passed}, "
                    f"speech={speech_score}, "
                    f"face={face_score}, "
                    f"liveness={liveness_score}")
        return {
            "ok": bool(passed),
            "speech": float(speech_score),
            "face": float(face_score),
            "liveness": float(liveness_score),
            "video_url": video_blob_url
        }
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

@app.get("/", response_class=FileResponse)
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
