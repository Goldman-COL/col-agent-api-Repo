import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import time
import uuid
from sqlalchemy import text

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
from app.liveness_cnn import calculate_liveness_score
from app.face_verification import FaceVerifier
from app.whisper import calculate_speech_score
from app.whisper import transcribe_audio
from app.infra.azure_blob import upload_blob, download_blob
from app.infra.db import ColSession, insert_kyc_record, insert_kyc_start_record, update_kyc_record_by_ssp_and_request_id, get_challenge_phrase
from fastapi import HTTPException
import cv2
import numpy as np
import requests

app = FastAPI(title="Video Verification API")

# Mount the static directory
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

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

face_verifier = FaceVerifier()

async def process_verification(video_file: UploadFile, ssp_id: int, profile_image_bytes: Optional[bytes], profile_image_url: Optional[str] = None, image_type: Optional[str] = None, request_id: Optional[str] = None):
    logger.info(f"Starting verification process for ssp_id {ssp_id}")
    expected_phrase = get_challenge_phrase(ssp_id, request_id)
    if not expected_phrase:
        logger.warning(f"No challenge phrase found for ssp_id {ssp_id}, request_id {request_id}")
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
        video_blob_name = f"{ssp_id}_{request_id}.webm"
        video_blob_url = upload_blob(
            os.environ["AZURE_RECORDINGS_CONTAINER"],
            video_blob_name,
            video_bytes,
            "video/webm"
        )
        await video_file.seek(0)
        transcription = await transcribe_audio(video_file)
        logger.info(f"Transcription: {transcription}")
        speech_score = calculate_speech_score(expected_phrase, transcription)
        logger.info(f"Speech score: {speech_score}, expected: '{expected_phrase}', got: '{transcription}'")
        face_score = await face_verifier.calculate_face_similarity(temp_file_path, ssp_id, profile_image_bytes)
        logger.info(f"Face score: {face_score}")
        liveness_score = await calculate_liveness_score(temp_file_path)
        logger.info(f"Liveness score: {liveness_score}")
        speech_passed = speech_score >= 0.60
        face_passed = face_score >= 0.80
        liveness_passed = liveness_score >= 0.50
        passed = speech_passed and face_passed and liveness_passed
        logger.info(f"Verification for ssp_id {ssp_id} completed: passed={passed}, "
                    f"speech={speech_score}, "
                    f"face={face_score}, "
                    f"liveness={liveness_score}")
        # Update KYC record if ssp_id and request_id are provided
        if ssp_id is not None and request_id is not None:
            try:
                update_kyc_record_by_ssp_and_request_id(
                    ssp_id=ssp_id,
                    request_id=request_id,
                    status="success" if passed else "failed",
                    video_url=video_blob_url,
                    image_url=profile_image_url or "",
                    image_type=image_type or ("url" if profile_image_url else "blob"),
                    speech_score=int(speech_score * 100),
                    face_score=int(face_score * 100),
                    liveness_score=int(liveness_score * 100)
                )
                logger.info(f"KYC record updated for ssp_id {ssp_id} and request_id {request_id}")
            except Exception as e:
                logger.error(f"Failed to update KYC record: {e}")
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

# @app.get("/", response_class=FileResponse)
# async def root():
#     return FileResponse("app/static/index.html")

@app.get("/challenge")
async def get_challenge(ssp_id: int, request_id: str):
    phrase = generate_challenge_phrase()
    try:
        update_kyc_record_by_ssp_and_request_id(
            ssp_id=ssp_id,
            request_id=request_id,
            status=None,
            video_url=None,
            image_url=None,
            image_type=None,
            speech_score=None,
            face_score=None,
            liveness_score=None,
            challenge_phrase=phrase
        )
        logger.info(f"Generated and stored challenge phrase for ssp_id {ssp_id}, request_id {request_id}: {phrase}")
    except Exception as e:
        logger.error(f"Failed to store challenge phrase: {e}")
        raise HTTPException(status_code=500, detail="Failed to store challenge phrase")
    return {"phrase": phrase}

@app.post("/verify")
async def verify_video(
    file: UploadFile = File(...),
    ssp_id: int = Form(...),
    profile_image_url: Optional[str] = Form(None),
    profile_image: Optional[UploadFile] = File(None),
    request_id: Optional[str] = Form(None)
):
    logger.info(f"Received verification request for ssp_id {ssp_id}")
    profile_image_bytes = None
    image_type = None
    used_profile_image_url = None
    # Scenario 1: Upload new image
    if profile_image is not None:
        content_type = profile_image.content_type
        extension = "jpg"
        if content_type == "image/png":
            extension = "png"
        elif content_type == "image/jpeg":
            extension = "jpg"
        blob_name = f"{ssp_id}_{request_id}.{extension}"
        content = await profile_image.read()
        used_profile_image_url = upload_blob(
            os.environ["AZURE_PROFILE_CONTAINER"],
            blob_name,
            content,
            content_type
        )
        profile_image_bytes = content
        image_type = "upload"
    # Scenario 2: Use existing image URL
    elif profile_image_url:
        try:
            resp = requests.get(profile_image_url, timeout=5)
            if resp.status_code == 200:
                profile_image_bytes = resp.content
                used_profile_image_url = profile_image_url
                image_type = "url"
            else:
                logger.error(f"Failed to download profile image from {profile_image_url}, status {resp.status_code}")
                return {"ok": False, "error": "Failed to download selected profile image."}
        except Exception as e:
            logger.error(f"Error downloading profile image from {profile_image_url}: {e}")
            return {"ok": False, "error": "Error downloading selected profile image."}
    else:
        return {"ok": False, "error": "You must either upload a new profile image or select an existing one."}
    result = await process_verification(file, ssp_id, profile_image_bytes, used_profile_image_url, image_type, request_id=request_id)
    return result

@app.post("/upload_profile")
async def upload_profile_photo(
    file: UploadFile = File(...),
    ssp_id: int = Form(...),
    request_id: str = Form(...)
):
    logger.info(f"Received profile photo upload for ssp_id {ssp_id} and request_id {request_id}")
    content_type = file.content_type
    if not content_type.startswith("image/"):
        return {"ok": False, "error": "Uploaded file is not an image"}
    extension = "jpg"
    if content_type == "image/png":
        extension = "png"
    elif content_type == "image/jpeg":
        extension = "jpg"
    blob_name = f"{ssp_id}_{request_id}.{extension}"
    try:
        content = await file.read()
        url = upload_blob(
            os.environ["AZURE_PROFILE_CONTAINER"],
            blob_name,
            content,
            content_type
        )
        logger.info(f"Profile photo saved for ssp_id {ssp_id} and request_id {request_id} at {url}")
        return {"ok": True, "profile_url": url}
    except Exception as e:
        logger.error(f"Error saving profile photo: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/profile_photo")
async def get_profile_photo(sspid: int):
    """Fetch all profile photo URLs for a given SSPID from COL_Server and check for face presence with user-friendly reason and technical explanation."""
    session = ColSession()
    try:
        sql = """
        SELECT s.ID, 'https://images.cityoflove.com/dynamic/images/' + 
            LEFT(sp.filename, 
                 CASE WHEN CHARINDEX('?ts=', sp.filename) > 0 
                      THEN CHARINDEX('?ts=', sp.filename) - 1 
                      ELSE LEN(sp.filename) 
                 END
            ) AS filename
        FROM tblSSP AS s
        LEFT JOIN tblSSPphoto AS sp ON sp.fk_SSP = s.ID
        WHERE s.ID = :sspid
        """
        results = session.execute(text(sql), {"sspid": sspid}).fetchall()
        photo_urls = [row.filename for row in results if row.filename]
        if not photo_urls:
            raise HTTPException(status_code=404, detail="Profile photo not found")
        photo_results = []
        for url in photo_urls:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    img_array = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    face_info = face_verifier.check_face_in_image(img)
                else:
                    face_info = {"has_face": False, "reason": "Image could not be loaded"}
            except Exception as e:
                logger.error(f"Error checking face for {url}: {e}")
                face_info = {"has_face": False, "reason": f"Error: {str(e)}"}

            # Map technical reason to user-friendly reason
            technical_reason = face_info.get("reason", "")
            if face_info["has_face"]:
                user_reason = "Face is clearly visible"
            elif "blurry" in technical_reason:
                user_reason = "Face is too blurry"
            elif "too small" in technical_reason:
                user_reason = "Face is too small in the image"
            elif "No face detected" in technical_reason:
                user_reason = "No face detected"
            elif "cropped" in technical_reason:
                user_reason = "Face is not fully visible"
            elif "detection confidence" in technical_reason:
                user_reason = "Face not detected clearly"
            else:
                user_reason = "Face not suitable for verification"

            photo_results.append({
                "url": url,
                "has_face": face_info["has_face"],
                "reason": user_reason,
                "explanation": technical_reason
            })
        return {"sspid": sspid, "photos": photo_results}
    finally:
        session.close()

@app.post("/start-video-kyc")
async def start_video_kyc(ssp_id: int = Form(...)):
    request_id = str(uuid.uuid4())
    kyc_id = insert_kyc_start_record(ssp_id, request_id, status="started")
    return {"kyc_id": kyc_id, "request_id": request_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info",
                access_log=True, use_colors=True, timeout_keep_alive=120)
