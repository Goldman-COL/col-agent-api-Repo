import os
import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import tempfile
import logging

logger = logging.getLogger("video-verification")

# Initialize the face analysis model
face_app = None

def init_face_model():
    global face_app
    if face_app is None:
        try:
            # Initialize the InsightFace model
            face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing InsightFace model: {e}")
            raise e

def extract_frame_from_video(video_bytes):
    """Extract a frame from the video for face analysis"""
    try:
        # Save video bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as video_temp:
            video_temp.write(video_bytes)
            video_temp_path = video_temp.name

        try:
            cap = cv2.VideoCapture(video_temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Use a frame from about 1/3 into the video to give time for user to settle
            target_frame = min(10, total_frames - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Failed to extract frame {target_frame} from video")
                return None
                
            return frame
        finally:
            # Clean up temporary file
            try:
                os.unlink(video_temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {e}")
    except Exception as e:
        logger.error(f"Error extracting frame from video: {e}")
        return None

def get_face_embedding(image):
    """Get face embedding from an image using InsightFace"""
    init_face_model()
    
    try:
        faces = face_app.get(image)
        
        if not faces:
            logger.warning("No face detected in the image")
            return None
            
        # Use the largest face if multiple faces are detected
        faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
        largest_face = faces[0]
        
        return largest_face.embedding
    except Exception as e:
        logger.error(f"Error getting face embedding: {e}")
        return None

def compare_face_embeddings(embedding1, embedding2):
    """Compare two face embeddings and return similarity score"""
    if embedding1 is None or embedding2 is None:
        return 0.0
        
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    # Normalize to 0-1 range (InsightFace similarity scores are typically in range -1 to 1)
    normalized_score = (similarity + 1) / 2
    
    return float(round(normalized_score, 2))

async def calculate_face_similarity(video_bytes: bytes, user_id: str, profile_image_bytes: bytes) -> float:
    """Calculate face similarity between video frame and profile photo"""
    try:
        if profile_image_bytes is None:
            logger.warning(f"No profile photo found for user {user_id}")
            return 0.0
            
        # Extract frame from video
        video_frame = extract_frame_from_video(video_bytes)
        if video_frame is None:
            return 0.0
            
        # Convert profile image bytes to numpy array
        profile_np = np.frombuffer(profile_image_bytes, np.uint8)
        profile_image = cv2.imdecode(profile_np, cv2.IMREAD_COLOR)
        if profile_image is None:
            logger.error("Failed to decode profile image")
            return 0.0
            
        # Get face embeddings
        video_embedding = get_face_embedding(video_frame)
        profile_embedding = get_face_embedding(profile_image)
        
        # Compare embeddings
        similarity = compare_face_embeddings(video_embedding, profile_embedding)
        
        logger.info(f"Face similarity score: {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"Error calculating face similarity: {e}")
        return 0.0
