import cv2
import asyncio
import logging
from deepface import DeepFace

logger = logging.getLogger("video-verification.liveness")

async def calculate_liveness_score(video_path: str, frame_sample_rate: int = 5) -> float:
    logger.info(f"Starting liveness calculation on video: {video_path} with sampling rate 1/{frame_sample_rate}")
    
    def analyze(path):
        logger.info(f"Opening video file for frame extraction: {path}")
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {path}")
            return 0.0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video properties - FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")
        
        frame_index = 0
        analyzed_frames = 0
        real_count = 0
        spoofed_count = 0
        error_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_index += 1
            
            if frame_index % frame_sample_rate != 0:
                continue
                
            analyzed_frames += 1
            
            if analyzed_frames % 5 == 0:
                logger.info(f"Processing frame {frame_index}/{frame_count} (analyzed: {analyzed_frames})")
            
            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                    anti_spoofing=True
                )
                
                if faces:
                    if faces[0].get("is_real", False):
                        real_count += 1
                        logger.debug(f"Frame {frame_index} detected as real with score {faces[0].get('antispoof_score')}")
                    else:
                        spoofed_count += 1
                        logger.debug(f"Frame {frame_index} detected as spoofed with score {faces[0].get('antispoof_score')}")
                else:
                    logger.debug(f"No faces detected in frame {frame_index}")
            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing frame {frame_index}: {str(e)}")
        
        cap.release()
        
        live_ratio = real_count / analyzed_frames if analyzed_frames > 0 else 0.0
        
        logger.info(f"Liveness analysis complete - "
                   f"Analyzed frames: {analyzed_frames}, "
                   f"Real frames: {real_count}, "
                   f"Spoofed frames: {spoofed_count}, "
                   f"Error frames: {error_count}, "
                   f"Live ratio: {live_ratio:.4f}")
        
        return live_ratio

    try:
        live_ratio = await asyncio.to_thread(analyze, video_path)
        logger.info(f"Final liveness score: {live_ratio:.4f}")
        return max(0.0, min(1.0, live_ratio))
    except Exception as e:
        logger.error(f"Exception during liveness calculation: {str(e)}")
        return 0.0
