import os
import cv2
import numpy as np
import asyncio
from insightface.app import FaceAnalysis
import logging
from typing import Optional, Tuple, Dict
import numpy.typing as npt
from app.infra.db import get_challenge_phrase

class FaceVerifier:
    def __init__(self, providers: list[str] = ['CPUExecutionProvider'], det_size: tuple[int, int] = (640, 640)):
        """Initialize the FaceVerifier with model configuration."""
        self.face_app: Optional[FaceAnalysis] = None
        self.providers = providers
        self.det_size = det_size
        self.logger = logging.getLogger("video-verification")
        # Thresholds for face visibility
        self.min_face_area_ratio = 0.02
        self.min_face_size = 50  # Lowered from 100 to allow smaller/partial faces
        self.min_confidence = 0.7  # Lowered from 0.9 to allow partial detections
        self.min_sharpness = 40.0
        self.min_tenengrad = 20.0  # Lowered from 30.0

    def init_face_model(self) -> None:
        """Initialize the InsightFace model."""
        if self.face_app is None:
            try:
                self.face_app = FaceAnalysis(providers=self.providers)
                self.face_app.prepare(ctx_id=0, det_size=self.det_size)
                self.logger.info("InsightFace model initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing InsightFace model: {e}")
                raise e

    def preprocess_image(self, image: npt.NDArray[np.uint8], max_size: int = 640) -> npt.NDArray[np.uint8]:
        """Resize image to reduce processing time while maintaining aspect ratio."""
        try:
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
            return image
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return image

    def compute_sharpness(self, image: npt.NDArray[np.uint8]) -> float:
        """Compute sharpness of an image using Laplacian variance."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(laplacian_var)
        except Exception as e:
            self.logger.error(f"Error computing sharpness: {e}")
            return 0.0

    def compute_tenengrad(self, image: npt.NDArray[np.uint8]) -> float:
        """Compute sharpness using the Tenengrad (Sobel) method."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = np.sqrt(gx ** 2 + gy ** 2).mean()
            return float(tenengrad)
        except Exception as e:
            self.logger.error(f"Error computing Tenengrad sharpness: {e}")
            return 0.0

    def is_face_clearly_visible(self, image: npt.NDArray[np.uint8], 
                              face: dict) -> Tuple[bool, str]:
        """
        Check if the detected face is clearly visible and not blurry.

        Args:
            image: Input image as a NumPy array (BGR format).
            face: Detected face object from insightface.

        Returns:
            Tuple of (is_visible, reason):
                - is_visible: True if face is clearly visible, False otherwise.
                - reason: Explanation for why the face is not visible (if applicable).
        """
        try:
            img_h, img_w = image.shape[:2]
            img_area = img_h * img_w

            confidence = face.det_score
            if confidence < self.min_confidence:
                return False, f"Low detection confidence: {confidence:.2f} < {self.min_confidence}"

            bbox = face.bbox
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_area = face_width * face_height
            area_ratio = face_area / img_area

            if face_width < self.min_face_size or face_height < self.min_face_size:
                return False, f"Face too small: {face_width}x{face_height} pixels"
            if area_ratio < self.min_face_area_ratio:
                return False, f"Face area too small: {area_ratio:.3f} < {self.min_face_area_ratio}"

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                return False, "Invalid face bounding box"
            face_region = image[y1:y2, x1:x2]

            # Compute both Laplacian and Tenengrad sharpness
            sharpness = self.compute_sharpness(face_region)
            tenengrad = self.compute_tenengrad(face_region)
            self.logger.info(f"Face sharpness (Laplacian): {sharpness:.2f} (threshold: {self.min_sharpness}), Tenengrad: {tenengrad:.2f} (threshold: {self.min_tenengrad})")

            if sharpness < self.min_sharpness:
                return False, f"Face too blurry (Laplacian): sharpness {sharpness:.2f} < {self.min_sharpness}"
            if tenengrad < self.min_tenengrad:
                return False, f"Face too blurry (Tenengrad): sharpness {tenengrad:.2f} < {self.min_tenengrad}"

            return True, "Face is clearly visible"
        except Exception as e:
            self.logger.error(f"Error checking face visibility: {e}")
            return False, f"Error: {str(e)}"

    def extract_frame_from_video(self, video_path: str, frame_index: int = 10) -> Optional[npt.NDArray[np.uint8]]:
        """
        Extract a frame from the video for face analysis.
        """
        if not isinstance(video_path, str) or not os.path.exists(video_path):
            self.logger.error(f"Invalid or missing video file: {video_path}")
            return None
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                self.logger.error(f"Video {video_path} has no frames")
                cap.release()
                return None
            target_frame = min(frame_index, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.logger.error(f"Failed to extract frame {target_frame} from {video_path}")
                return None
            return frame
        except Exception as e:
            self.logger.error(f"Error extracting frame from {video_path}: {e}")
            return None

    def check_face_in_image(self, image: npt.NDArray[np.uint8]) -> Dict[str, str]:
        """
        Check if a face is clearly visible in the image and return details.
        """
        self.init_face_model()
        try:
            image = self.preprocess_image(image)
            img_h, img_w = image.shape[:2]
            if img_h / img_w > 2:
                self.logger.info("Image likely cropped, head may not be visible")
                return {"has_face": False, "reason": "Head likely cropped out of frame"}
            faces = self.face_app.get(image)
            if not faces:
                self.logger.info("No face detected in the image")
                return {"has_face": False, "reason": "No face detected"}
            if len(faces) > 1:
                self.logger.warning(f"Multiple faces detected: {len(faces)}, using largest")
            faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
            largest_face = faces[0]
            is_visible, reason = self.is_face_clearly_visible(image, largest_face)
            if not is_visible:
                self.logger.info(f"Face not clearly visible: {reason}")
                return {"has_face": False, "reason": reason}
            self.logger.info("Face is clearly visible")
            return {"has_face": True, "reason": "Face is clearly visible"}
        except Exception as e:
            self.logger.error(f"Error checking face in image: {e}")
            return {"has_face": False, "reason": f"Error: {str(e)}"}

    def get_face_embedding(self, image: npt.NDArray[np.uint8]) -> Optional[npt.NDArray[np.float32]]:
        """
        Get face embedding from an image if the face is clearly visible.
        """
        result = self.check_face_in_image(image)
        if not result["has_face"]:
            return None
        try:
            image = self.preprocess_image(image)  # Re-preprocess if needed
            faces = self.face_app.get(image)
            if not faces:
                return None
            faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
            return faces[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting face embedding: {e}")
            return None

    def compare_face_embeddings(self, embedding1: Optional[npt.NDArray[np.float32]], 
                              embedding2: Optional[npt.NDArray[np.float32]]) -> float:
        """
        Compare two face embeddings and return similarity score.

        Args:
            embedding1: First face embedding.
            embedding2: Second face embedding.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        if embedding1 is None or embedding2 is None:
            self.logger.warning("One or both embeddings are None")
            return 0.0
        try:
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            normalized_score = np.clip((similarity + 1) / 2, 0.0, 1.0)
            return float(round(normalized_score, 2))
        except Exception as e:
            self.logger.error(f"Error comparing embeddings: {e}")
            return 0.0

    async def calculate_face_similarity(self, video_path: str, user_id: str, profile_image_bytes: Optional[bytes]) -> float:
        """
        Calculate face similarity between video frame and profile photo.
        """
        try:
            if not isinstance(video_path, str) or not os.path.exists(video_path):
                self.logger.error(f"Invalid video path for user {user_id}: {video_path}")
                return 0.0
            video_frame = await asyncio.to_thread(self.extract_frame_from_video, video_path)
            if video_frame is None:
                self.logger.error(f"No valid frame extracted for user {user_id}")
                return 0.0
            if not profile_image_bytes:
                self.logger.warning(f"No profile photo bytes provided for user {user_id}")
                return 0.0
            np_arr = np.frombuffer(profile_image_bytes, np.uint8)
            profile_image = await asyncio.to_thread(cv2.imdecode, np_arr, cv2.IMREAD_COLOR)
            if profile_image is None:
                self.logger.error(f"Failed to decode profile photo for user {user_id}")
                return 0.0
            video_embedding = await asyncio.to_thread(self.get_face_embedding, video_frame)
            profile_embedding = await asyncio.to_thread(self.get_face_embedding, profile_image)
            similarity = await asyncio.to_thread(self.compare_face_embeddings, video_embedding, profile_embedding)
            self.logger.info(f"Face similarity score for user {user_id}: {similarity}")
            return similarity
        except Exception as e:
            self.logger.error(f"Error calculating face similarity for user {user_id}: {e}")
            return 0.0