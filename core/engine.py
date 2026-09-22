from typing import List, Optional, Tuple
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognitionEngine:
    """
    Wraps InsightFace buffalo_l pipeline for detection, alignment,
    embedding extraction, and matching.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        use_gpu: bool = True,
        det_size: Tuple[int, int] = (640, 640),
    ):
        self.model_name = model_name
        self.det_size = det_size

        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        print(f"Initializing InsightFace model '{model_name}' (GPU={use_gpu})...")
        try:
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        except Exception as e:
            if use_gpu:
                print(f"CUDA initialization failed ({e}), falling back to CPU...")
                self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
                self.app.prepare(ctx_id=-1, det_size=self.det_size)
            else:
                raise e

    def analyze_frame(self, frame: np.ndarray):
        """Runs face detection and feature extraction on an OpenCV BGR frame."""
        return self.app.get(frame)

    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts embedding of the most prominent face in an image.
        Returns normalized 512-d vector or None if no face found.
        """
        faces = self.app.get(image)
        if not faces:
            return None

        # Pick the largest face by bounding box area
        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return largest_face.normed_embedding

    def extract_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """Reads image from disk and extracts embedding of the primary face."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        return self.extract_face_embedding(img)

    @staticmethod
    def match_embedding(
        embedding: np.ndarray,
        gallery_names: List[str],
        gallery_matrix: Optional[np.ndarray],
        threshold: float = 0.45,
    ) -> Tuple[str, float]:
        """
        Calculates cosine similarities against all gallery embeddings.
        Returns (name, score). If score < threshold or empty gallery, returns ('STRANGER', score).
        """
        if gallery_matrix is None or len(gallery_names) == 0:
            return "STRANGER", 0.0

        # Cosine similarity for unit vectors: dot product
        similarities = np.dot(gallery_matrix, embedding)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            return gallery_names[best_idx], best_score
        return "STRANGER", best_score
