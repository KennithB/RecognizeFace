from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class ProfileManager:
    """
    Manages known face profiles stored as .npy vectors.
    """

    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, np.ndarray] = {}
        self.reload_profiles()

    def reload_profiles(self) -> Dict[str, np.ndarray]:
        """Loads all .npy embedding vectors from the profiles directory."""
        self.profiles.clear()
        for file_path in self.profiles_dir.glob("*.npy"):
            try:
                embedding = np.load(str(file_path))
                if embedding.ndim == 1 and embedding.shape[0] == 512:
                    # Ensure normalized
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    self.profiles[file_path.stem] = embedding
            except Exception as e:
                print(f"Failed to load profile {file_path}: {e}")
        return self.profiles

    def save_profile(self, name: str, embedding: np.ndarray) -> bool:
        """Saves or updates a single profile embedding."""
        clean_name = name.strip()
        if not clean_name:
            return False

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        target_file = self.profiles_dir / f"{clean_name}.npy"
        np.save(str(target_file), embedding.astype(np.float32))
        self.profiles[clean_name] = embedding
        return True

    def save_profile_from_multiple(self, name: str, embeddings: List[np.ndarray]) -> bool:
        """Averages multiple embeddings for a single profile, normalizes, and saves."""
        if not embeddings:
            return False
        
        stacked = np.vstack(embeddings)
        mean_embedding = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        return self.save_profile(name, mean_embedding)

    def delete_profile(self, name: str) -> bool:
        """Removes a profile from disk and memory."""
        target_file = self.profiles_dir / f"{name}.npy"
        deleted = False
        if target_file.exists():
            target_file.unlink()
            deleted = True
        if name in self.profiles:
            del self.profiles[name]
            deleted = True
        return deleted

    def list_profiles(self) -> List[str]:
        """Returns sorted list of profile names."""
        return sorted(list(self.profiles.keys()))

    def get_gallery_matrix(self):
        """
        Returns (names_list, embeddings_matrix) for vectorized cosine similarity.
        """
        names = list(self.profiles.keys())
        if not names:
            return [], None
        embeddings = np.stack([self.profiles[name] for name in names], axis=0)
        return names, embeddings
