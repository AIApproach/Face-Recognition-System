"""
Face detection and recognition module.
Updated for TensorFlow 2.x+ with fixed dimension handling
"""
import os
import pickle
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
import src.facenet as facenet
# import mtcnn.mtcnn as MTCNN

# --- Configuration ---
GPU_MEMORY_LIMIT = 1024  # MB
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/facenet/")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "models/known_embeddings.pkl")
DEBUG = False

class Face:
    """Represents a detected face with associated attributes."""
    def __init__(self):
        self.name: Optional[str] = None
        self.score: Optional[float] = None
        self.bounding_box: Optional[np.ndarray] = None  # [x1, y1, x2, y2]
        self.image: Optional[np.ndarray] = None
        self.container_image: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None

class Recognition:
    """Main face recognition pipeline combining detection, encoding and identification."""
    def __init__(self):
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT)]
                )
            except RuntimeError as e:
                print(e)
        
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image: np.ndarray, person_name: str) -> Optional[List[Face]]:
        """Add a new face identity to the recognition system."""
        faces = self.detect.find_faces(image)
        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            self.identifier.add_embedding(face.embedding, person_name)
            return faces
        return None

    def identify(self, image: np.ndarray) -> List[Face]:
        """Identify faces in an image."""
        faces = self.detect.find_faces(image)
        for i, face in enumerate(faces):
            if DEBUG:
                cv2.imshow(f"Face {i}", face.image)
                cv2.waitKey(1)
            face.embedding = self.encoder.generate_embedding(face)
            face.name, face.score = self.identifier.identify(face)
        return faces

class Identifier:
    """Handles face identification by comparing embeddings."""
    def __init__(self):
        self.candidate_count = 5
        self.threshold = 0.65
        self.known_embeddings: np.ndarray = np.zeros((0, 128))  # Default shape for FaceNet
        self.labels: List[str] = []

        # Load known embeddings if file exists
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, 'rb') as f:
                loaded_embeddings, loaded_labels = pickle.load(f)
                
                # Verify embedding dimensions
                if loaded_embeddings.shape[1] != 128:
                    print(f"Warning: Loaded embeddings have dimension {loaded_embeddings.shape[1]}, truncating to 128")
                    loaded_embeddings = loaded_embeddings[:, :128]
                
                self.known_embeddings = loaded_embeddings
                self.labels = loaded_labels

    def add_embedding(self, embedding: np.ndarray, label: str) -> None:
        """Add a new embedding to the known embeddings database."""
        if embedding.shape[0] != 128:
            raise ValueError(f"Embedding must be 128-dimensional, got {embedding.shape[0]}")
            
        self.known_embeddings = np.vstack([self.known_embeddings, embedding])
        self.labels.append(label)
        self._save_embeddings()

    def _save_embeddings(self) -> None:
        """Save current embeddings to disk."""
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump((self.known_embeddings, self.labels), f)

    def identify(self, face: Face) -> Tuple[str, float]:
        """Identify a face by comparing against known embeddings."""
        if face.embedding is None or len(self.known_embeddings) == 0:
            return "unknown", 0.0

        # Ensure consistent dimensions
        if face.embedding.shape[0] != 128:
            print(f"Error: Face embedding has wrong dimension {face.embedding.shape[0]}, expected 128")
            return "unknown", 0.0

        face_embedding = face.embedding.reshape(1, -1)
        similarity = cosine_similarity(self.known_embeddings, face_embedding).squeeze()

        # Find top candidates
        candidate_idx = np.argpartition(np.abs(similarity), -self.candidate_count)[-self.candidate_count:]
        candidate_idx = candidate_idx[np.abs(similarity[candidate_idx]) >= self.threshold]
        # Get most common label among candidates
        candidate_labels = [self.labels[i] for i in candidate_idx]

        if not candidate_labels:
            return "unknown", 0.0
        
        best_label = max(candidate_labels, key=candidate_labels.count)
        best_score = max(similarity[candidate_idx])
        
        return best_label, best_score

class Encoder:
    """Generates face embeddings using a pre-trained FaceNet model."""
    def __init__(self):
        # Load FaceNet model
        model_path = os.path.join(MODEL_DIR, "facenet_keras.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FaceNet model not found at {model_path}")
        
        self.model = load_model(model_path)
        # Verify embedding dimension
        test_input = np.zeros((1, 160, 160, 3), dtype=np.float32)
        test_embedding = self.model.predict(test_input)
        print(f"Model produces embeddings of dimension: {test_embedding.shape[1]}")
        
    def generate_embedding(self, face: Face) -> Optional[np.ndarray]:
        """Generate embedding for a face image."""
        if face.image is None:
            return None

        # Preprocess and get embedding
        prewhitened = facenet.prewhiten(face.image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(prewhitened, axis=0), dtype=tf.float32)
        embedding = self.model(input_tensor).numpy()[0]
        return embedding

class Detection:
    """Handles face detection using MTCNN."""
    def __init__(self, crop_size: int = 160, crop_margin: int = 32):
        self.crop_size = crop_size
        self.crop_margin = crop_margin
        self.minsize = 20
        self.mtcnn = MTCNN()

    def find_faces(self, image: np.ndarray) -> List[Face]:
        """Detect faces in an image."""
        faces = []
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        try:
            detections = self.mtcnn.detect_faces(image)
        except Exception as e:
            print(f"Face detection error: {e}")
            return faces
        
        for detection in detections:
            if detection['confidence'] < 0.9:  # Confidence threshold
                continue
                
            face = Face()
            face.container_image = image
            
            # Get bounding box coordinates
            bb = detection['box']
            x, y, width, height = bb
            
            # Calculate coordinates with margin
            x1 = max(0, x - self.crop_margin // 2)
            y1 = max(0, y - self.crop_margin // 2)
            x2 = min(image.shape[1], x + width + self.crop_margin // 2)
            y2 = min(image.shape[0], y + height + self.crop_margin // 2)
            
            face.bounding_box = np.array([x1, y1, x2, y2])
            
            # Crop and resize face
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
                
            face.image = cv2.resize(
                cropped,
                (self.crop_size, self.crop_size),
                interpolation=cv2.INTER_AREA
            )
            faces.append(face)
            
        return faces