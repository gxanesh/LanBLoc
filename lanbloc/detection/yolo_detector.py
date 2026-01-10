"""
YOLO-based Landmark Detection Module

This module provides landmark detection using YOLOv11 (Ultralytics).
Supports both pre-trained and custom-trained models for landmark recognition.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import LandmarkDetection from core to avoid circular imports
@dataclass
class LandmarkDetection:
    """Detected landmark with bounding box."""
    landmark_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    class_id: int = 0


class YOLOLandmarkDetector:
    """
    YOLO-based landmark detector using Ultralytics YOLOv11.
    
    This class wraps the Ultralytics YOLO model for landmark detection
    in the LanBLoc localization pipeline.
    
    Attributes:
        model: Ultralytics YOLO model
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for NMS
        class_names: Mapping from class IDs to landmark IDs
    
    Example:
        >>> detector = YOLOLandmarkDetector(weights="weights/landmarks.pt")
        >>> detections = detector.detect(image)
        >>> for det in detections:
        ...     print(f"{det.landmark_id}: {det.confidence:.2f}")
    """
    
    def __init__(
        self,
        weights: str = "yolov8s.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        image_size: int = 640,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize YOLO landmark detector.
        
        Args:
            weights: Path to YOLO weights file or model name
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device for inference ("cuda", "cpu", or GPU index)
            image_size: Input image size for inference
            class_names: Optional mapping from class IDs to landmark IDs
        """
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.image_size = image_size
        self.class_names = class_names or {}
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"YOLOLandmarkDetector initialized: weights={weights}, "
                   f"conf={confidence_threshold}, device={device}")
    
    def _load_model(self):
        """Load YOLO model from weights."""
        try:
            from ultralytics import YOLO
            
            model = YOLO(self.weights)
            
            # Set device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    model.to("cuda")
                else:
                    logger.warning("CUDA not available, using CPU")
                    model.to("cpu")
            else:
                model.to(self.device)
            
            # Update class names from model if not provided
            if not self.class_names and hasattr(model, 'names'):
                self.class_names = {i: f"l{i+1}" for i in range(len(model.names))}
            
            return model
            
        except ImportError:
            logger.error("Ultralytics not installed. Run: pip install ultralytics")
            return None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None
    
    def detect(
        self,
        image: np.ndarray,
        return_raw: bool = False
    ) -> List[LandmarkDetection]:
        """
        Detect landmarks in image.
        
        Args:
            image: Input image (BGR or RGB)
            return_raw: If True, also return raw YOLO results
            
        Returns:
            List of LandmarkDetection objects
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for i in range(len(boxes)):
                    # Get bounding box (xyxy format)
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Convert to (x, y, w, h) format
                    bbox = (
                        int(x1),
                        int(y1),
                        int(x2 - x1),
                        int(y2 - y1)
                    )
                    
                    # Get class and confidence
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Get landmark ID
                    landmark_id = self.class_names.get(class_id, f"l{class_id + 1}")
                    
                    detections.append(LandmarkDetection(
                        landmark_id=landmark_id,
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id
                    ))
            
            logger.debug(f"Detected {len(detections)} landmarks")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[List[LandmarkDetection]]:
        """
        Detect landmarks in multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists (one per image)
        """
        if self.model is None:
            return [[] for _ in images]
        
        try:
            # Run batch inference
            results = self.model(
                images,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                verbose=False
            )
            
            all_detections = []
            
            for result in results:
                detections = []
                boxes = result.boxes
                
                if boxes is not None:
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        class_id = int(boxes.cls[i].cpu().numpy())
                        confidence = float(boxes.conf[i].cpu().numpy())
                        landmark_id = self.class_names.get(class_id, f"l{class_id + 1}")
                        
                        detections.append(LandmarkDetection(
                            landmark_id=landmark_id,
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id
                        ))
                
                all_detections.append(detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [[] for _ in images]
    
    def set_class_names(self, class_names: Dict[int, str]) -> None:
        """
        Set custom class name mapping.
        
        Args:
            class_names: Dictionary mapping class IDs to landmark IDs
        """
        self.class_names = class_names
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[LandmarkDetection],
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detections
            colors: Optional color mapping for landmarks
            
        Returns:
            Image with drawn detections
        """
        import cv2
        
        vis_image = image.copy()
        
        for det in detections:
            x, y, w, h = det.bbox
            
            # Get color
            if colors and det.landmark_id in colors:
                color = colors[det.landmark_id]
            else:
                # Generate color from landmark ID hash
                hash_val = hash(det.landmark_id)
                color = (
                    (hash_val & 0xFF),
                    ((hash_val >> 8) & 0xFF),
                    ((hash_val >> 16) & 0xFF)
                )
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{det.landmark_id}: {det.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x, y - label_h - baseline - 5),
                (x + label_w, y),
                color,
                -1
            )
            cv2.putText(
                vis_image,
                label,
                (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return vis_image


class MockLandmarkDetector:
    """
    Mock detector for testing without YOLO.
    
    Returns predefined detections for testing the localization pipeline.
    """
    
    def __init__(self, detections: Optional[List[LandmarkDetection]] = None):
        """
        Initialize mock detector.
        
        Args:
            detections: Predefined detections to return
        """
        self.detections = detections or []
    
    def detect(self, image: np.ndarray) -> List[LandmarkDetection]:
        """Return predefined detections."""
        return self.detections
    
    def set_detections(self, detections: List[LandmarkDetection]) -> None:
        """Set detections to return."""
        self.detections = detections
