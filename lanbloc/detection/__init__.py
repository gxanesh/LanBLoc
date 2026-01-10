"""
Detection module for LanBLoc.

Contains YOLO-based landmark detection.
"""

from .yolo_detector import YOLOLandmarkDetector

__all__ = ["YOLOLandmarkDetector"]
