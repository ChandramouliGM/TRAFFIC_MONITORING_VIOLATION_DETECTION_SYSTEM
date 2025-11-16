# Handle missing imports gracefully
try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from typing import List, Tuple, Dict, Optional
import streamlit as st
from datetime import datetime, timedelta
import os
import tempfile
import base64

def convert_bbox_format(bbox: List[int], from_format: str = 'xywh', to_format: str = 'xyxy') -> List[int]:
    """Convert bounding box between different formats"""
    if from_format == to_format:
        return bbox
    
    if from_format == 'xywh' and to_format == 'xyxy':
        # [x, y, width, height] to [x1, y1, x2, y2]
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    
    elif from_format == 'xyxy' and to_format == 'xywh':
        # [x1, y1, x2, y2] to [x, y, width, height]
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
    
    else:
        raise ValueError(f"Unsupported conversion: {from_format} to {to_format}")

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes"""
    # Convert to xyxy format if needed
    if len(box1) == 4 and len(box2) == 4:
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    else:
        return 0.0
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if not detections:
        return []
    
    # Sort by confidence (descending)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    for detection in sorted_detections:
        should_keep = True
        
        for kept_detection in keep:
            iou = calculate_iou(detection['bbox'], kept_detection['bbox'])
            
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(detection)
    
    return keep

def resize_image_maintain_aspect(image: np.ndarray, target_width: int = None, 
                                target_height: int = None, max_size: int = None) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    original_height, original_width = image.shape[:2]
    
    if max_size:
        # Resize to fit within max_size x max_size
        if original_width > original_height:
            target_width = max_size
            target_height = int(original_height * max_size / original_width)
        else:
            target_height = max_size
            target_width = int(original_width * max_size / original_height)
    
    elif target_width and not target_height:
        ratio = target_width / original_width
        target_height = int(original_height * ratio)
    
    elif target_height and not target_width:
        ratio = target_height / original_height
        target_width = int(original_width * ratio)
    
    elif not target_width and not target_height:
        return image
    
    return cv2.resize(image, (target_width, target_height))

def encode_image_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
    """Encode image to base64 string"""
    try:
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Encode image
        _, buffer = cv2.imencode(f'.{format.lower()}', image_bgr)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/{format.lower()};base64,{image_base64}"
    
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return ""

def create_video_thumbnail(video_path: str, timestamp: float = None) -> Optional[np.ndarray]:
    """Create thumbnail from video at specified timestamp"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set timestamp (use middle of video if not specified)
        if timestamp is None:
            timestamp = (total_frames / fps) / 2
        
        # Set frame position
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return None
    
    except Exception as e:
        st.error(f"Error creating thumbnail: {str(e)}")
        return None

def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """Validate video file format and properties"""
    try:
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # Check basic properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Validate properties
        if frame_count <= 0:
            return False, "Video has no frames"
        
        if fps <= 0:
            return False, "Invalid frame rate"
        
        if width <= 0 or height <= 0:
            return False, "Invalid video dimensions"
        
        return True, "Video file is valid"
    
    except Exception as e:
        return False, f"Error validating video: {str(e)}"

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def format_duration(seconds: float) -> str:
    """Format duration in seconds to readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(remaining_minutes)}m"

def calculate_processing_progress(current_frame: int, total_frames: int) -> float:
    """Calculate processing progress percentage"""
    if total_frames <= 0:
        return 0.0
    
    return min(current_frame / total_frames, 1.0)

def estimate_remaining_time(processed_frames: int, total_frames: int, 
                          elapsed_time: float) -> float:
    """Estimate remaining processing time"""
    if processed_frames <= 0 or elapsed_time <= 0:
        return 0.0
    
    frames_per_second = processed_frames / elapsed_time
    remaining_frames = total_frames - processed_frames
    
    if frames_per_second <= 0:
        return 0.0
    
    return remaining_frames / frames_per_second

def create_detection_summary(detections: List[Dict]) -> Dict:
    """Create summary statistics for detections"""
    if not detections:
        return {
            'total_detections': 0,
            'vehicle_types': {},
            'average_confidence': 0.0,
            'unique_plates': set()
        }
    
    vehicle_types = {}
    confidences = []
    unique_plates = set()
    
    for detection in detections:
        # Count vehicle types
        vehicle_type = detection.get('vehicle_type', 'Unknown')
        vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
        
        # Collect confidences
        confidence = detection.get('confidence', 0.0)
        confidences.append(confidence)
        
        # Collect unique plates
        plate = detection.get('license_plate')
        if plate:
            unique_plates.add(plate)
    
    return {
        'total_detections': len(detections),
        'vehicle_types': vehicle_types,
        'average_confidence': np.mean(confidences) if confidences else 0.0,
        'unique_plates': unique_plates
    }

def save_detection_results(detections: List[Dict], output_path: str) -> bool:
    """Save detection results to file"""
    try:
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_detections = []
        
        for detection in detections:
            serializable_detection = {}
            
            for key, value in detection.items():
                if isinstance(value, np.ndarray):
                    serializable_detection[key] = value.tolist()
                elif isinstance(value, datetime):
                    serializable_detection[key] = value.isoformat()
                else:
                    serializable_detection[key] = value
            
            serializable_detections.append(serializable_detection)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_detections, f, indent=2)
        
        return True
    
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
        return False

def load_detection_results(file_path: str) -> List[Dict]:
    """Load detection results from file"""
    try:
        import json
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert timestamp strings back to datetime objects
        for detection in data:
            if 'timestamp' in detection:
                detection['timestamp'] = datetime.fromisoformat(detection['timestamp'])
        
        return data
    
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return []

class PerformanceMonitor:
    """Monitor processing performance and provide metrics"""
    
    def __init__(self):
        self.start_time = None
        self.frame_times = []
        self.processed_frames = 0
    
    def start(self):
        """Start performance monitoring"""
        self.start_time = datetime.now()
        self.frame_times = []
        self.processed_frames = 0
    
    def log_frame(self):
        """Log processing time for a frame"""
        if self.start_time:
            self.processed_frames += 1
            current_time = datetime.now()
            self.frame_times.append(current_time)
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.start_time or self.processed_frames == 0:
            return {}
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'elapsed_time': elapsed_time,
            'processed_frames': self.processed_frames,
            'fps': fps,
            'average_frame_time': elapsed_time / self.processed_frames if self.processed_frames > 0 else 0
        }
