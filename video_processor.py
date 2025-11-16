# Handle missing imports gracefully
try:
    import cv2
except ImportError:
    cv2 = None
    
import numpy as np
from typing import Generator, Tuple, List
import streamlit as st

class VideoProcessor:
    """Handles video file processing and frame extraction"""
    
    def __init__(self, video_path: str):
        """Initialize video processor with video file path"""
        self.video_path = video_path
        
        if cv2 is None:
            st.warning("⚠️ OpenCV not available. Using mock video processor.")
            self.cap = None
            # Mock video properties
            self.fps = 30.0
            self.total_frames = 300  # 10 seconds at 30fps
            self.width = 640
            self.height = 480
            self.duration = 10.0
            return
            
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties with safeguards
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:  # Guard against invalid fps
            self.fps = 30.0  # Default to 30 fps
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Note: total_frames may be 0 for many video formats, handled in get_frames
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.total_frames > 0 else 0
    
    def get_total_frames(self) -> int:
        """Get total number of frames in video"""
        return self.total_frames
    
    def get_fps(self) -> float:
        """Get frames per second of video"""
        return self.fps
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get video dimensions (width, height)"""
        return self.width, self.height
    
    def get_duration(self) -> float:
        """Get video duration in seconds"""
        return self.duration
    
    def get_frames(self, start_frame: int = 0, end_frame: int = None) -> Generator[np.ndarray, None, None]:
        """Generator that yields video frames"""
        if self.cap is None:
            # Generate mock frames - use reliable fallback
            effective_end = end_frame if end_frame is not None else max(self.total_frames, 100)
            for i in range(start_frame, effective_end):
                # Create a simple colored frame
                mock_frame = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
                # Add some variation based on frame number
                mock_frame[:, :, 0] = (i * 2) % 255
                yield mock_frame
            return
        
        # Set starting position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        
        if end_frame is None:
            # Read until video ends - don't rely on total_frames
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb
                current_frame += 1
        else:
            # Read specific range
            while current_frame < end_frame:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb
                current_frame += 1
    
    def get_frame_at_time(self, timestamp: float) -> np.ndarray:
        """Get frame at specific timestamp (in seconds)"""
        if self.cap is None:
            # Return mock frame for the timestamp
            mock_frame = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
            mock_frame[:, :, 0] = int((timestamp * 30) % 255)  # Vary by time
            return mock_frame
            
        frame_number = int(timestamp * self.fps)
        return self.get_frame_at_index(frame_number)
    
    def get_frame_at_index(self, frame_index: int) -> np.ndarray:
        """Get frame at specific index"""
        if frame_index >= self.total_frames:
            raise ValueError(f"Frame index {frame_index} exceeds total frames {self.total_frames}")
        
        if self.cap is None:
            # Return mock frame for the index
            mock_frame = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
            mock_frame[:, :, 0] = int((frame_index * 2) % 255)  # Vary by frame index
            return mock_frame
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"Could not read frame at index {frame_index}")
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def crop_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crop region from frame using bounding box [x, y, width, height]"""
        if len(bbox) != 4:
            raise ValueError("Bounding box must have 4 elements: [x, y, width, height]")
        
        x, y, width, height = bbox
        
        # Ensure coordinates are within frame bounds
        x = max(0, min(x, frame.shape[1]))
        y = max(0, min(y, frame.shape[0]))
        width = max(1, min(width, frame.shape[1] - x))
        height = max(1, min(height, frame.shape[0] - y))
        
        return frame[y:y+height, x:x+width]
    
    def resize_frame(self, frame: np.ndarray, target_width: int = None, target_height: int = None) -> np.ndarray:
        """Resize frame while maintaining aspect ratio"""
        if cv2 is None:
            # Use numpy-based resizing as fallback
            return self._numpy_resize(frame, target_width, target_height)
        
        original_height, original_width = frame.shape[:2]
        
        if target_width and target_height:
            return cv2.resize(frame, (target_width, target_height))
        elif target_width:
            ratio = target_width / original_width
            target_height = int(original_height * ratio)
            return cv2.resize(frame, (target_width, target_height))
        elif target_height:
            ratio = target_height / original_height
            target_width = int(original_width * ratio)
            return cv2.resize(frame, (target_width, target_height))
        else:
            return frame
    
    def _numpy_resize(self, frame: np.ndarray, target_width: int = None, target_height: int = None) -> np.ndarray:
        """Simple numpy-based resize fallback when cv2 is not available"""
        if target_width is None and target_height is None:
            return frame
        
        original_height, original_width = frame.shape[:2]
        
        if target_width and target_height:
            # Simple nearest neighbor resizing
            y_indices = np.round(np.linspace(0, original_height - 1, target_height)).astype(int)
            x_indices = np.round(np.linspace(0, original_width - 1, target_width)).astype(int)
            return frame[np.ix_(y_indices, x_indices)]
        elif target_width:
            ratio = target_width / original_width
            target_height = int(original_height * ratio)
            y_indices = np.round(np.linspace(0, original_height - 1, target_height)).astype(int)
            x_indices = np.round(np.linspace(0, original_width - 1, target_width)).astype(int)
            return frame[np.ix_(y_indices, x_indices)]
        elif target_height:
            ratio = target_height / original_height
            target_width = int(original_width * ratio)
            y_indices = np.round(np.linspace(0, original_height - 1, target_height)).astype(int)
            x_indices = np.round(np.linspace(0, original_width - 1, target_width)).astype(int)
            return frame[np.ix_(y_indices, x_indices)]
        
        return frame
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        if cv2 is None:
            # Return frame with numpy-based simple annotations when cv2 is not available
            return self._numpy_draw_boxes(frame, detections)
        
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            label = detection.get('class', 'Unknown')
            confidence = detection.get('confidence', 0.0)
            
            x, y, width, height = bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(
                annotated_frame, 
                (x, y - label_size[1] - 10), 
                (x + label_size[0], y), 
                (0, 255, 0), 
                -1
            )
            
            # Text
            cv2.putText(
                annotated_frame, 
                label_text, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        return annotated_frame
    
    def draw_violation_boxes(self, frame: np.ndarray, violations: List[dict]) -> np.ndarray:
        """Draw red bounding boxes for violation vehicles"""
        if cv2 is None:
            # Return frame with numpy-based red annotations when cv2 is not available
            return self._numpy_draw_violation_boxes(frame, violations)
        
        annotated_frame = frame.copy()
        
        for violation in violations:
            bbox = violation.get('bbox', [0, 0, 0, 0])
            vehicle_type = violation.get('vehicle_type', 'Unknown')
            violation_type = violation.get('violation_type', 'Violation')
            license_plate = violation.get('license_plate', '')
            confidence = violation.get('confidence', 0.0)
            
            x, y, width, height = bbox
            
            # Draw red bounding box for violation
            cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 0, 255), 3)
            
            # Create violation label
            label_text = f"VIOLATION: {violation_type}"
            plate_text = f"Plate: {license_plate}" if license_plate else ""
            vehicle_text = f"{vehicle_type}: {confidence:.2f}"
            
            # Calculate text sizes
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            plate_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0] if plate_text else (0, 0)
            vehicle_size = cv2.getTextSize(vehicle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Calculate background size
            max_width = max(label_size[0], plate_size[0], vehicle_size[0])
            total_height = label_size[1] + (plate_size[1] if plate_text else 0) + vehicle_size[1] + 20
            
            # Draw red background for violation text
            cv2.rectangle(
                annotated_frame, 
                (x, y - total_height - 5), 
                (x + max_width + 10, y), 
                (0, 0, 255), 
                -1
            )
            
            # Draw violation text in white
            text_y = y - total_height + 15
            cv2.putText(
                annotated_frame, 
                label_text, 
                (x + 5, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
            # Draw license plate text if available
            if plate_text:
                text_y += plate_size[1] + 5
                cv2.putText(
                    annotated_frame, 
                    plate_text, 
                    (x + 5, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
            
            # Draw vehicle info
            text_y += vehicle_size[1] + 5
            cv2.putText(
                annotated_frame, 
                vehicle_text, 
                (x + 5, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        return annotated_frame
    
    def _numpy_draw_boxes(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Simple numpy-based bounding box drawing when cv2 is not available"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            x, y, width, height = bbox
            
            # Ensure coordinates are within bounds
            h, w = frame.shape[:2]
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            width = max(1, min(width, w-x))
            height = max(1, min(height, h-y))
            
            # Draw simple box outline by setting border pixels to green
            try:
                # Top and bottom edges
                annotated_frame[y:y+2, x:x+width] = [0, 255, 0]
                annotated_frame[y+height-2:y+height, x:x+width] = [0, 255, 0]
                
                # Left and right edges
                annotated_frame[y:y+height, x:x+2] = [0, 255, 0]
                annotated_frame[y:y+height, x+width-2:x+width] = [0, 255, 0]
            except (IndexError, ValueError):
                # Skip if coordinates are invalid
                pass
        
        return annotated_frame
    
    def _numpy_draw_violation_boxes(self, frame: np.ndarray, violations: List[dict]) -> np.ndarray:
        """Simple numpy-based red bounding box drawing for violations when cv2 is not available"""
        annotated_frame = frame.copy()
        
        for violation in violations:
            bbox = violation.get('bbox', [0, 0, 0, 0])
            x, y, width, height = bbox
            
            # Ensure coordinates are within bounds
            h, w = frame.shape[:2]
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            width = max(1, min(width, w-x))
            height = max(1, min(height, h-y))
            
            # Draw thick red box outline (3 pixels thick)
            try:
                # Top and bottom edges (3 pixels thick)
                annotated_frame[y:y+3, x:x+width] = [255, 0, 0]
                annotated_frame[y+height-3:y+height, x:x+width] = [255, 0, 0]
                
                # Left and right edges (3 pixels thick)
                annotated_frame[y:y+height, x:x+3] = [255, 0, 0]
                annotated_frame[y:y+height, x+width-3:x+width] = [255, 0, 0]
                
                # Add violation indicator (red patch at top-left corner)
                if y >= 20 and x >= 60:
                    annotated_frame[y-20:y-10, x:x+60] = [255, 0, 0]
            except (IndexError, ValueError):
                # Skip if coordinates are invalid
                pass
        
        return annotated_frame
    
    def extract_thumbnail(self, timestamp: float = None) -> np.ndarray:
        """Extract thumbnail from video at specified timestamp or middle of video"""
        if timestamp is None:
            timestamp = self.duration / 2  # Middle of video
        
        return self.get_frame_at_time(timestamp)
    
    def save_frame(self, frame: np.ndarray, output_path: str) -> bool:
        """Save frame to file"""
        if cv2 is None:
            try:
                from PIL import Image
                image = Image.fromarray(frame)
                image.save(output_path)
                return True
            except Exception as e:
                st.error(f"Error saving frame (PIL fallback): {str(e)}")
                return False
        
        try:
            # Convert RGB back to BGR for saving
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return cv2.imwrite(output_path, frame_bgr)
        except Exception as e:
            st.error(f"Error saving frame: {str(e)}")
            return False
    
    def create_video_writer(self, output_path: str, fps: float = None) -> 'cv2.VideoWriter':
        """Create video writer for output video"""
        if cv2 is None:
            st.warning("⚠️ Cannot create video writer - OpenCV not available")
            return None
            
        if fps is None:
            fps = self.fps
            
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        
        if not writer.isOpened():
            st.error(f"❌ Could not open video writer for {output_path}")
            return None
            
        return writer
    
    def close(self):
        """Release video capture object"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
    
    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.close()

class FrameBuffer:
    """Buffer for storing and managing video frames"""
    
    def __init__(self, max_size: int = 100):
        """Initialize frame buffer with maximum size"""
        self.max_size = max_size
        self.frames = []
        self.frame_indices = []
    
    def add_frame(self, frame: np.ndarray, frame_index: int):
        """Add frame to buffer"""
        if len(self.frames) >= self.max_size:
            # Remove oldest frame
            self.frames.pop(0)
            self.frame_indices.pop(0)
        
        self.frames.append(frame.copy())
        self.frame_indices.append(frame_index)
    
    def get_frame(self, frame_index: int) -> np.ndarray:
        """Get frame by index from buffer"""
        try:
            buffer_index = self.frame_indices.index(frame_index)
            return self.frames[buffer_index]
        except ValueError:
            return None
    
    def get_recent_frames(self, count: int = 10) -> List[Tuple[int, np.ndarray]]:
        """Get most recent frames"""
        recent_count = min(count, len(self.frames))
        return list(zip(
            self.frame_indices[-recent_count:],
            self.frames[-recent_count:]
        ))
    
    def clear(self):
        """Clear buffer"""
        self.frames.clear()
        self.frame_indices.clear()
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.frames)
