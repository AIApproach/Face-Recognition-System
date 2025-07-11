# coding=utf-8
"""Real-time face detection with attendance recording."""
import os
import argparse
import sys
import time
import colorsys
import csv
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional

import face  # Assuming this is your face recognition module

class AttendanceLogger:
    """Handles attendance recording to CSV file."""
    
    def __init__(self, filename: str = "attendance.csv"):
        self.filename = filename
        self.last_recognition_time: Dict[str, float] = {}
        self.cooldown = 30  # Seconds between recordings for same person
        self._ensure_csv_header()
        
    def _ensure_csv_header(self):
        """Ensure CSV file exists with proper header."""
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Timestamp", "Confidence"])
    
    def record_attendance(self, name: str, confidence: float):
        """Record attendance if not recently recorded."""
        current_time = time.time()
        
        # Skip if this person was recently recorded
        if name in self.last_recognition_time:
            if current_time - self.last_recognition_time[name] < self.cooldown:
                return False
        
        # Record the attendance
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp, f"{confidence:.2f}"])
        
        self.last_recognition_time[name] = current_time
        return True

class FaceVisualizer:
    """Handles visualization of faces and attendance status."""
    
    def __init__(self):
        self.color_map = ColorMap()
        self.font = self._load_font()
        self.attendance_font = ImageFont.load_default()
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.fps = 0
        self.attendance_status = ""
        self.attendance_status_time = 0
        
    def _load_font(self):
        """Try to load a nice font with fallback to default."""
        font_paths = [
            os.path.join(os.path.dirname(__file__), "SourceHanSansCN-Medium.otf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, 12)
                except:
                    continue
        return ImageFont.load_default()
        
    def visualize(self, frame: np.ndarray, faces: List['face.Face'], attendance_logger: AttendanceLogger) -> np.ndarray:
        """Draw faces, info, and attendance status on the frame."""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Update FPS counter
        self._update_fps()
        
        # Draw each face
        for face in faces:
            self._draw_face(draw, face, attendance_logger)
            
        # Draw FPS and other info
        self._draw_info(draw, frame.shape[1])
        
        # Draw attendance status if recent
        if time.time() - self.attendance_status_time < 3:  # Show for 3 seconds
            self._draw_attendance_status(draw, frame.shape)
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _draw_face(self, draw: ImageDraw.Draw, face: 'face.Face', attendance_logger: AttendanceLogger):
        """Draw bounding box and label for a single face."""
        x1, y1, x2, y2 = face.bounding_box.astype(int)
        color = self.color_map[face.name]
        
        # Dynamic font sizing based on face size
        face_size = x2 - x1
        font_size = max(face_size // 10, 10)
        
        try:
            font = self.font.font_variant(size=font_size)
        except:
            font = ImageFont.load_default()
        
        # Prepare label text
        label = f"{face.name or 'Unknown'}"
        if face.score is not None:
            label += f" {face.score:.2f}"
            
        # Get text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        th = sum(font.getmetrics())
        left, top, right, bottom = font.getbbox(label)

        tw = right - left

        start_y = max(0, y1 - th)
        
        # Draw background and text
        draw.rectangle(
            [(x1, start_y), (x1 + tw + 1, start_y + th)],
            fill=color
        )
        draw.text(
            (x1 + 1, start_y),
            label,
            fill=(255, 255, 255),
            font=font
        )
        
        # Draw bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=2
        )
        
        # Record attendance if recognized
        if face.name and face.name != "Unknown":
            if attendance_logger.record_attendance(face.name, face.score or 0):
                self.attendance_status = f"Recorded: {face.name}"
                self.attendance_status_time = time.time()
    
    def _update_fps(self):
        """Calculate and update FPS every second."""
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = now
    
    def _draw_info(self, draw: ImageDraw.Draw, frame_width: int):
        """Draw FPS and other info on frame."""
        info_text = f"FPS: {self.fps:.1f}"
        text_bbox = draw.textbbox((0, 0), info_text, font=self.font)
        text_w = text_bbox[2] - text_bbox[0]
        
        draw.text(
            (frame_width - text_w - 10, 10),
            info_text,
            fill=(0, 255, 0),
            font=self.font
        )
    
    def _draw_attendance_status(self, draw: ImageDraw.Draw, frame_shape: Tuple[int, int]):
        """Draw attendance recording status at bottom of frame."""
        text_bbox = draw.textbbox((0, 0), self.attendance_status, font=self.attendance_font)
        text_w = text_bbox[2] - text_bbox[0]
        x = (frame_shape[1] - text_w) // 2  # Center horizontally
        y = frame_shape[0] - 30  # Near bottom
        
        draw.rectangle(
            [(x - 5, y - 5), (x + text_w + 5, y + text_bbox[3] - text_bbox[1] + 5)],
            fill=(0, 0, 0)
        )
        draw.text(
            (x, y),
            self.attendance_status,
            fill=(0, 255, 0),
            font=self.attendance_font
        )

class ColorMap:
    """Manages distinct colors for face bounding boxes."""
    
    def __init__(self, num_colors: int = 100):
        self.colors = self._generate_distinct_colors(num_colors)
        self.assigned_colors: Dict[str, Tuple[int, int, int]] = {}
        self.next_color = 0
    
    def __getitem__(self, name: Optional[str]) -> Tuple[int, int, int]:
        """Get color for a name, assigning new one if needed."""
        if name not in self.assigned_colors:
            self.assigned_colors[name] = self.colors[self.next_color % len(self.colors)]
            self.next_color += 1
        return self.assigned_colors[name]
    
    @staticmethod
    def _generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors using HSV color space."""
        colors = []
        golden_ratio = 0.618033988749895
        h = 0.0
        
        for _ in range(n):
            h += golden_ratio
            h %= 1
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.95)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

def main(args):
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize components
    recognizer = face.Recognition()
    visualizer = FaceVisualizer()
    attendance_logger = AttendanceLogger()
    
    # Frame processing parameters
    process_every_n_frames = 3
    frame_counter = 0
    faces = []
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Warning: Couldn't read frame")
                break
            
            # Process frame (skip some frames for better performance)
            if frame_counter % process_every_n_frames == 0:
                faces = recognizer.identify(frame)
            
            # Visualize results
            frame = visualizer.visualize(frame, faces, attendance_logger)
            cv2.imshow('Face Recognition - Attendance', frame)
            
            frame_counter += 1
            
            # Exit on 'q' or ESC
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Attendance records saved to {attendance_logger.filename}")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--output', type=str, default="attendance.csv",
                       help='Filename for attendance records')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)