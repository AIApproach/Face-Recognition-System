# coding=utf-8
"""Performs face detection in realtime with modern Python and OpenCV."""
import os
import argparse
import sys
import time
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import face


class FaceVisualizer:
    """Handles visualization of detected faces with bounding boxes and labels."""
    
    def __init__(self, num_colors: int = 100):
        self.color_map = ColorMap(num_colors)
        self.font_path = self._get_font_path()
        
    @staticmethod
    def _get_font_path() -> Optional[str]:
        """Try to find a suitable font file."""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "SourceHanSansCN-Medium.otf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def visualize(self, frame: np.ndarray, faces: List['face.Face'], fps: int = 0) -> np.ndarray:
        """Draw bounding boxes and labels on detected faces with FPS counter."""
        if len(frame.shape) == 2:  # Convert grayscale to color
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Update color map with new face names
        labels = [f.name for f in faces if f.name is not None]
        self.color_map.update(labels)
        
        for face in faces:
            self._draw_face(draw, face)
        
        # Add FPS counter
        self._draw_fps(draw, fps, frame.shape[1])
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _draw_face(self, draw: ImageDraw.Draw, face: 'face.Face') -> None:
        """Draw a single face's bounding box and label."""
        xmin, ymin, xmax, ymax = face.bounding_box.astype(int)
        color = self.color_map[face.name or "unknown"]
        
        # Dynamic font sizing
        face_width = xmax - xmin
        font_size = max(face_width // 6, 10)
        
        try:
            font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Prepare label text
        label = f"{face.name or 'Unknown'}" + (f" {face.score:.2f}" if face.score else "")

        th = sum(font.getmetrics())
        left, top, right, bottom = font.getbbox(label)

        tw = right - left

        start_y = max(0, ymin - th)
        
        # Draw background rectangle
        draw.rectangle(
            [(xmin, start_y), (xmin + tw + 1, start_y + th)],
            fill=color
        )
        
        # Draw text
        draw.text(
            (xmin + 1, start_y),
            label,
            fill=(255, 255, 255),
            font=font,
            anchor="la"
        )
        
        # Draw bounding box
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline=color,
            width=2
        )
    
    @staticmethod
    def _draw_fps(draw: ImageDraw.Draw, fps: int, frame_width: int) -> None:
        """Draw FPS counter in top-right corner."""
        fps_text = f"FPS: {fps}"
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), fps_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        
        draw.text(
            (frame_width - text_width - 10, 10),
            fps_text,
            fill=(0, 255, 0),
            font=font
        )


class ColorMap:
    """Manages color assignments for face bounding boxes."""
    def __init__(self, num_classes: int = 100):
        self.color_list = self._generate_color_map(num_classes)
        self.color_map: Dict[str, Tuple[int, int, int]] = {}
        self.ptr = 0

    def __getitem__(self, key: str) -> Tuple[int, int, int]:
        return self.color_map.get(key, (255, 0, 0))  # Default to red if key not found

    def update(self, keys: List[str]) -> None:
        """Update color map with new face labels."""
        for key in keys:
            if key not in self.color_map:
                i = self.ptr % len(self.color_list)
                self.color_map[key] = self.color_list[i]
                self.ptr += 1

    @staticmethod
    def _generate_color_map(num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        color_map = num_classes * [0, 0, 0]
        for i in range(num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        return [tuple(color_map[i:i + 3]) for i in range(0, len(color_map), 3)]


def main(args):
    # Configuration
    frame_interval = 3  # Process every Nth frame
    fps_update_interval = 1.0  # Update FPS counter every second
    frame_rate = 0
    frame_count = 0
    
    # Initialize components
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video capture")
        return
    
    face_recognition = face.Recognition()
    visualizer = FaceVisualizer()
    
    # Timing variables
    last_fps_update = time.time()
    start_time = time.time()
    
    if args.debug:
        print("Debug mode enabled")
        face.debug = True
    
    try:
        while True:
            # Capture frame
            ret, frame = video_capture.read()
            if not ret:
                print("Warning: Could not read frame")
                break
            
            # Process frame
            current_time = time.time()
            if frame_count % frame_interval == 0:
                faces = face_recognition.identify(frame)
            
            # Update FPS counter
            if current_time - last_fps_update >= fps_update_interval:
                frame_rate = int(frame_count / (current_time - start_time))
                frame_count = 0
                start_time = current_time
                last_fps_update = current_time
            
            # Visualize results
            frame = visualizer.visualize(frame, faces, frame_rate)
            cv2.imshow('Face Recognition', frame)
            
            frame_count += 1
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # 27 is ESC
                break
                
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Real-time face recognition')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))