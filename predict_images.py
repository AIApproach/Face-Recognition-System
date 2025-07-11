"""Performs face detection in images."""
# MIT License

# Copyright (c) 2025 Tareq Al-Kushari

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

import os
import argparse
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple

import face

# Configure output directory
output_dir = 'output/'
os.makedirs(output_dir, exist_ok=True)

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

def draw_boxes(img: np.ndarray, faces: List['face.Face']) -> np.ndarray:
    """Draw bounding boxes and labels on detected faces.
    
    Args:
        img: Input image as numpy array
        faces: List of detected Face objects
        
    Returns:
        Image with drawn boxes and labels as numpy array
    """
    # Initialize color map and labels
    labels = [face.name for face in faces]
    color_map = ColorMap(100)
    color_map.update(labels)
    
    # Try to load font - fallback to default if not found
    try:
        font_path = os.path.join(os.path.dirname(__file__), "SourceHanSansCN-Medium.otf")
        font = ImageFont.truetype(font_path, 10)  # Initial size, will be adjusted
    except:
        font = ImageFont.load_default()
    
    # Convert to PIL Image for drawing
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    for face in faces:
        # Get face bounding box coordinates
        xmin, ymin, xmax, ymax = face.bounding_box.astype(int)
        color = color_map[face.name]
        
        # Calculate appropriate font size based on face size
        font_size = max(int((xmax - xmin) // 6), 10)
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Prepare label text
        text = f"{face.name} {face.score:.4f}"

        th = sum(font.getmetrics())

        left, top, right, bottom = font.getbbox(text)

        tw = right - left
        
        # Calculate text position
        # start_y = max(0, ymin - text_height)
        start_y = max(0, ymin - th)
        
        # Draw background rectangle and text
        # draw.rectangle(
        #     [(xmin, start_y), (xmin + text_width + 1, start_y + text_height)],
        #     fill=color
        # )
        draw.rectangle(
            [(xmin, start_y), (xmin + tw + 1, start_y + th)],
            fill=color
        )
        draw.text(
            (xmin + 1, start_y),
            text,
            fill=(255, 255, 255),
            font=font,
            anchor="la"
        )
        
        # Draw face bounding box
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            width=2,
            outline=color
        )
    
    return np.array(pil_img)

def process_image(image_path: str, debug: bool = False) -> None:
    """Process a single image for face detection and recognition.
    
    Args:
        image_path: Path to input image file
        debug: Whether to enable debug outputs
    """
    # Prepare output path
    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, file_name)
    
    # Load image
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Initialize face recognition
    face_recognition = face.Recognition()
    if debug:
        print("Debug mode enabled")
        face.debug = True
    
    # Detect and recognize faces
    faces = face_recognition.identify(img_array)
    
    # Draw bounding boxes and save results
    if faces:
        image_with_boxes = draw_boxes(img_array, faces)
        
        # Save and display results
        cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        cv2.imshow('Face Detection Results', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No faces detected in the image.")

def main(args):
    """Main function to handle command line arguments and execution."""
    process_image(args.image_files, args.debug)

def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform face detection and recognition on images.'
    )
    parser.add_argument(
        'image_files',
        help='Path to the input image'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug outputs'
    )
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))