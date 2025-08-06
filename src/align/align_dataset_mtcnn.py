"""
This is the align_dataset_mtcnn.py file.
Face alignment using MTCNN with TensorFlow 2.x
Performs face detection, alignment and stores face thumbnails in the output directory.
"""
# MIT License

# Copyright (c) 2025 Tareq Al-Kushari

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

import os
import sys
import cv2
import argparse
import random
import numpy as np
from time import sleep
from PIL import Image
import tensorflow as tf

# Add parent directory to path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

from mtcnn.mtcnn import MTCNN
import facenet

def main(args):
    sleep(random.random())  # Random delay to avoid race conditions
    
    # Setup output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Store revision info
    src_path = os.path.dirname(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    
    # Load dataset
    dataset = facenet.get_dataset(args.input_dir)
    
    print('Creating MTCNN detector')
    
    # Configure GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if args.gpu_memory_fraction < 1.0:
                memory_limit = args.gpu_memory_fraction * gpus[0].memory_limit
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")

    # Initialize MTCNN detector
    detector = MTCNN()

    # Create bounding boxes file
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, f'bounding_boxes_{random_key:05d}.txt')
    
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        
        if args.random_order:
            random.shuffle(dataset)
            
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            if args.random_order:
                random.shuffle(cls.image_paths)
                
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = os.path.join(output_class_dir, f'{filename}.png')
                print(f"Processing: {image_path}")
                
                if not os.path.exists(output_filename):
                    try:
                        img = np.array(Image.open(image_path))
                    except (IOError, ValueError, IndexError) as e:
                        print(f"Error loading {image_path}: {e}")
                        text_file.write(f"{output_filename}\n")
                        continue
                        
                    # Convert to RGB if needed
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    elif img.shape[2] == 4:
                        img = img[:, :, :3]
                        
                    # Detect faces
                    try:
                        detections = detector.detect_faces(img)
                    except Exception as e:
                        print(f"Face detection failed on {image_path}: {e}")
                        text_file.write(f"{output_filename}\n")
                        continue
                        
                    if detections:
                        for i, detection in enumerate(detections):
                            if detection['confidence'] < args.min_confidence:
                                continue
                                
                            # Get bounding box with margin
                            bb = detection['box']
                            x, y, width, height = bb
                            
                            x1 = max(0, x - args.margin // 2)
                            y1 = max(0, y - args.margin // 2)
                            x2 = min(img.shape[1], x + width + args.margin // 2)
                            y2 = min(img.shape[0], y + height + args.margin // 2)
                            
                            # Crop and resize face
                            cropped = img[y1:y2, x1:x2]
                            if cropped.size == 0:
                                continue
                                
                            scaled = cv2.resize(
                                cropped, 
                                (args.image_size, args.image_size),
                                interpolation=cv2.INTER_LINEAR
                            )
                            
                            # Save aligned face
                            if args.detect_multiple_faces:
                                output_filename_n = f"{os.path.splitext(output_filename)[0]}_{i}.png"
                            else:
                                output_filename_n = output_filename
                                
                            Image.fromarray(scaled).save(output_filename_n)
                            nrof_successfully_aligned += 1
                            
                            # Write bounding box info
                            text_file.write(
                                f"{output_filename_n} {x1} {y1} {x2} {y2}\n"
                            )
                            
                            # Only process one face if not detecting multiple
                            if not args.detect_multiple_faces:
                                break
                    else:
                        print(f"No faces detected in {image_path}")
                        text_file.write(f"{output_filename}\n")
    
    print(f"\nAlignment complete:")
    print(f"Total images processed: {nrof_images_total}")
    print(f"Successfully aligned: {nrof_successfully_aligned}")
    print(f"Bounding boxes saved to: {bounding_boxes_filename}")

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Align faces in images using MTCNN detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory with unaligned images'
    )
    parser.add_argument(
        'output_dir', 
        type=str,
        help='Directory to save aligned face thumbnails'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        help='Size of aligned face images (height, width) in pixels',
        default=160
    )
    parser.add_argument(
        '--margin',
        type=int,
        help='Margin to add to bounding box (height, width) in pixels',
        default=44
    )
    parser.add_argument(
        '--min_confidence',
        type=float,
        help='Minimum confidence threshold for face detection',
        default=0.9
    )
    parser.add_argument(
        '--random_order',
        help='Shuffle image order for parallel processing',
        action='store_true'
    )
    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        help='Fraction of GPU memory to allocate',
        default=1.0
    )
    parser.add_argument(
        '--detect_multiple_faces',
        help='Detect and align multiple faces per image',
        action='store_true'
    )
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))