"""
TensorFlow implementation of MTCNN face detection/alignment algorithm.
Updated for TensorFlow 2.x+
"""
import os
import numpy as np
import tensorflow as tf
import cv2
import joblib
from typing import List, Tuple, Dict, Optional, Union

class Network(tf.keras.Model):
    """Base network class for MTCNN components."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers_dict = {}
        
    def load_weights(self, data_path: str, ignore_missing: bool = False) -> None:
        """Load network weights from numpy file."""
        if os.path.exists(data_path):  # First checks the local filesystem
            return joblib.load(data_path)
        
        # If no file is found, raise an error
        raise FileNotFoundError(f"Weights file '{data_path}' not found in the system or in the package assets.")

    def conv(self, 
             inputs: tf.Tensor,
             filters: int,
             kernel_size: Tuple[int, int],
             strides: Tuple[int, int] = (1, 1),
             padding: str = 'SAME',
             name: str = None,
             activation: bool = True,
             use_bias: bool = True) -> tf.Tensor:
        """Convolutional layer wrapper."""
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=name
        )(inputs)
        if activation:
            conv = tf.nn.relu(conv)
        return conv

    def prelu(self, inputs: tf.Tensor, name: str = None) -> tf.Tensor:
        """Parametric ReLU layer."""
        alpha = self.add_weight(
            name=f'{name}/alpha',
            shape=(int(inputs.shape[-1]),),
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        return tf.maximum(0.0, inputs) + alpha * tf.minimum(0.0, inputs)

    def max_pool(self, 
                inputs: tf.Tensor,
                pool_size: Tuple[int, int],
                strides: Tuple[int, int],
                padding: str = 'SAME',
                name: str = None) -> tf.Tensor:
        """Max pooling layer wrapper."""
        return tf.keras.layers.MaxPool2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            name=name
        )(inputs)

    def dense(self, 
             inputs: tf.Tensor,
             units: int,
             name: str = None,
             activation: bool = True) -> tf.Tensor:
        """Fully connected layer wrapper."""
        if len(inputs.shape) > 2:
            inputs = tf.keras.layers.Flatten()(inputs)
        
        dense = tf.keras.layers.Dense(
            units=units,
            name=name,
            activation='relu' if activation else None
        )(inputs)
        return dense

    def softmax(self, inputs: tf.Tensor, axis: int = -1, name: str = None) -> tf.Tensor:
        """Softmax activation wrapper."""
        return tf.keras.layers.Softmax(axis=axis, name=name)(inputs)

class PNet(Network):
    """Proposal Network (PNet) for MTCNN."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()
        
    def build_network(self):
        inputs = tf.keras.Input(shape=(None, None, 3), name='input')
        
        # Network architecture
        x = self.conv(inputs, 10, (3, 3), name='conv1')
        x = self.prelu(x, name='PReLU1')
        x = self.max_pool(x, (2, 2), (2, 2), name='pool1')
        
        x = self.conv(x, 16, (3, 3), name='conv2')
        x = self.prelu(x, name='PReLU2')
        
        x = self.conv(x, 32, (3, 3), name='conv3')
        x = self.prelu(x, name='PReLU3')
        
        # Output branches
        prob = self.conv(x, 2, (1, 1), activation=False, name='conv4-1')
        prob = self.softmax(prob, axis=3, name='prob1')
        
        bbox = self.conv(x, 4, (1, 1), activation=False, name='conv4-2')
        
        self._model = tf.keras.Model(inputs=inputs, outputs=[bbox, prob], name='PNet')
    
    def call(self, inputs):
        return self._model(inputs)

class RNet(Network):
    """Refinement Network (RNet) for MTCNN."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()
        
    def build_network(self):
        inputs = tf.keras.Input(shape=(24, 24, 3), name='input')
        
        # Network architecture
        x = self.conv(inputs, 28, (3, 3), name='conv1')
        x = self.prelu(x, name='prelu1')
        x = self.max_pool(x, (3, 3), (2, 2), name='pool1')
        
        x = self.conv(x, 48, (3, 3), name='conv2')
        x = self.prelu(x, name='prelu2')
        x = self.max_pool(x, (3, 3), (2, 2), padding='VALID', name='pool2')
        
        x = self.conv(x, 64, (2, 2), name='conv3')
        x = self.prelu(x, name='prelu3')
        
        x = self.dense(x, 128, name='conv4')
        x = self.prelu(x, name='prelu4')
        
        # Output branches
        prob = self.dense(x, 2, activation=False, name='conv5-1')
        prob = self.softmax(prob, axis=1, name='prob1')
        
        bbox = self.dense(x, 4, activation=False, name='conv5-2')
        
        self._model = tf.keras.Model(inputs=inputs, outputs=[bbox, prob], name='RNet')
    
    def call(self, inputs):
        return self._model(inputs)

class ONet(Network):
    """Output Network (ONet) for MTCNN."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()
        
    def build_network(self):
        inputs = tf.keras.Input(shape=(48, 48, 3), name='input')
        
        # Network architecture
        x = self.conv(inputs, 32, (3, 3), name='conv1')
        x = self.prelu(x, name='prelu1')
        x = self.max_pool(x, (3, 3), (2, 2), name='pool1')
        
        x = self.conv(x, 64, (3, 3), name='conv2')
        x = self.prelu(x, name='prelu2')
        x = self.max_pool(x, (3, 3), (2, 2), padding='VALID', name='pool2')
        
        x = self.conv(x, 64, (3, 3), name='conv3')
        x = self.prelu(x, name='prelu3')
        x = self.max_pool(x, (2, 2), (2, 2), name='pool3')
        
        x = self.conv(x, 128, (2, 2), name='conv4')
        x = self.prelu(x, name='prelu4')
        
        x = self.dense(x, 256, name='conv5')
        x = self.prelu(x, name='prelu5')
        
        # Output branches
        prob = self.dense(x, 2, activation=False, name='conv6-1')
        prob = self.softmax(prob, axis=1, name='prob1')
        
        bbox = self.dense(x, 4, activation=False, name='conv6-2')
        landmarks = self.dense(x, 10, activation=False, name='conv6-3')
        
        self._model = tf.keras.Model(inputs=inputs, outputs=[bbox, landmarks, prob], name='ONet')
    
    def call(self, inputs):
        return self._model(inputs)

def create_mtcnn(model_path: str = None) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Create MTCNN networks with weights loaded."""
    if not model_path:
        model_path = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize networks
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    
    # Load weights
    pnet.load_weights(os.path.join(model_path, 'pnet.lz4'))
    rnet.load_weights(os.path.join(model_path, 'rnet.lz4'))
    onet.load_weights(os.path.join(model_path, 'onet.lz4'))
    
    return pnet, rnet, onet

# ---------------------------- Utility Functions ----------------------------

def bbreg(boundingbox: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """Calibrate bounding boxes."""
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox

def generateBoundingBox(imap: np.ndarray, 
                       reg: np.ndarray, 
                       scale: float, 
                       t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Use heatmap to generate bounding boxes."""
    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1, dy1, dx2, dy2 = [np.flipud(arr) for arr in [dx1, dy1, dx2, dy2]]
    
    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
    
    if reg.size == 0:
        reg = np.empty((0, 3))
    
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg

def nms(boxes: np.ndarray, 
        threshold: float, 
        method: str = 'Union') -> np.ndarray:
    """Non-maximum suppression."""
    if boxes.size == 0:
        return np.empty((0, 3))
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        
        I = I[np.where(o <= threshold)]
    
    pick = pick[0:counter]
    return pick

def pad(total_boxes: np.ndarray, 
        w: int, 
        h: int) -> Tuple[np.ndarray, ...]:
    """Compute padding coordinates."""
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w
    
    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

def rerec(bboxA: np.ndarray) -> np.ndarray:
    """Convert bounding boxes to square."""
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA

def imresample(img: np.ndarray, sz: Tuple[int, int]) -> np.ndarray:
    """Resize image using OpenCV."""
    return cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)


def detect_face(self, image, fit_to_image=True, limit_boundaries_landmarks=False, box_format="xywh", output_type="json",
                postprocess=True, batch_stack_justification="center", **kwargs):
    """
    Runs face detection on a single image or batch of images through the configured stages.

    Args:
        image (str, bytes, np.ndarray or tf.Tensor or list): The input image or batch of images.
                                                            It can be a file path, a tensor, or raw bytes.
        fit_to_image (bool, optional): Whether to fit bounding boxes and landmarks within image boundaries. Default is True.
        limit_boundaries_landmarks (bool, optional): Whether to ensure landmarks stay within image boundaries. Default is False.            
        box_format (str, optional): The format of the bounding box. Can be "xywh" for [X1, Y1, width, height] or "xyxy" for [X1, Y1, X2, Y2]. 
                                    Default is "xywh".
        output_type (str, optional): The output format. Can be "json" for dictionary output or "numpy" for numpy array output. Default is "json".
        postprocess (bool, optional): Flag to enable postprocessing. The postprocessing includes functionality affected by `fit_to_image`, 
                                    `limit_boundaries_landmarks` and removing padding effects caused by batching images with different shapes.
        batch_stack_justification (str, optional): The justification of the smaller images w.r.t. the largest images when 
                                                stacking in batch processing, which requires padding smaller images to the size of the 
                                                biggest one. 
        **kwargs: Additional parameters passed to the stages. The following parameters are used:

            - **StagePNet**:
                - min_face_size (int, optional): The minimum size of a face to detect. Default is 20.
                - min_size (int, optional): The minimum size to start the image pyramid. Default is 12.
                - scale_factor (float, optional): The scaling factor for the image pyramid. Default is 0.709.
                - threshold_pnet (float, optional): The confidence threshold for proposals from PNet. Default is 0.6.
                - nms_pnet1 (float, optional): The IoU threshold for the first round of NMS per scale. Default is 0.5.
                - nms_pnet2 (float, optional): The IoU threshold for the second round of NMS across all scales. Default is 0.7.
                
            - **StageRNet**:
                - threshold_rnet (float, optional): Confidence threshold for RNet proposals. Default is 0.7.
                - nms_rnet (float, optional): IoU threshold for Non-Maximum Suppression in RNet. Default is 0.7.

            - **StageONet**:
                - threshold_onet (float, optional): Confidence threshold for ONet proposals. Default is 0.8.
                - nms_onet (float, optional): IoU threshold for Non-Maximum Suppression in ONet. Default is 0.7.
        
    Returns:
        list or list of lists: A list of detected faces (in case a single image) or a list of lists of detected faces 
                            (one per image in the batch). If the stages are `face_and_landmarks_detection`, 
                            the output will have the detected faces and landmarks in JSON format. 
                            In case of `face_detection_only`, only the bounding boxes will be provided in 
                            JSON format.
    """
    return_tensor = output_type == "numpy"
    as_width_height = box_format == "xywh"

    is_batch = isinstance(image, list)
    images = image if is_batch else [image]

    with tf.device(self._device):
        # Load the images into memory and normalize them into a single tensor
        try:
            images_raw = load_images_batch(images)
            images_normalized, images_oshapes, pad_param = standarize_batch(images_raw,
                                                                            justification=batch_stack_justification,
                                                                            normalize=True)

            bboxes_batch = None

            # Process images through each stage (PNet, RNet, ONet)
            for stage in self.stages:
                bboxes_batch = stage(bboxes_batch=bboxes_batch, images_normalized=images_normalized, images_oshapes=images_oshapes, **kwargs)

        except tf.errors.InvalidArgumentError:  # No faces found
            bboxes_batch = np.empty((0, 16))
            pad_param = None

        if postprocess and pad_param is not None:
            # Adjust bounding boxes and landmarks to account for padding offsets
            bboxes_batch = fix_bboxes_offsets(bboxes_batch, pad_param)

            # Optionally, limit the bounding boxes and landmarks to stay within image boundaries
            if fit_to_image:
                bboxes_batch = limit_bboxes(bboxes_batch, images_shapes=images_oshapes, limit_landmarks=limit_boundaries_landmarks)

        # Convert bounding boxes and landmarks to JSON format if required
        if return_tensor:
            result = bboxes_batch

            if as_width_height:
                result[:, 3] = result[:, 3] - result[:, 1]
                result[:, 4] = result[:, 4] - result[:, 2]

        else:
            result = to_json(bboxes_batch,
                            images_count=len(images),
                            output_as_width_height=as_width_height,
                            input_as_width_height=False)
            result = result[0] if (not is_batch and len(result) > 0) else result

    return result