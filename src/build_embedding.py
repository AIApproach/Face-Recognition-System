import os
import sys
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.image import img_to_array

# For MTCNN face detection (you might want to consider updating this to a newer face detection library)
from align.detect_face import detect_face

# Directory of labeled images
dataset_dir = 'dataset/my_dataset/all/'
image_size = 160  # For FaceNet

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def main(args):
    known_embeddings = []
    labels = []

    # Load the model using modern TensorFlow
    print('Loading feature extraction model')
    model = tf.keras.models.load_model(args.model)
    
    # Get the input and output tensors
    images_placeholder = model.inputs[0]
    embeddings = model.outputs[0]

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((image_size, image_size))  # Resize for FaceNet
            img_np = img_to_array(img)
            prewhitened = prewhiten(img_np)

            # Get the embedding (using modern TF inference)
            embedding = model.predict(np.expand_dims(prewhitened, axis=0))

            known_embeddings.append(embedding[0])  # shape (512,)
            labels.append(person_name)
    
    known_embeddings = np.array(known_embeddings)  # shape: (n_images, 512)
    labels = np.array(labels)  # shape: (n_images,)

    known_embeddings_filename = os.path.expanduser(args.known_embeddings_filename)

    with open(known_embeddings_filename, 'wb') as f:
        pickle.dump((known_embeddings, labels), f)
    print('Saved known_embeddings model to file "%s"' % known_embeddings_filename)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('known_embeddings_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))