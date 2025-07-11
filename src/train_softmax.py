"""
Training a face recognizer with TensorFlow using softmax cross entropy loss
This is the train_softmax.py file.
"""
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
import time
import sys
import random
import datetime
import argparse
import math
import numpy as np
import h5py
import importlib
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

import facenet
import lfw

def main(args):
    # Set random seeds for reproducibility
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Create output directories
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # stat_file_name = os.path.join(log_dir, 'stat.h5')
    stat_file_name = os.path.join(log_dir, 'stat.npz')  # Changed to .npz format

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store git revision info
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    # Load dataset
    dataset = facenet.get_dataset(args.data_dir)
    if args.filter_filename:
        dataset = filter_dataset(dataset, os.path.expanduser(args.filter_filename), 
            args.filter_percentile, args.filter_min_nrof_images_per_class)
        
    if args.validation_set_split_ratio > 0.0:
        train_set, val_set = facenet.split_dataset(dataset, args.validation_set_split_ratio, 
                                                  args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []
        
    nrof_classes = len(train_set)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

    # Import the model definition
    network = importlib.import_module(args.model_def)
    image_size = (args.image_size, args.image_size)

    # Create the model
    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
    prelogits, _ = network.inference(inputs, args.keep_probability, 
                                    phase_train=True, 
                                    bottleneck_layer_size=args.embedding_size,
                                    weight_decay=args.weight_decay)
    
    logits = layers.Dense(len(train_set), 
                         activation=None, 
                         kernel_initializer='glorot_uniform',
                         kernel_regularizer=l2(args.weight_decay),
                         name='Logits')(prelogits)
    
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    # Create the model
    model = Model(inputs=inputs, outputs=[logits, embeddings])

    # Add center loss if specified
    if args.center_loss_factor > 0.0:
        from center_loss import CenterLoss
        center_loss_layer = CenterLoss(nrof_classes, args.embedding_size, args.center_loss_alfa)
        center_loss = center_loss_layer(prelogits, labels)
        model.add_loss(args.center_loss_factor * center_loss)

    # Add prelogits norm loss if specified
    if args.prelogits_norm_loss_factor > 0.0:
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=args.prelogits_norm_p, axis=1))
        model.add_loss(args.prelogits_norm_loss_factor * prelogits_norm)

    # Compile the model
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    model.compile(optimizer=optimizer,
                  loss={'Logits': losses.SparseCategoricalCrossentropy(from_logits=True)},
                  metrics={'Logits': metrics.SparseCategoricalAccuracy(name='accuracy')})

    # Load pretrained weights if specified
    if pretrained_model:
        print('Loading pretrained model: %s' % pretrained_model)
        model.load_weights(pretrained_model)

    # Create data pipelines
    train_image_list, train_label_list = facenet.get_image_paths_and_labels(train_set)
    val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)

    print('Number of classes in training set: %d' % nrof_classes)
    print('Number of examples in training set: %d' % len(train_image_list))
    print('Number of classes in validation set: %d' % len(val_set))
    print('Number of examples in validation set: %d' % len(val_image_list))

    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_image_list))
    train_dataset = train_dataset.map(lambda x, y: facenet.load_and_preprocess_image(
        x, y, image_size, args.random_rotate, args.random_crop, args.random_flip, 
        args.use_fixed_image_standardization), 
        num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Create validation dataset if needed
    if len(val_image_list) > 0:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
        val_dataset = val_dataset.map(lambda x, y: facenet.load_and_preprocess_image(
            x, y, image_size, False, False, False, args.use_fixed_image_standardization), 
            num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(args.batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        val_dataset = None

    # Callbacks
    # callbacks = [
    #     ModelCheckpoint(os.path.join(model_dir, 'model-{epoch:03d}.h5'), 
    #                    save_weights_only=True),
    #     TensorBoard(log_dir=log_dir),
    # ]

    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'model-{epoch:03d}.h5'),
            save_weights_only=True,
            monitor='val_accuracy' if val_dataset else 'accuracy',
            mode='max',
            save_best_only=True,
            save_freq='epoch'
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1 if args.log_histograms else 0,
            update_freq='epoch'
        ),
        tf.keras.callbacks.BackupAndRestore(os.path.join(model_dir, 'backup'))
    ]

    if args.learning_rate_schedule_file:
        lr_scheduler = LearningRateScheduler(
            lambda epoch: facenet.get_learning_rate_from_file(args.learning_rate_schedule_file, epoch+1))
        callbacks.append(lr_scheduler)

    # Training loop
    print('Starting training')
    # history = model.fit(
    #     train_dataset,
    #     epochs=args.max_nrof_epochs,
    #     callbacks=callbacks,
    #     validation_data=val_dataset,
    #     validation_freq=args.validate_every_n_epochs
    # )

    try:
        history = model.fit(
            train_dataset,
            epochs=args.max_nrof_epochs,
            callbacks=callbacks,
            validation_data=val_dataset,
            validation_freq=args.validate_every_n_epochs,
            verbose=1
        )
        
        # Save statistics safely
        try:
            np.savez(stat_file_name, **history.history)
        except Exception as e:
            print(f"Warning: Could not save training statistics: {e}")
            
    except Exception as e:
        print(f"Training failed: {e}")
        # Emergency save
        model.save_weights(os.path.join(model_dir, 'emergency_save.h5'))
        raise


    # Evaluate on LFW if specified
    if args.lfw_dir:
        print('Evaluating on LFW')
        evaluate_lfw(model, lfw_paths, actual_issame, args.lfw_batch_size, 
                    args.lfw_nrof_folds, args.lfw_distance_metric, 
                    args.lfw_subtract_mean, args.lfw_use_flipped_images)

    return model_dir

def get_optimizer(optimizer_name, learning_rate):
    """Get optimizer based on name"""
    if optimizer_name == 'ADAGRAD':
        return optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'ADADELTA':
        return optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer_name == 'ADAM':
        return optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSPROP':
        return optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'MOM':
        return optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError('Invalid optimization algorithm')

def evaluate_lfw(model, lfw_paths, actual_issame, batch_size, nrof_folds, distance_metric, subtract_mean, use_flipped_images):
    """Evaluate model on LFW dataset"""
    print('Running forward pass on LFW images')
    
    # Prepare embeddings
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    
    # Process images in batches
    embedding_size = model.output[1].shape[1]
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(0, nrof_images, batch_size):
        batch_paths = lfw_paths[i:i+batch_size]
        images = [facenet.load_and_preprocess_image(p, 0, (160, 160), False, False, False, True) for p in batch_paths]
        images = np.stack(images)
        _, embeddings = model.predict(images)
        emb_array[i:i+batch_size, :] = embeddings
        
        if use_flipped_images:
            # Add flipped versions
            flipped_images = np.flip(images, axis=2)  # Flip horizontally
            _, flipped_embeddings = model.predict(flipped_images)
            emb_array[i:i+batch_size, :] = (embeddings + flipped_embeddings) / 2
    
    # Evaluate
    _, _, accuracy, val, val_std, far = lfw.evaluate(
        emb_array, actual_issame, nrof_folds=nrof_folds, 
        distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    return accuracy, val, far

def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
        help='Number of epoch between validation', default=5)
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--lfw_distance_metric', type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--lfw_subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))