"""Helper for evaluation on the Labeled Faces in the Wild dataset."""
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
import numpy as np
from facenet import calculate_roc, calculate_val

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    """Evaluate embeddings on LFW dataset
    
    Args:
        embeddings: numpy array of shape [nrof_images, embedding_size]
        actual_issame: list of bool indicating whether pairs are matching
        nrof_folds: number of cross-validation folds
        distance_metric: 0 for Euclidean distance, 1 for cosine similarity
        subtract_mean: whether to subtract mean embedding before evaluation
    
    Returns:
        tpr: true positive rates
        fpr: false positive rates
        accuracy: accuracies for each fold
        val: validation rate at FAR=1e-3
        val_std: standard deviation of validation rate
        far: false accept rate
    """
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    
    # Calculate ROC curve
    tpr, fpr, accuracy = calculate_roc(
        thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        distance_metric=distance_metric,
        subtract_mean=subtract_mean
    )
    
    # Calculate validation metrics at FAR=1e-3
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(
        thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame),
        1e-3,
        nrof_folds=nrof_folds,
        distance_metric=distance_metric,
        subtract_mean=subtract_mean
    )
    
    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs):
    """Get paths to images and ground truth labels from pairs file
    
    Args:
        lfw_dir: path to LFW dataset directory
        pairs: numpy array of pairs from read_pairs()
    
    Returns:
        path_list: list of image paths
        issame_list: list of bool indicating whether pairs are matching
    """
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    
    for pair in pairs:
        if len(pair) == 3:
            # Matching pair
            path0 = add_extension(os.path.join(lfw_dir, pair[0], f"{pair[0]}_{int(pair[1]):04d}"))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], f"{pair[0]}_{int(pair[2]):04d}"))
            issame = True
        elif len(pair) == 4:
            # Non-matching pair
            path0 = add_extension(os.path.join(lfw_dir, pair[0], f"{pair[0]}_{int(pair[1]):04d}"))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], f"{pair[2]}_{int(pair[3]):04d}"))
            issame = False
        
        # Only add the pair if both paths exist
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    
    if nrof_skipped_pairs > 0:
        print(f'Skipped {nrof_skipped_pairs} image pairs')
    
    return path_list, issame_list

def add_extension(path):
    """Add image extension to path if file exists
    
    Args:
        path: path without extension
    
    Returns:
        path with .jpg or .png extension if file exists
    
    Raises:
        RuntimeError if neither file exists
    """
    for ext in ['.jpg', '.png']:
        if os.path.exists(path + ext):
            return path + ext
    raise RuntimeError(f'No file "{path}" with extension png or jpg.')

def read_pairs(pairs_filename):
    """Read pairs from text file
    
    Args:
        pairs_filename: path to pairs text file
    
    Returns:
        numpy array of pairs
    """
    pairs = []
    with open(pairs_filename, 'r') as f:
        # Skip header line
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)