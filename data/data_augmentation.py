"""
Enhanced data augmentation module based on the paper:
'An Efficient Approach to Face Emotion Recognition with Convolutional Neural Networks'

This module provides functions for dataset balancing and augmentation
as described in the paper.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT


def create_balanced_generators(train_dir, test_dir, target_samples_per_class=None):
    """
    Create data generators with balanced classes as described in the paper.
    
    Args:
        train_dir: Directory containing training data
        test_dir: Directory containing test data
        target_samples_per_class: Target number of samples per class after balancing
                                 (if None, use 8000 as in the paper)
        
    Returns:
        train_generator, validation_generator, test_generator
    """
    if target_samples_per_class is None:
        target_samples_per_class = 8000  # As used in the paper
    
    # Data augmentation for training set following the paper's approach
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=10,           # Paper uses rotation between -10 and 10 degrees
        width_shift_range=0.1,       # Additional augmentation
        height_shift_range=0.1,      # Additional augmentation
        zoom_range=(1.1, 1.2),       # Paper uses zoom between 1.1 and 1.2
        horizontal_flip=True,        # Paper uses horizontal mirroring
        validation_split=VALIDATION_SPLIT
    )
    
    # Only rescale for testing set
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',  
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',  
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale', 
        class_mode='categorical',
        shuffle=False
    )
    
    # Create a balanced train generator that applies more augmentation to smaller classes
    balanced_train_generator = create_balanced_generator(
        train_generator, 
        target_samples_per_class=target_samples_per_class
    )
    
    return balanced_train_generator, validation_generator, test_generator


def create_balanced_generator(generator, target_samples_per_class):
    """
    Balance a generator by applying more augmentation to underrepresented classes.
    
    Args:
        generator: Original data generator
        target_samples_per_class: Target number of samples per class
        
    Returns:
        Balanced generator
    """
    class_counts = np.bincount(generator.classes)
    n_classes = len(generator.class_indices)
    
    print(f"Original class distribution: {class_counts}")
    
    # Calculate augmentation factor needed for each class
    augmentation_factors = {}
    for class_idx in range(n_classes):
        orig_count = class_counts[class_idx]
        if orig_count == 0:
            augmentation_factors[class_idx] = 0
        else:
            factor = max(1, int(np.ceil(target_samples_per_class / orig_count)))
            augmentation_factors[class_idx] = factor
    
    print(f"Augmentation factors: {augmentation_factors}")
    
    # Custom generator function that yields balanced batches
    def balanced_generator():
        while True:
            # Initialize arrays for batch data and labels
            batch_x = []
            batch_y = []
            
            # Keep track of samples per class in current batch
            current_batch_class_counts = [0] * n_classes
            
            # Determine target samples per class in batch to maintain balance
            target_per_class_in_batch = BATCH_SIZE // n_classes
            
            while len(batch_x) < BATCH_SIZE:
                # Get a batch from the original generator
                x, y = next(generator)
                
                # Process each sample in the batch
                for i in range(len(x)):
                    # Get class index for this sample
                    class_idx = np.argmax(y[i])
                    
                    # If we have enough samples for this class in the current batch, skip
                    if current_batch_class_counts[class_idx] >= target_per_class_in_batch:
                        continue
                    
                    # Add sample to batch
                    batch_x.append(x[i])
                    batch_y.append(y[i])
                    current_batch_class_counts[class_idx] += 1
                    
                    # If batch is full, yield it
                    if len(batch_x) == BATCH_SIZE:
                        break
            
            # Convert to numpy arrays and yield
            yield np.array(batch_x), np.array(batch_y)
    
    # Create a tf.data.Dataset from the generator
    output_types = (tf.float32, tf.float32)
    output_shapes = ((None,) + IMG_SIZE + (1,), (None, n_classes))
    
    dataset = tf.data.Dataset.from_generator(
        balanced_generator,
        output_types=output_types,
        output_shapes=output_shapes
    )
    
    # Create a balanced generator that can be used like a Keras generator
    balanced_gen = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # Add necessary attributes to mimic Keras generator
    balanced_gen.batch_size = BATCH_SIZE
    balanced_gen.n = target_samples_per_class * n_classes
    balanced_gen.classes = generator.classes
    balanced_gen.class_indices = generator.class_indices
    balanced_gen.samples = balanced_gen.n
    
    return balanced_gen


def get_class_weights(train_generator):
    """
    Calculate class weights to handle class imbalance.
    
    Args:
        train_generator: Training data generator
        
    Returns:
        Dictionary of class weights
    """
    class_counts = np.bincount(train_generator.classes)
    total_samples = np.sum(class_counts)
    n_classes = len(train_generator.class_indices)
    
    # Calculate class weights
    class_weights = {i: total_samples / (n_classes * count) if count > 0 else 1.0 
                     for i, count in enumerate(class_counts)}
    
    return class_weights 