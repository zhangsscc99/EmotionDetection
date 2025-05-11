"""
Data loading and preprocessing module.
Handles loading images from directories and applying appropriate augmentations.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT


def create_data_generators(train_dir, test_dir):
    """
    Create data generators for training, validation, and test sets.
    
    Args:
        train_dir: Directory containing training data
        test_dir: Directory containing test data
        
    Returns:
        train_generator, validation_generator, test_generator
    """
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
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
    
    return train_generator, validation_generator, test_generator


def get_class_weights(train_generator):
    """
    Calculate class weights to handle class imbalance.
    
    Args:
        train_generator: Training data generator
        
    Returns:
        Dictionary of class weights
    """
    class_counts = train_generator.classes
    total_samples = len(class_counts)
    n_classes = len(train_generator.class_indices)
    
    # Count number of samples in each class
    class_totals = np.bincount(class_counts, minlength=n_classes)
    
    # Calculate class weights
    class_weights = {i: total_samples / (n_classes * count) if count > 0 else 1.0 
                     for i, count in enumerate(class_totals)}
    
    return class_weights 