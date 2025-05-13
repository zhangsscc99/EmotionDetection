"""
Implementation of the 5-layer CNN model from the paper:
'An Efficient Approach to Face Emotion Recognition with Convolutional Neural Networks'
by Christian Białek, Andrzej Matiolański, and Michał Grega.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, LEARNING_RATE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, Dense, Flatten, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam


def build_five_layer_model():
    """
    Build and compile the 5-layer CNN model for emotion recognition
    based on the paper architecture.
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Conv layer 1 (48x48x128)
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 2 (24x24x256)
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 3 (12x12x512)
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 4 (6x6x1024)
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 5 (3x3x1024)
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flatten layer
    model.add(Flatten())
    
    # Dense Layer 1 (512)
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Dense Layer 2 (256)
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_six_layer_model():
    """
    Build and compile the 6-layer CNN model for emotion recognition
    based on the paper architecture.
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Conv layer 1 (48x48x128)
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 2 (24x24x256)
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 3 (12x12x512)
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 4 (6x6x1024)
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 5 (3x3x1024)
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Conv layer 6 (1x1x2048)
    model.add(Conv2D(2048, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Global Average Pooling
    model.add(GlobalAveragePooling2D())
    
    # Output Layer
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 