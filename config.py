"""
Configuration file for the Facial Emotion Recognition project.
Contains paths, hyperparameters, and other configuration settings.
"""

import os

# Data paths
TRAIN_DIR = r"C:\Users\lyqtt\Downloads\archive\train"
TEST_DIR = r"C:\Users\lyqtt\Downloads\archive\test"

# Model parameters
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
NUM_CLASSES = 7
EPOCHS = 70

# Training parameters
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# Model checkpoint paths
MODEL_DIR = os.path.join("models", "saved")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")
# Ensure forward slashes for cross-platform compatibility
MODEL_PATH = MODEL_PATH.replace('\\', '/')

# Class mapping
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
} 