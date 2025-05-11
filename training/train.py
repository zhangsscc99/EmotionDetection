"""
Training script for the facial emotion recognition model.
"""

import os
import sys
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAIN_DIR, TEST_DIR, EPOCHS, MODEL_PATH
from data.data_loader import create_data_generators, get_class_weights
from models.cnn_model import build_emotion_model
from models.model_utils import get_callbacks, model_summary_to_file


def train_model():
    """
    Train the facial emotion recognition model.
    
    Returns:
        Trained model and training history
    """
    print("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators(
        TRAIN_DIR, TEST_DIR
    )
    
    print("Building model...")
    model = build_emotion_model()
    model_summary_to_file(model)
    
    # Calculate class weights to handle imbalance
    class_weights = get_class_weights(train_generator)
    print("Class weights:", class_weights)
    
    # Get callbacks
    callbacks = get_callbacks()
    
    print("Starting training...")
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save training history
    history_dict = history.history
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f)
    
    print("Model and history saved.")
    
    return model, history


if __name__ == "__main__":
    train_model() 