"""
Training script for the facial emotion recognition model.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAIN_DIR, TEST_DIR, EPOCHS, MODEL_PATH
from data.data_loader import create_data_generators, get_class_weights
from models.cnn_model import build_emotion_model
from models.model_utils import get_callbacks, model_summary_to_file


def ensure_dir_exists(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def convert_to_serializable(obj):
    """
    Convert NumPy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def train_model(start_epoch=0, checkpoint_path=None):
    """
    Train the facial emotion recognition model.
    
    Args:
        start_epoch: Epoch to start training from (for resuming training)
        checkpoint_path: Path to checkpoint file to resume training from
        
    Returns:
        Trained model and training history
    """
    # Ensure directories exist
    ensure_dir_exists(os.path.join("models", "saved"))
    ensure_dir_exists(os.path.join("models", "checkpoints"))
    ensure_dir_exists("logs")
    
    print("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators(
        TRAIN_DIR, TEST_DIR
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # Ensure checkpoint path uses forward slashes
        checkpoint_path = checkpoint_path.replace('\\', '/')
        model = tf.keras.models.load_model(checkpoint_path)
        print("Model loaded successfully.")
    else:
        print("Building new model...")
        model = build_emotion_model()
        
    model_summary_to_file(model)
    
    # Calculate class weights to handle imbalance
    class_weights = get_class_weights(train_generator)
    print("Class weights:", class_weights)
    
    # Get callbacks
    callbacks = get_callbacks()
    
    print(f"Starting training from epoch {start_epoch + 1}...")
    start_time = time.time()
    
    # Calculate initial epoch for resuming training
    initial_epoch = start_epoch
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save training history
    history_path = f'training_history_{time.strftime("%Y%m%d_%H%M%S")}.json'
    # Convert history to serializable format
    history_dict = history.history
    serializable_history = convert_to_serializable(history_dict)
    
    # If resuming training and history file exists, try to merge histories
    if start_epoch > 0 and os.path.exists('training_history.json'):
        try:
            with open('training_history.json', 'r') as f:
                previous_history = json.load(f)
                
            # Merge previous history with current history
            for key in previous_history:
                if key in serializable_history:
                    serializable_history[key] = previous_history[key] + serializable_history[key]
        except Exception as e:
            print(f"Warning: Could not merge history files. Error: {e}")
    
    # Save serializable history
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f)
    
    # Also save to standard filename for compatibility
    with open('training_history.json', 'w') as f:
        json.dump(serializable_history, f)
    
    print(f"Training history saved to {history_path} and training_history.json")
    print(f"Final model saved to {MODEL_PATH}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the emotion recognition model")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch to start from (if resuming)")
    
    args = parser.parse_args()
    
    if args.resume:
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            # Find the latest checkpoint
            checkpoints_dir = os.path.join("models", "checkpoints")
            if os.path.exists(checkpoints_dir):
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.h5')]
                if checkpoint_files:
                    checkpoint_files.sort(reverse=True)  # Latest first
                    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_files[0])
                    print(f"Found latest checkpoint: {checkpoint_path}")
                else:
                    print("No checkpoints found. Starting from scratch.")
                    checkpoint_path = None
            else:
                print("Checkpoints directory not found. Starting from scratch.")
                checkpoint_path = None
        
        train_model(start_epoch=args.start_epoch, checkpoint_path=checkpoint_path)
    else:
        train_model() 