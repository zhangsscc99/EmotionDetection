"""
Training script for the efficient CNN model for facial emotion recognition.
Based on the paper: 'An Efficient Approach to Face Emotion Recognition with Convolutional Neural Networks'
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAIN_DIR, TEST_DIR, EPOCHS, BATCH_SIZE
from data.data_loader import create_data_generators
from data.data_augmentation import create_balanced_generators, get_class_weights
from models.efficient_cnn_model import build_five_layer_model, build_six_layer_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Define new paths for the efficient models
MODEL_DIR = os.path.join("models", "saved3")
CHECKPOINT_DIR = os.path.join("models", "checkpoints3")

# Ensure forward slashes for cross-platform compatibility
def get_model_path(model_type):
    """Get the appropriate model path for the model type"""
    path = os.path.join(MODEL_DIR, f"efficient_{model_type}_model.h5")
    return path.replace('\\', '/')


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


def get_efficient_callbacks(model_type, log_dir="logs/efficient"):
    """
    Get callbacks for model training.
    
    Args:
        model_type: Type of model ('five_layer' or 'six_layer')
        log_dir: Directory to save TensorBoard logs
        
    Returns:
        List of Keras callbacks
    """
    # Create log directory if it doesn't exist
    model_log_dir = os.path.join(log_dir, model_type)
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)
    
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # Get the appropriate model path
    model_path = get_model_path(model_type)
    
    # Save best model
    best_model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Save model every 5 epochs - using proper path formatting
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"efficient_{model_type}_model_epoch_{{epoch:02d}}.h5")
    # Ensure forward slashes for file paths (works on both Windows and Unix)
    checkpoint_path = checkpoint_path.replace('\\', '/')
    
    periodic_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='min',
        save_freq='epoch',
        period=5  # Save every 5 epochs
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        verbose=1,
        restore_best_weights=True
    )
    
    # Learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        verbose=1,
        min_delta=0.0001,
        min_lr=1e-6
    )
    
    # TensorBoard for visualization
    tensorboard = TensorBoard(
        log_dir=model_log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    return [best_model_checkpoint, periodic_checkpoint, early_stopping, reduce_lr, tensorboard]


def model_summary_to_file(model, model_type, file_path=None):
    """
    Save model summary to a text file.
    
    Args:
        model: Keras model
        model_type: Type of model ('five_layer' or 'six_layer')
        file_path: Path to save the summary
    """
    if file_path is None:
        file_path = f"efficient_{model_type}_model_summary.txt"
        
    # Create a string io object to capture summary
    import io
    from contextlib import redirect_stdout
    
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()


def train_efficient_model(model_type='five_layer', start_epoch=0, checkpoint_path=None, use_balanced_data=True):
    """
    Train the efficient CNN model for facial emotion recognition.
    
    Args:
        model_type: Type of model to train ('five_layer' or 'six_layer')
        start_epoch: Epoch to start training from (for resuming training)
        checkpoint_path: Path to checkpoint file to resume training from
        use_balanced_data: Whether to use balanced data generators
        
    Returns:
        Trained model and training history
    """
    # Ensure directories exist
    ensure_dir_exists(MODEL_DIR)
    ensure_dir_exists(CHECKPOINT_DIR)
    ensure_dir_exists("logs/efficient")
    
    print("Creating data generators...")
    if use_balanced_data:
        # Use the balanced data generators as described in the paper
        print("Using balanced data generators with the paper's augmentation strategy")
        train_generator, validation_generator, test_generator = create_balanced_generators(
            TRAIN_DIR, TEST_DIR
        )
    else:
        # Use the standard data generators
        print("Using standard data generators")
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
        print(f"Building new {model_type} model...")
        if model_type == 'five_layer':
            model = build_five_layer_model()
        else:
            model = build_six_layer_model()
        
    model_summary_to_file(model, model_type)
    
    # Calculate class weights to handle imbalance
    # If using balanced generators, class weights are less important
    if use_balanced_data:
        class_weights = None
        print("Using balanced generators - class weights not applied")
    else:
        class_weights = get_class_weights(train_generator)
        print("Class weights:", class_weights)
    
    # Get callbacks
    callbacks = get_efficient_callbacks(model_type)
    
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
    history_path = f'efficient_{model_type}_history_{time.strftime("%Y%m%d_%H%M%S")}.json'
    # Convert history to serializable format
    history_dict = history.history
    serializable_history = convert_to_serializable(history_dict)
    
    # If resuming training and history file exists, try to merge histories
    if start_epoch > 0 and os.path.exists(f'efficient_{model_type}_history.json'):
        try:
            with open(f'efficient_{model_type}_history.json', 'r') as f:
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
    with open(f'efficient_{model_type}_history.json', 'w') as f:
        json.dump(serializable_history, f)
    
    model_path = get_model_path(model_type)
    print(f"Training history saved to {history_path} and efficient_{model_type}_history.json")
    print(f"Final model saved to {model_path}")
    
    # Evaluate the model on test data
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save evaluation results to a file
    eval_results = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f'efficient_{model_type}_evaluation.json', 'w') as f:
        json.dump(eval_results, f)
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the efficient emotion recognition model")
    parser.add_argument("--model", type=str, choices=['five_layer', 'six_layer'], default='five_layer',
                        help="Type of model to train")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch to start from (if resuming)")
    parser.add_argument("--no-balance", action="store_true", help="Do not use balanced data generators")
    
    args = parser.parse_args()
    
    if args.resume:
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            # Find the latest checkpoint
            if os.path.exists(CHECKPOINT_DIR):
                # Filter checkpoints to get only those matching the specified model type
                checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) 
                                  if f.endswith('.h5') and f'efficient_{args.model}_model' in f]
                if checkpoint_files:
                    checkpoint_files.sort(reverse=True)  # Latest first
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
                    print(f"Found latest checkpoint: {checkpoint_path}")
                else:
                    print(f"No checkpoints found for {args.model} model. Starting from scratch.")
                    checkpoint_path = None
            else:
                print("Checkpoints directory not found. Starting from scratch.")
                checkpoint_path = None
        
        train_efficient_model(
            model_type=args.model, 
            start_epoch=args.start_epoch, 
            checkpoint_path=checkpoint_path,
            use_balanced_data=not args.no_balance
        )
    else:
        train_efficient_model(
            model_type=args.model,
            use_balanced_data=not args.no_balance
        ) 