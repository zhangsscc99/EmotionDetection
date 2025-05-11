"""
Utility functions for model training and management.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH

from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau, 
    TensorBoard
)


def get_callbacks(log_dir="logs"):
    """
    Get callbacks for model training.
    
    Args:
        log_dir: Directory to save TensorBoard logs
        
    Returns:
        List of Keras callbacks
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.join("models", "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    # Save best model
    best_model_checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Save model every 5 epochs
    periodic_checkpoint = ModelCheckpoint(
        os.path.join(checkpoints_dir, "model_epoch_{epoch:02d}.h5"),
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        save_freq=5  # Save every 5 epochs (newer TF/Keras versions)
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
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    return [best_model_checkpoint, periodic_checkpoint, early_stopping, reduce_lr, tensorboard]


def model_summary_to_file(model, file_path="model_summary.txt"):
    """
    Save model summary to a text file.
    
    Args:
        model: Keras model
        file_path: Path to save the summary
    """
    # Create a string io object to capture summary
    import io
    from contextlib import redirect_stdout
    
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            model.summary() 