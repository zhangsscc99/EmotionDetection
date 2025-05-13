"""
Implementation of the ensemble model from the paper:
'An Efficient Approach to Face Emotion Recognition with Convolutional Neural Networks'

This module provides functionality to create ensembles of multiple models
for improved facial emotion recognition performance.
"""

import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES


class EnsembleModel:
    """
    An ensemble model that combines predictions from multiple models 
    by averaging their outputs.
    """
    
    def __init__(self, models, model_names=None):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of Keras models to ensemble
            model_names: Optional list of model names for reference
        """
        self.models = models
        self.model_names = model_names if model_names else [f"model_{i}" for i in range(len(models))]
        self.num_models = len(models)
        
        # Validate that all models have the same input and output shapes
        input_shapes = [model.input_shape for model in models]
        output_shapes = [model.output_shape for model in models]
        
        if len(set(str(shape) for shape in input_shapes)) > 1:
            raise ValueError("All models must have the same input shape")
        
        if len(set(str(shape) for shape in output_shapes)) > 1:
            raise ValueError("All models must have the same output shape")
            
        print(f"Created ensemble with {self.num_models} models:")
        for i, name in enumerate(self.model_names):
            print(f"  - {name}")
    
    def predict(self, x, verbose=0):
        """
        Make predictions using the ensemble by averaging predictions from all models.
        
        Args:
            x: Input data
            verbose: Verbosity level
            
        Returns:
            Averaged predictions from all models
        """
        predictions = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            if verbose > 0:
                print(f"Getting predictions from {self.model_names[i]}...")
            pred = model.predict(x, verbose=verbose)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def evaluate(self, x, y, verbose=1):
        """
        Evaluate the ensemble model on test data.
        
        Args:
            x: Test inputs
            y: Test labels
            verbose: Verbosity level
            
        Returns:
            Loss and accuracy of the ensemble
        """
        if verbose > 0:
            print("Evaluating ensemble model...")
        
        # Get predictions
        predictions = self.predict(x, verbose=verbose)
        
        # Calculate accuracy
        if y.shape[-1] > 1:  # One-hot encoded
            true_labels = np.argmax(y, axis=-1)
            pred_labels = np.argmax(predictions, axis=-1)
        else:
            true_labels = y
            pred_labels = np.round(predictions)
        
        accuracy = np.mean(pred_labels == true_labels)
        
        # Calculate loss (categorical crossentropy)
        epsilon = 1e-7  # Small constant to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.sum(y * np.log(predictions)) / y.shape[0]
        
        if verbose > 0:
            print(f"Ensemble accuracy: {accuracy:.4f}")
            print(f"Ensemble loss: {loss:.4f}")
            
        return loss, accuracy
        
    def evaluate_generator(self, generator, steps=None, verbose=1):
        """
        Evaluate the ensemble model on a data generator.
        
        Args:
            generator: Keras data generator
            steps: Number of steps to evaluate
            verbose: Verbosity level
            
        Returns:
            Loss and accuracy of the ensemble
        """
        if steps is None:
            steps = len(generator)
        
        if verbose > 0:
            print(f"Evaluating ensemble model on {steps} batches...")
        
        y_true = []
        y_pred = []
        
        # Get predictions for each batch
        for i in range(steps):
            if verbose > 0 and i % 10 == 0:
                print(f"Processing batch {i+1}/{steps}")
                
            x_batch, y_batch = next(generator)
            batch_pred = self.predict(x_batch, verbose=0)
            
            y_true.append(y_batch)
            y_pred.append(batch_pred)
        
        # Concatenate batches
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        
        # Calculate metrics
        return self.evaluate(y_pred, y_true, verbose=verbose)


def load_ensemble_from_paths(model_paths, model_names=None):
    """
    Load an ensemble model from a list of model file paths.
    
    Args:
        model_paths: List of paths to model files
        model_names: Optional list of model names
        
    Returns:
        EnsembleModel instance
    """
    models = []
    
    for path in model_paths:
        # Ensure path uses forward slashes
        path = path.replace('\\', '/')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        model = tf.keras.models.load_model(path)
        models.append(model)
    
    return EnsembleModel(models, model_names)


def create_five_six_layer_ensemble():
    """
    Create an ensemble of five-layer and six-layer models
    as described in the paper.
    
    Returns:
        EnsembleModel instance
    """
    # Paths to saved models
    model_dir = os.path.join("models", "saved3")
    five_layer_path = os.path.join(model_dir, "efficient_five_layer_model.h5")
    six_layer_path = os.path.join(model_dir, "efficient_six_layer_model.h5")
    
    # Ensure paths use forward slashes
    five_layer_path = five_layer_path.replace('\\', '/')
    six_layer_path = six_layer_path.replace('\\', '/')
    
    # Check if models exist
    models_exist = True
    if not os.path.exists(five_layer_path):
        print(f"Warning: Five-layer model not found at {five_layer_path}")
        models_exist = False
    
    if not os.path.exists(six_layer_path):
        print(f"Warning: Six-layer model not found at {six_layer_path}")
        models_exist = False
    
    if not models_exist:
        raise FileNotFoundError("One or both models not found. Please train the models first.")
    
    # Load models
    five_layer_model = tf.keras.models.load_model(five_layer_path)
    six_layer_model = tf.keras.models.load_model(six_layer_path)
    

#

    # Create ensemble
    return EnsembleModel(
        [five_layer_model, six_layer_model],
        ["five_layer", "six_layer"]
    ) 