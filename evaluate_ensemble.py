"""
Evaluation script for the ensemble model based on the paper:
'An Efficient Approach to Face Emotion Recognition with Convolutional Neural Networks'

This script evaluates the ensemble of five-layer and six-layer models
on the test dataset and generates performance metrics.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from config import TRAIN_DIR, TEST_DIR, EMOTION_LABELS
from data.data_loader import create_data_generators
from models.ensemble_model import create_five_six_layer_ensemble, load_ensemble_from_paths


def evaluate_ensemble():
    """
    Evaluate the ensemble of five-layer and six-layer models.
    
    Returns:
        Evaluation metrics
    """
    # Load test data
    _, _, test_generator = create_data_generators(TRAIN_DIR, TEST_DIR)
    
    try:
        # Create ensemble model
        ensemble = create_five_six_layer_ensemble()
        
        # Evaluate on test data
        print("Evaluating ensemble model on test data...")
        test_loss, test_accuracy = ensemble.evaluate_generator(test_generator)
        
        # Get predictions for confusion matrix and classification report
        y_true = []
        y_pred = []
        
        # Reset generator
        test_generator.reset()
        
        # Get predictions for each batch
        for i in range(len(test_generator)):
            x_batch, y_batch = next(test_generator)
            batch_pred = ensemble.predict(x_batch, verbose=0)
            
            true_labels = np.argmax(y_batch, axis=1)
            pred_labels = np.argmax(batch_pred, axis=1)
            
            y_true.extend(true_labels)
            y_pred.extend(pred_labels)
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Get classification report
        cr = classification_report(y_true, y_pred, target_names=list(EMOTION_LABELS.values()))
        
        # Print results
        print(f"Ensemble Model Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"Ensemble Model Test Loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(cr)
        
        # Save results
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.tolist(),
            'normalized_confusion_matrix': cm_normalized.tolist(),
            'classification_report': cr,
            'date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_path = 'ensemble_evaluation.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Evaluation results saved to {results_path}")
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, cm_normalized, list(EMOTION_LABELS.values()))
        
        return results
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train both five-layer and six-layer models first.")
        return None


def plot_confusion_matrix(cm, cm_normalized, class_names):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        cm_normalized: Normalized confusion matrix
        class_names: List of class names
    """
    # Create directory for plots if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Plot raw counts
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Raw Counts)')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix_raw.png'), dpi=300)
    
    # Plot normalized
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix_normalized.png'), dpi=300)
    print(f"Confusion matrices saved to {plots_dir}/")


def load_and_evaluate_custom_ensemble(model_paths, model_names=None):
    """
    Load and evaluate a custom ensemble of models.
    
    Args:
        model_paths: List of paths to model files
        model_names: Optional list of model names
        
    Returns:
        Evaluation metrics
    """
    # Load test data
    _, _, test_generator = create_data_generators(TRAIN_DIR, TEST_DIR)
    
    try:
        # Load ensemble model
        ensemble = load_ensemble_from_paths(model_paths, model_names)
        
        # Evaluate on test data
        print("Evaluating custom ensemble model on test data...")
        test_loss, test_accuracy = ensemble.evaluate_generator(test_generator)
        
        # Get predictions for confusion matrix and classification report
        y_true = []
        y_pred = []
        
        # Reset generator
        test_generator.reset()
        
        # Get predictions for each batch
        for i in range(len(test_generator)):
            x_batch, y_batch = next(test_generator)
            batch_pred = ensemble.predict(x_batch, verbose=0)
            
            true_labels = np.argmax(y_batch, axis=1)
            pred_labels = np.argmax(batch_pred, axis=1)
            
            y_true.extend(true_labels)
            y_pred.extend(pred_labels)
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Get classification report
        cr = classification_report(y_true, y_pred, target_names=list(EMOTION_LABELS.values()))
        
        # Print results
        print(f"Custom Ensemble Model Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"Custom Ensemble Model Test Loss: {test_loss:.4f}")
        print("\nClassification Report:")
        print(cr)
        
        # Save results
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.tolist(),
            'normalized_confusion_matrix': cm_normalized.tolist(),
            'classification_report': cr,
            'date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_path = 'custom_ensemble_evaluation.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Evaluation results saved to {results_path}")
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, cm_normalized, list(EMOTION_LABELS.values()))
        
        return results
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the specified model files exist.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ensemble models")
    parser.add_argument("--custom", action="store_true", help="Evaluate a custom ensemble")
    parser.add_argument("--models", nargs='+', help="Paths to model files for custom ensemble")
    parser.add_argument("--names", nargs='+', help="Names of models for custom ensemble")
    
    args = parser.parse_args()
    
    if args.custom:
        if not args.models:
            print("Error: --models argument is required for custom ensemble evaluation")
            sys.exit(1)
        
        load_and_evaluate_custom_ensemble(args.models, args.names)
    else:
        evaluate_ensemble() 