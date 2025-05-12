"""
Evaluation module for the facial emotion recognition model.
"""

import os
import sys
import numpy as np
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TEST_DIR, MODEL_PATH, EMOTION_LABELS
from data.data_loader import create_data_generators
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


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


def evaluate_model(model=None):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained Keras model (if None, will load from MODEL_PATH)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model if not provided
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    
    # Get data generators
    _, _, test_generator = create_data_generators(TEST_DIR, TEST_DIR)
    
    # Evaluate model
    print("Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes
    
    # Generate classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=list(EMOTION_LABELS.values()),
        output_dict=True
    )
    print("Classification Report:")
    print(classification_report(
        y_true, y_pred, 
        target_names=list(EMOTION_LABELS.values())
    ))
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Save results
    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist()
    }
    
    # Ensure results are serializable
    serializable_results = convert_to_serializable(results)
    
    with open("evaluation_results.json", "w") as f:
        json.dump(serializable_results, f, indent=4)
    
    return results


if __name__ == "__main__":
    evaluate_model() 