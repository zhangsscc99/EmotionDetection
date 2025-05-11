"""
Visualization module for displaying model results.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMOTION_LABELS


def plot_training_history(history_path="training_history.json", save_path="plots"):
    """
    Plot training history graphs.
    
    Args:
        history_path: Path to the saved training history
        save_path: Directory to save the plots
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Model Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title('Model Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()
    
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(results_path="evaluation_results.json", save_path="plots"):
    """
    Plot confusion matrix.
    
    Args:
        results_path: Path to the evaluation results
        save_path: Directory to save the plots
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Load evaluation results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    conf_matrix = np.array(results['confusion_matrix'])
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=list(EMOTION_LABELS.values()),
               yticklabels=list(EMOTION_LABELS.values()))
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Confusion matrix plot saved to {save_path}")


def plot_class_distribution(train_dir, test_dir, save_path="plots"):
    """
    Plot class distribution in training and test sets.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        save_path: Directory to save the plots
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get class distribution
    train_counts = []
    test_counts = []
    emotions = []
    
    for class_id, emotion in EMOTION_LABELS.items():
        emotions.append(emotion)
        train_class_dir = os.path.join(train_dir, emotion)
        test_class_dir = os.path.join(test_dir, emotion)
        
        # Count samples in train dir
        if os.path.exists(train_class_dir):
            train_counts.append(len(os.listdir(train_class_dir)))
        else:
            train_counts.append(0)
        
        # Count samples in test dir
        if os.path.exists(test_class_dir):
            test_counts.append(len(os.listdir(test_class_dir)))
        else:
            test_counts.append(0)
    
    # Plot class distribution
    plt.figure(figsize=(15, 10))
    
    x = np.arange(len(emotions))
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='Training Set')
    plt.bar(x + width/2, test_counts, width, label='Test Set')
    
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title('Class Distribution in Training and Test Sets', fontsize=16)
    plt.xticks(x, emotions)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_distribution.png'))
    plt.close()
    
    print(f"Class distribution plot saved to {save_path}")


def plot_sample_images(data_dir, num_samples=2, save_path="plots"):
    """
    Plot sample images from each emotion class.
    
    Args:
        data_dir: Path to data directory
        num_samples: Number of samples to plot for each class
        save_path: Directory to save the plots
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.figure(figsize=(15, 10))
    
    for i, (_, emotion) in enumerate(EMOTION_LABELS.items()):
        class_dir = os.path.join(data_dir, emotion)
        
        if not os.path.exists(class_dir):
            continue
            
        image_files = os.listdir(class_dir)[:num_samples]
        
        for j, img_file in enumerate(image_files):
            img_path = os.path.join(class_dir, img_file)
            
            # Read grayscale image
            from PIL import Image
            img = Image.open(img_path).convert('L')
            
            # Plot image
            plt.subplot(len(EMOTION_LABELS), num_samples, i*num_samples + j + 1)
            plt.imshow(img, cmap='gray')
            if j == 0:
                plt.ylabel(emotion, fontsize=12)
            plt.xticks([])
            plt.yticks([])
    
    plt.suptitle('Sample Images from Each Emotion Class', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(save_path, 'sample_images.png'))
    plt.close()
    
    print(f"Sample images plot saved to {save_path}")


def plot_roc_curves(results_path="evaluation_results.json", save_path="plots"):
    """
    Plot ROC curves for each emotion class.
    
    Args:
        results_path: Path to the evaluation results
        save_path: Directory to save the plots
    """
    # This is a placeholder - in a real implementation, we would need the actual
    # predictions and ground truth labels to calculate ROC curves.
    # The current evaluation_results.json does not contain the necessary data.
    
    print("ROC curve plotting requires actual prediction probabilities and ground truth labels.")
    print("This function needs to be run after model evaluation with appropriate data collection.")


if __name__ == "__main__":
    from config import TRAIN_DIR, TEST_DIR
    
    # Plot training history
    plot_training_history()
    
    # Plot confusion matrix
    plot_confusion_matrix()
    
    # Plot class distribution
    plot_class_distribution(TRAIN_DIR, TEST_DIR)
    
    # Plot sample images
    plot_sample_images(TRAIN_DIR)
    
    # Note: ROC curves plot is not included here as it requires 
    # prediction probabilities which are not saved in the current implementation 