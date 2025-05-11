"""
Main script for the Facial Emotion Recognition project.
This script runs the entire pipeline from data loading to evaluation and visualization.
"""

import os
import sys
import argparse
import tensorflow as tf

from config import TRAIN_DIR, TEST_DIR
from utils.utils import setup_logging, set_memory_growth, get_hardware_info
from training.train import train_model
from evaluation.evaluate import evaluate_model
from visualization.visualize import (
    plot_training_history, 
    plot_confusion_matrix, 
    plot_class_distribution,
    plot_sample_images
)


def main(args):
    """
    Main function to run the pipeline.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Facial Emotion Recognition pipeline")
    
    # Configure GPU memory growth
    set_memory_growth()
    
    # Print hardware information
    hw_info = get_hardware_info()
    logger.info(f"Hardware information: {hw_info}")
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs(os.path.join("models", "saved"), exist_ok=True)
    os.makedirs(os.path.join("models", "checkpoints"), exist_ok=True)
    
    # Visualize sample images and class distribution if requested
    if args.explore_data:
        logger.info("Exploring data...")
        plot_sample_images(TRAIN_DIR)
        plot_class_distribution(TRAIN_DIR, TEST_DIR)
    
    # Train model if requested
    if args.train:
        logger.info("Training model...")
        # Check if resuming training
        if args.resume:
            logger.info(f"Resuming training from epoch {args.start_epoch}")
            model, history = train_model(
                start_epoch=args.start_epoch, 
                checkpoint_path=args.checkpoint
            )
        else:
            model, history = train_model()
    else:
        model = None
    
    # Evaluate model if requested
    if args.evaluate:
        logger.info("Evaluating model...")
        results = evaluate_model(model)
    
    # Visualize results if requested
    if args.visualize and os.path.exists("training_history.json"):
        logger.info("Visualizing results...")
        plot_training_history()
        
        if os.path.exists("evaluation_results.json"):
            plot_confusion_matrix()
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--explore-data", action="store_true", help="Explore and visualize the dataset")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch to start from (if resuming)")
    
    args = parser.parse_args()
    
    # If --all is specified, run everything
    if args.all:
        args.train = True
        args.evaluate = True
        args.visualize = True
        args.explore_data = True
    
    # If no arguments provided, display help
    if not (args.train or args.evaluate or args.visualize or args.explore_data):
        parser.print_help()
        sys.exit(1)
    
    main(args) 