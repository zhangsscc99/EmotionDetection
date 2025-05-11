"""
Utility functions for the project.
"""

import os
import sys
import time
import logging
from datetime import datetime
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
    
    Returns:
        Logger object
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up logger
    logger = logging.getLogger("emotion_detection")
    logger.setLevel(log_level)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"emotion_detection_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def timer_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
    
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def set_memory_growth():
    """
    Configure TensorFlow to use memory growth to avoid memory allocation issues.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            print(f"Memory growth configuration error: {e}")


def create_required_directories(directories):
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def get_hardware_info():
    """
    Get information about available hardware.
    
    Returns:
        Dictionary containing hardware information
    """
    hardware_info = {
        "gpu_available": tf.config.list_physical_devices('GPU'),
        "tensorflow_version": tf.__version__,
    }
    
    if hardware_info["gpu_available"]:
        hardware_info["num_gpus"] = len(hardware_info["gpu_available"])
        hardware_info["gpu_names"] = [gpu.name for gpu in hardware_info["gpu_available"]]
    else:
        hardware_info["num_gpus"] = 0
        hardware_info["gpu_names"] = []
    
    return hardware_info 