"""
Prediction script for facial emotion recognition.
This script loads a trained model and performs inference on new images.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from config import MODEL_PATH, IMG_SIZE, EMOTION_LABELS


def preprocess_image(image_path):
    """
    Preprocess an image for model prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image ready for model input
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face detector (Haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Process each face
    processed_faces = []
    face_locations = []
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to expected input size
            resized_face = cv2.resize(face_roi, IMG_SIZE)
            
            # Normalize pixel values
            normalized_face = resized_face / 255.0
            
            # Expand dimensions for model input
            processed_face = np.expand_dims(np.expand_dims(normalized_face, -1), 0)
            
            processed_faces.append(processed_face)
            face_locations.append((x, y, w, h))
    else:
        # If no face detected, process the entire image
        resized_image = cv2.resize(gray, IMG_SIZE)
        normalized_image = resized_image / 255.0
        processed_image = np.expand_dims(np.expand_dims(normalized_image, -1), 0)
        processed_faces.append(processed_image)
        face_locations.append((0, 0, gray.shape[1], gray.shape[0]))
    
    return image, processed_faces, face_locations


def predict_emotion(model, processed_face):
    """
    Predict emotion from a preprocessed face image.
    
    Args:
        model: Trained emotion recognition model
        processed_face: Preprocessed face image
        
    Returns:
        Predicted emotion label and probabilities
    """
    # Make prediction
    prediction = model.predict(processed_face)[0]
    
    # Get emotion with highest probability
    emotion_idx = np.argmax(prediction)
    emotion_label = EMOTION_LABELS[emotion_idx]
    
    # Get probability
    emotion_prob = prediction[emotion_idx]
    
    # Create dictionary with all emotions and probabilities
    emotion_probs = {EMOTION_LABELS[i]: float(prediction[i]) for i in range(len(EMOTION_LABELS))}
    
    return emotion_label, emotion_prob, emotion_probs


def visualize_prediction(image, face_locations, emotion_labels, emotion_probs):
    """
    Visualize prediction on the image.
    
    Args:
        image: Original image
        face_locations: List of face bounding box coordinates
        emotion_labels: List of predicted emotion labels
        emotion_probs: List of emotion probabilities
        
    Returns:
        Image with visualization
    """
    # Create a copy of the image
    output = image.copy()
    
    # Draw each face and emotion
    for (x, y, w, h), emotion, prob in zip(face_locations, emotion_labels, emotion_probs):
        # Draw rectangle around face
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Prepare text
        text = f"{emotion}: {prob:.2f}"
        
        # Determine background color based on emotion
        colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 128),  # Brown
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (255, 128, 0), # Cyan
            'Neutral': (128, 128, 128) # Gray
        }
        color = colors.get(emotion, (255, 255, 255))
        
        # Draw filled rectangle for text background
        cv2.rectangle(output, (x, y-25), (x+len(text)*10, y), color, -1)
        
        # Draw text
        cv2.putText(output, text, (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return output


def main(image_path, output_path=None, display=True):
    """
    Main function to predict emotion from an image.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (optional)
        display: Whether to display the result
    """
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    
    # Preprocess image
    image, processed_faces, face_locations = preprocess_image(image_path)
    
    # Predict emotion for each face
    emotion_labels = []
    emotion_probs = []
    
    for face in processed_faces:
        emotion, prob, _ = predict_emotion(model, face)
        emotion_labels.append(emotion)
        emotion_probs.append(prob)
    
    # Visualize prediction
    output = visualize_prediction(image, face_locations, emotion_labels, emotion_probs)
    
    # Display result
    if display:
        cv2.imshow("Emotion Recognition", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save result
    if output_path:
        cv2.imwrite(output_path, output)
        print(f"Output saved to {output_path}")
    
    # Print predictions
    for i, (emotion, prob) in enumerate(zip(emotion_labels, emotion_probs)):
        print(f"Face {i+1}: {emotion} ({prob:.2f})")
    
    return emotion_labels, emotion_probs


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", "-o", help="Path to output image")
    parser.add_argument("--no-display", action="store_true", help="Do not display output")
    
    args = parser.parse_args()
    
    # Run prediction
    main(args.image_path, args.output, not args.no_display) 