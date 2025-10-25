"""
Real-Time Rock Paper Scissors Gesture Detection

MACHINE LEARNING CONCEPTS COVERED:
================================
Computer Vision Pipeline: Converting video streams to ML predictions
Real-time Inference: Applying trained models to live data streams
Region of Interest (ROI): Focusing model on relevant parts of input
Model Deployment: Using trained models in production applications
Latency vs Accuracy Trade-offs: Balancing speed and performance
Edge Case Handling: Dealing with poor lighting, motion blur, etc.

This script demonstrates the complete computer vision application pipeline:
1. Video Capture: Real-time frame acquisition from webcam
2. Preprocessing: Converting video frames to model-compatible format
3. Inference: Running the trained CNN on live data
4. Post-processing: Interpreting model outputs for user feedback
5. Visualization: Displaying results with meaningful UI elements

Key challenges in real-time ML applications:
- Maintaining consistent frame rates (30+ FPS)
- Handling variable lighting conditions
- Managing camera positioning and stability
- Providing user feedback for poor predictions
- Balancing model complexity with inference speed

Features:
- Real-time video capture from webcam (30+ FPS)
- Region of interest (ROI) selection for focused gesture detection
- Live gesture classification with real-time confidence scores
- Visual feedback with bounding box and text overlay
- Mirror effect for natural user experience
- Low-latency inference using optimized CNN

Controls:
- Press 'q' to quit the application

Prerequisites:
- Trained model file: rock_paper_scissors_model.h5
- Webcam/camera available
- Required libraries: opencv-python, tensorflow, numpy

Usage:
    python real_time_cv.py

Technical Notes:
- ROI is set to 80% of frame dimensions, centered for optimal hand detection
- Model expects 64x64 RGB images (matches training input size)
- Confidence threshold can be adjusted for different sensitivity levels
- Uses OpenCV for efficient video processing and display
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = 'rock_paper_scissors_model.h5'
CLASSES = ['rock', 'paper', 'scissors']
IMG_SIZE = 64
ROI_MARGIN = 0.1  # 10% margin on each side (80% ROI)
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for reliable predictions

def preprocess_frame(frame):
    """
    Preprocess a video frame for model prediction.

    Args:
        frame (numpy.ndarray): Input video frame (BGR format)

    Returns:
        numpy.ndarray: Preprocessed frame ready for model input
    """
    # Resize to model input dimensions
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values to [0, 1]
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

def get_prediction(frame):
    """
    Get gesture prediction and confidence from a video frame.

    Args:
        frame (numpy.ndarray): Input video frame

    Returns:
        tuple: (predicted_class, confidence_score)
    """
    # Preprocess frame
    processed = preprocess_frame(frame)

    # Get model prediction
    prediction = model.predict(processed, verbose=0)

    # Get class with highest probability
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    return CLASSES[class_idx], confidence

def draw_ui_overlay(frame, gesture, confidence, roi_coords):
    """
    Draw user interface elements on the video frame.

    Args:
        frame (numpy.ndarray): Video frame to draw on
        gesture (str): Predicted gesture class
        confidence (float): Prediction confidence score
        roi_coords (tuple): ROI coordinates (x1, y1, x2, y2)

    Returns:
        numpy.ndarray: Frame with UI elements drawn
    """
    height, width = frame.shape[:2]

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_coords[0], roi_coords[1]),
                  (roi_coords[2], roi_coords[3]), (255, 0, 0), 2)

    # Display gesture prediction
    text_color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f'Gesture: {gesture} ({confidence:.2f})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Display instructions
    cv2.putText(frame, "Press 'q' to quit", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

def main():
    """Main real-time detection loop."""
    print("Loading trained model...")
    try:
        global model
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Train the model first using model.py")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure a camera is connected.")
        return

    print("Real-time gesture detection started!")
    print("Position your hand in the blue rectangle.")
    print("Press 'q' to quit.")

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Define region of interest (ROI)
            height, width = frame.shape[:2]
            x1 = int(width * ROI_MARGIN)
            y1 = int(height * ROI_MARGIN)
            x2 = int(width * (1 - ROI_MARGIN))
            y2 = int(height * (1 - ROI_MARGIN))
            roi = frame[y1:y2, x1:x2]

            # Get gesture prediction
            gesture, confidence = get_prediction(roi)

            # Draw UI elements
            roi_coords = (x1, y1, x2, y2)
            frame = draw_ui_overlay(frame, gesture, confidence, roi_coords)

            # Display frame
            cv2.imshow('Rock Paper Scissors - Real Time Detection', frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. Application closed.")

if __name__ == "__main__":
    main()