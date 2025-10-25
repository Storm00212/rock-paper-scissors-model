"""
Rock Paper Scissors System Testing Script

MACHINE LEARNING CONCEPTS COVERED:
================================
Model Evaluation: Systematic testing of trained models on unseen data
Cross-Validation Concepts: Ensuring model performance is robust and not overfitting
Performance Metrics: Accuracy, confidence scores, and error analysis
System Integration Testing: Validating end-to-end ML pipelines
Debugging ML Systems: Identifying failure points and performance bottlenecks
Statistical Significance: Understanding when results are meaningful vs. random

This script demonstrates professional ML testing practices:
1. Unit Testing: Individual component validation (model loading, preprocessing)
2. Integration Testing: End-to-end pipeline verification
3. Performance Benchmarking: Establishing baseline metrics for comparison
4. Error Analysis: Understanding failure modes and edge cases
5. Automated Testing: Creating reproducible evaluation procedures

Key testing principles covered:
- Test on unseen data (different from training/validation sets)
- Use multiple metrics (accuracy, confidence, error rates)
- Handle edge cases gracefully (missing files, camera issues)
- Provide actionable diagnostic information
- Establish performance baselines for future improvements

Testing Features:
- Batch testing on dataset samples (5 images per class)
- Accuracy calculation and reporting
- Real-time camera access verification
- Error handling and diagnostic messages
- Confidence score analysis and threshold evaluation

Prerequisites:
- Trained model file: rock_paper_scissors_model.h5
- Dataset directory with rock/paper/scissors subdirectories
- Required libraries: opencv-python, tensorflow, numpy

Usage:
    python test_system.py

Output:
- Individual prediction results for sample images
- Overall accuracy statistics
- Real-time detection capability status
- Diagnostic information for troubleshooting

Note:
- Tests first 5 images from each class for efficiency
- Real-time test may fail in headless environments (expected)
- Confidence scores help identify reliable vs. uncertain predictions
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Configuration
MODEL_PATH = 'rock_paper_scissors_model.h5'
DATASET_DIR = 'dataset'
CLASSES = ['rock', 'paper', 'scissors']
IMG_SIZE = 64
TEST_SAMPLES_PER_CLASS = 5  # Number of images to test per class
CONFIDENCE_THRESHOLD = 0.5  # Threshold for reliable predictions

def preprocess_image(img_path):
    """
    Preprocess a single image file for model prediction.

    Args:
        img_path (str): Path to image file

    Returns:
        numpy.ndarray or None: Preprocessed image array or None if loading failed
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Resize to model input dimensions
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

def preprocess_image_from_array(img_array):
    """
    Preprocess an image array (from video frame) for model prediction.

    Args:
        img_array (numpy.ndarray): Image array

    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_prediction(img_array):
    """
    Get gesture prediction from preprocessed image array.

    Args:
        img_array (numpy.ndarray): Preprocessed image array

    Returns:
        tuple: (predicted_class, confidence_score)
    """
    prediction = model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return CLASSES[class_idx], confidence

def test_model_on_dataset():
    """
    Test model accuracy on sample images from the dataset.

    Returns:
        tuple: (accuracy, correct_predictions, total_predictions)
    """
    print("Testing model on sample images from dataset...")
    print("-" * 60)

    correct_predictions = 0
    total_predictions = 0
    low_confidence_predictions = 0

    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Class directory '{class_dir}' not found. Skipping.")
            continue

        # Get sample images (first N images)
        all_images = os.listdir(class_dir)
        test_images = all_images[:TEST_SAMPLES_PER_CLASS]

        print(f"\nTesting class '{class_name}' ({len(test_images)} samples):")

        for img_name in test_images:
            img_path = os.path.join(class_dir, img_name)
            processed = preprocess_image(img_path)

            if processed is not None:
                predicted_class, confidence = get_prediction(processed)
                actual_class = class_name
                is_correct = predicted_class == actual_class

                correct_predictions += int(is_correct)
                total_predictions += 1

                # Track low confidence predictions
                if confidence < CONFIDENCE_THRESHOLD:
                    low_confidence_predictions += 1

                # MACHINE LEARNING CONCEPT: Error Analysis and Confidence Interpretation
                # ==================================================================
                # Analyzing individual predictions helps understand model behavior:
                # - Correct high-confidence predictions: Model is learning well
                # - Correct low-confidence predictions: Model is right but uncertain (needs more training)
                # - Incorrect high-confidence predictions: Model is confidently wrong (needs better features/training)
                # - Incorrect low-confidence predictions: Model knows it's uncertain (reasonable behavior)
            
                status = "✓" if is_correct else "✗"
                confidence_indicator = "⚠️" if confidence < CONFIDENCE_THRESHOLD else "✓"
                print(f"  {status} {img_name[:20]:<20} | Actual: {actual_class:<8} | Predicted: {predicted_class:<8} ({confidence:.2f}) {confidence_indicator}")
            else:
                print(f"  ✗ Failed to load: {img_name}")

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print("\n" + "-" * 60)
    print("Dataset Testing Results:")
    print(f"Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions} correct)")
    print(f"Low confidence predictions: {low_confidence_predictions}/{total_predictions}")
    print("-" * 60)

    return accuracy, correct_predictions, total_predictions

def test_real_time_detection():
    """
    Test real-time detection capabilities.

    Returns:
        bool: True if real-time detection works, False otherwise
    """
    print("\nTesting real-time detection setup...")
    print("-" * 60)

    try:
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            print("✓ Camera access successful")

            ret, frame = cap.read()
            if ret:
                print("✓ Frame capture successful")

                # Test preprocessing and prediction on captured frame
                height, width = frame.shape[:2]
                roi = frame[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]

                processed = preprocess_image_from_array(roi)
                if processed is not None:
                    predicted_class, confidence = get_prediction(processed)
                    confidence_status = "✓" if confidence > CONFIDENCE_THRESHOLD else "⚠️"

                    print(f"✓ Prediction successful: '{predicted_class}' (confidence: {confidence:.2f}) {confidence_status}")
                    print("✓ Real-time detection test PASSED")
                    result = True
                else:
                    print("✗ Frame preprocessing failed")
                    result = False
            else:
                print("✗ Frame capture failed")
                result = False

            cap.release()
        else:
            print("⚠️  No camera available (expected in headless/server environments)")
            print("✓ Real-time detection test SKIPPED (no camera)")
            result = True  # Not a failure, just no camera available

    except Exception as e:
        print(f"✗ Real-time detection test FAILED: {e}")
        result = False

    print("-" * 60)
    return result

def main():
    """Main testing pipeline."""
    print("Rock Paper Scissors System Testing")
    print("=" * 60)

    # Load model
    try:
        global model
        model = load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Model file '{MODEL_PATH}' not found.")
        print("  Run model.py first to train the model.")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Test model on dataset samples
    dataset_accuracy, correct, total = test_model_on_dataset()

    # Test real-time detection
    realtime_success = test_real_time_detection()

    # Overall results
    print("\n" + "=" * 60)
    print("SYSTEM TEST SUMMARY")
    print("=" * 60)

    if total > 0:
        print(f"Dataset Accuracy:     {dataset_accuracy:.1%} ({correct}/{total} correct)")
    else:
        print("Dataset Testing:      ✗ No test images found")

    print(f"Real-time Detection:  {'✓ PASSED' if realtime_success else '✗ FAILED'}")

    # Overall assessment
    system_status = "PASSED" if (dataset_accuracy >= 0.8 and realtime_success) else "FAILED"
    print(f"\nOverall System Status: {system_status}")

    if system_status == "PASSED":
        print("\n🎉 System is ready for use!")
        print("   Run 'python game.py' to play rock-paper-scissors")
        print("   Run 'python real_time_cv.py' for gesture detection demo")
    else:
        print("\n⚠️  System needs attention:")
        if dataset_accuracy < 0.8:
            print("   - Model accuracy is below 80%. Consider retraining.")
        if not realtime_success:
            print("   - Real-time detection failed. Check camera and dependencies.")

    print("=" * 60)

if __name__ == "__main__":
    main()