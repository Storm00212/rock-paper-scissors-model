"""
Rock Paper Scissors Image Preprocessing Script

MACHINE LEARNING CONCEPTS COVERED:
===============================
Data Preprocessing: Preparing raw data for machine learning algorithms
Feature Engineering: Transforming images into suitable numerical representations
Data Splitting: Dividing data into train/validation/test sets to prevent overfitting
One-Hot Encoding: Converting categorical labels to numerical format for neural networks
Data Normalization: Scaling pixel values to [0,1] range for better gradient descent convergence

This script demonstrates the critical first step in any computer vision ML pipeline:
converting raw images into a format that neural networks can understand and learn from.

Dataset Structure:
- dataset/
  - rock/     (contains rock gesture images)
  - paper/    (contains paper gesture images)
  - scissors/ (contains scissors gesture images)

Preprocessing Steps:
1. Load all images from each class directory
2. Resize images to 64x64 pixels for consistent input size (fixed-length feature vectors)
3. Normalize pixel values to [0, 1] range (improves training stability and convergence)
4. Convert labels to one-hot encoded format (required for multi-class classification)
5. Split data into train (70%), validation (15%), and test (15%) sets (prevents data leakage)
6. Save preprocessed data to compressed numpy file (efficient storage and loading)

Why these steps matter in ML:
- Consistent input size: Neural networks require fixed-dimension inputs
- Normalization: Prevents features with large ranges from dominating learning
- Train/val/test split: Allows unbiased evaluation of model generalization
- One-hot encoding: Enables softmax activation for multi-class probability outputs

Usage:
    python preprocess.py

Output:
    - preprocessed_data.npz: Contains X_train, y_train, X_val, y_val, X_test, y_test arrays
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configuration
DATASET_DIR = 'dataset'  # Directory containing class subdirectories
CLASSES = ['rock', 'paper', 'scissors']  # Gesture classes
IMG_SIZE = 64  # Target image size (64x64 pixels)
TEST_SPLIT = 0.3  # 30% for validation + test
VAL_SPLIT = 0.5  # 50% of test_split for validation (15% total), 50% for test (15% total)
RANDOM_STATE = 42  # For reproducible splits

def load_data():
    """
    Load and preprocess all images from the dataset.

    MACHINE LEARNING CONCEPT: Feature Extraction and Dataset Creation
    ============================================================
    This function demonstrates how raw images are converted into structured datasets.
    Each image becomes a feature vector, and we build the (X, y) pairs needed for supervised learning.

    The output shapes are crucial:
    - X: (num_samples, height, width, channels) - feature matrix
    - y: (num_samples, num_classes) - one-hot encoded labels for multi-class classification

    Returns:
        tuple: (data, labels) where data is numpy array of images, labels is numpy array of class indices
    """
    data = []
    labels = []

    print("Loading and preprocessing images...")

    for label, class_name in enumerate(CLASSES):
        #Enumerates the classess into rock, paper and scissors.
        class_dir = os.path.join(DATASET_DIR, class_name)
        print(f"Processing class '{class_name}' from {class_dir}")
        #error handling for directory error.
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue

        image_count = 0
        for img_name in os.listdir(class_dir):
            #loeads the image in each class.
            img_path = os.path.join(class_dir, img_name)

            # Read image
            img = cv2.imread(img_path)

            if img is not None:
                # Resize to target dimensions 64pixels
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Normalize pixel values to [0, 1]
                img = img / 255.0
              #creates arrays for the tables.
                data.append(img)
                labels.append(label)
                image_count += 1
            else:
                print(f"Warning: Could not read image {img_path}")

        print(f"Loaded {image_count} images for class '{class_name}'")

    # Convert to numpy arrays - efficient numerical computation
    data = np.array(data, dtype=np.float32)  # Float32 for GPU acceleration
    labels = np.array(labels, dtype=np.int32)

    # MACHINE LEARNING CONCEPT: One-Hot Encoding
    # ==========================================
    # Convert categorical labels (0, 1, 2) to binary vectors ([1,0,0], [0,1,0], [0,0,1])
    # This is required for multi-class classification with softmax activation
    # Without one-hot encoding, the model would learn ordinal relationships between classes
    # Now you know why you learned matrices in class boii.
    labels = to_categorical(labels, num_classes=len(CLASSES))

    print(f"Total images loaded: {len(data)}")
    print(f"Image shape: {data.shape[1:]} -> {data.shape[1]}x{data.shape[2]} pixels, {data.shape[3]} channels")
    print(f"Label shape: {labels.shape[1:]} -> {labels.shape[1]} classes (one-hot encoded)")
    print(f"Data type: {data.dtype} (optimal for neural network computations)")

    return data, labels

def main():
    """Main preprocessing pipeline."""
    # Load and preprocess data
    X, y = load_data()

    # MACHINE LEARNING CONCEPT: Train/Validation/Test Split
    # ===================================================
    # Why split data? To evaluate how well our model generalizes to unseen data.
    #
    # Training Set (70%): Used to learn model parameters (weights & biases)
    # Validation Set (15%): Used to tune hyperparameters and prevent overfitting during training
    # Test Set (15%): Used ONLY for final evaluation - never seen during training
    #
    # Stratification ensures each split maintains the same class distribution as the original data,
    # preventing biased evaluation (e.g., if one class is underrepresented in test set).

    # First split: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y.argmax(axis=1)
    )

    # Second split: 50% of temp for validation, 50% for test (15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=y_temp.argmax(axis=1)
    )

    # Print dataset statistics
    print("\nDataset split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Verify class distribution
    print("\nClass distribution:")
    for i, class_name in enumerate(CLASSES):
        train_count = y_train[:, i].sum()
        val_count = y_val[:, i].sum()
        test_count = y_test[:, i].sum()
        print(f"{class_name}: Train={int(train_count)}, Val={int(val_count)}, Test={int(test_count)}")

    # Save preprocessed data
    output_file = 'preprocessed_data.npz'
    np.savez_compressed(output_file,
                       X_train=X_train, y_train=y_train,
                       X_val=X_val, y_val=y_val,
                       X_test=X_test, y_test=y_test)

    print(f"\nPreprocessed data saved to '{output_file}'")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()