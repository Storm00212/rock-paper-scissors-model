"""
Rock Paper Scissors Image Preprocessing Script

This script loads images from the dataset directory, preprocesses them by resizing and normalizing,
and splits the data into training, validation, and test sets for machine learning model training.

Dataset Structure:
- dataset/
  - rock/     (contains rock gesture images)
  - paper/    (contains paper gesture images)
  - scissors/ (contains scissors gesture images)

Preprocessing Steps:
1. Load all images from each class directory
2. Resize images to 64x64 pixels for consistent input size
3. Normalize pixel values to [0, 1] range
4. Convert labels to one-hot encoded format
5. Split data into train (70%), validation (15%), and test (15%) sets
6. Save preprocessed data to compressed numpy file

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

    Returns:
        tuple: (data, labels) where data is numpy array of images, labels is numpy array of class indices
    """
    data = []
    labels = []

    print("Loading and preprocessing images...")

    for label, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        print(f"Processing class '{class_name}' from {class_dir}")

        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue

        image_count = 0
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            # Read image
            img = cv2.imread(img_path)

            if img is not None:
                # Resize to target dimensions
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Normalize pixel values to [0, 1]
                img = img / 255.0

                data.append(img)
                labels.append(label)
                image_count += 1
            else:
                print(f"Warning: Could not read image {img_path}")

        print(f"Loaded {image_count} images for class '{class_name}'")

    # Convert to numpy arrays
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=len(CLASSES))

    print(f"Total images loaded: {len(data)}")
    print(f"Image shape: {data.shape[1:]}")  # (height, width, channels)
    print(f"Label shape: {labels.shape[1:]}")  # (num_classes,)

    return data, labels

def main():
    """Main preprocessing pipeline."""
    # Load and preprocess data
    X, y = load_data()

    # Split into train, validation, and test sets
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