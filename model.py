"""
Rock Paper Scissors CNN Model Training Script

This script builds, trains, and evaluates a Convolutional Neural Network (CNN) for classifying
rock-paper-scissors hand gestures from images. The model uses a sequential architecture with
convolutional and pooling layers for feature extraction, followed by dense layers for classification.

Model Architecture:
- 3 Convolutional layers with increasing filters (32, 64, 128)
- MaxPooling layers for downsampling
- Flatten layer to convert 2D features to 1D
- Dense layer with 128 neurons and ReLU activation
- Dropout layer (50%) for regularization
- Output layer with 3 neurons (softmax for multi-class classification)

Training Features:
- Adam optimizer with categorical cross-entropy loss
- Early stopping to prevent overfitting (monitors validation loss)
- Training history visualization
- Model evaluation on test set

Usage:
    python model.py

Prerequisites:
    - Run preprocess.py first to generate preprocessed_data.npz
    - Required libraries: tensorflow, numpy, matplotlib

Output:
    - rock_paper_scissors_model.h5: Trained model weights
    - training_history.png: Training curves plot
    - Console output with test accuracy
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Configuration
MODEL_SAVE_PATH = 'rock_paper_scissors_model.h5'
HISTORY_PLOT_PATH = 'training_history.png'
PREPROCESSED_DATA_PATH = 'preprocessed_data.npz'
EPOCHS = 20
BATCH_SIZE = 32  # Default batch size for fit()

def build_model(input_shape=(64, 64, 3), num_classes=3):
    """
    Build the CNN model architecture.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes

    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Prevent overfitting
        Dense(num_classes, activation='softmax')  # Multi-class classification
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras History object from model.fit()
    """
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline."""
    print("Loading preprocessed data...")
    try:
        data = np.load(PREPROCESSED_DATA_PATH)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        print(f"Data loaded successfully. Training samples: {X_train.shape[0]}")
    except FileNotFoundError:
        print(f"Error: {PREPROCESSED_DATA_PATH} not found. Run preprocess.py first.")
        return

    print("\nBuilding CNN model...")
    model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    model.summary()

    print("\nSetting up training callbacks...")
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',      # Monitor validation loss
        patience=5,              # Stop after 5 epochs of no improvement
        restore_best_weights=True,  # Restore weights from best epoch
        verbose=1
    )

    print(f"\nTraining model for up to {EPOCHS} epochs...")
    print("Early stopping enabled - training will stop if validation loss doesn't improve for 5 epochs")

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    print(f"\nTraining completed. Best epoch: {len(history.history['loss'])}")

    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

    print("\nGenerating training history plots...")
    plot_training_history(history)

    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Additional metrics
    print("\nTraining Summary:")
    print(f"- Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"- Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"- Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"- Total epochs trained: {len(history.history['loss'])}")

if __name__ == "__main__":
    main()