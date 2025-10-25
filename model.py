"""
Rock Paper Scissors CNN Model Training Script

MACHINE LEARNING CONCEPTS COVERED:
================================
Convolutional Neural Networks (CNNs): Specialized for image recognition tasks
Feature Learning: Automatic extraction of hierarchical features from raw pixels
Backpropagation: Algorithm that computes gradients and updates model weights
Gradient Descent: Optimization algorithm that minimizes loss function
Overfitting Prevention: Techniques like dropout and early stopping
Model Evaluation: Using validation and test sets to measure generalization
Hyperparameter Tuning: Selecting optimal learning rates, batch sizes, etc.

This script demonstrates the complete machine learning workflow:
1. Model Architecture Design (feature engineering through layers)
2. Loss Function Selection (categorical crossentropy for multi-class)
3. Optimizer Choice (Adam for adaptive learning rates)
4. Training Loop (forward pass, loss computation, backpropagation, weight updates)
5. Regularization (dropout to prevent overfitting)
6. Early Stopping (prevent overfitting by monitoring validation performance)

CNN Architecture Explanation:
- Convolutional layers learn spatial patterns (edges, textures, shapes)
- Pooling layers reduce spatial dimensions and provide translation invariance
- Dense layers learn high-level combinations of features
- Softmax converts logits to class probabilities

Why CNNs for images? Traditional neural networks would require millions of parameters
for raw pixels. CNNs exploit spatial structure to learn efficiently with far fewer parameters.

Training Features:
- Adam optimizer: Adaptive learning rates for faster convergence
- Categorical cross-entropy loss: Measures difference between predicted and true probability distributions
- Early stopping: Prevents overfitting by monitoring validation loss
- Training history visualization: Tracks learning progress and identifies issues
- Model evaluation on test set: Unbiased performance assessment

Usage:
    python model.py

Prerequisites:
    - Run preprocess.py first to generate preprocessed_data.npz
    - Required libraries: tensorflow, numpy, matplotlib

Output:
    - rock_paper_scissors_model.h5: Trained model weights (learned parameters)
    - training_history.png: Training curves plot (visualizes learning process)
    - Console output with test accuracy (generalization performance)
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

    MACHINE LEARNING CONCEPT: Neural Network Architecture Design
    ===========================================================
    Designing a neural network is like designing a feature extraction pipeline.
    Each layer transforms the data to make classification easier:

    Convolutional Layers: Learn spatial patterns (edges, textures, shapes of hands)
    Pooling Layers: Reduce spatial size while retaining important features
    Dense Layers: Learn combinations of features for final classification

    Why this specific architecture?
    - Progressive increase in filters (32→64→128): Learn simple to complex features
    - ReLU activation: Introduces non-linearity, helps learn complex patterns
    - MaxPooling: Provides translation invariance (gesture position doesn't matter)
    - Dropout: Randomly deactivates neurons to prevent co-adaptation and overfitting

    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes

    Returns:
        Sequential: Compiled Keras model ready for training
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

    # MACHINE LEARNING CONCEPT: Loss Functions and Optimizers
    # ======================================================
    # Loss Function: Categorical Cross-Entropy
    # - Measures how well predicted probabilities match true labels
    # - For multi-class: L = -Σ(y_true * log(y_pred)) across all classes
    # - Lower loss = better predictions, drives learning during training
    #
    # Optimizer: Adam (Adaptive Moment Estimation)
    # - Combines benefits of RMSProp + Momentum
    # - Adapts learning rates for each parameter individually
    # - Maintains moving averages of gradients and squared gradients
    # - Generally works well without much hyperparameter tuning

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

    # MACHINE LEARNING CONCEPT: Training Loop and Batch Processing
    # ===========================================================
    # Training happens in epochs (full passes through training data) and batches (subsets).
    #
    # Batch Size: Number of samples processed before updating weights
    # - Larger batches: More stable gradients, faster training, higher memory usage
    # - Smaller batches: Noisier gradients, slower training, better generalization
    #
    # Epochs: One complete pass through the training data
    # - Too few: Underfitting (model hasn't learned enough)
    # - Too many: Overfitting (model memorizes training data)
    #
    # Validation Data: Used to monitor generalization during training
    # - Helps detect overfitting (training loss decreases, validation loss increases)

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

    # MACHINE LEARNING CONCEPT: Model Evaluation and Generalization
    # ============================================================
    # Test set evaluation measures how well the model performs on completely unseen data.
    # This is the true measure of generalization - how well the model works in the real world.
    #
    # Key metrics:
    # - Loss: How wrong the model's predictions are (lower is better)
    # - Accuracy: Percentage of correct predictions (higher is better)
    #
    # Compare training vs validation vs test performance:
    # - Training performance: How well model memorized training data
    # - Validation performance: How well model generalizes during training
    # - Test performance: True generalization to new data

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Additional metrics and analysis
    print("\nTraining Summary:")
    print(f"- Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"- Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"- Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"- Total epochs trained: {len(history.history['loss'])}")

    # MACHINE LEARNING CONCEPT: Overfitting Detection
    # ==============================================
    # Compare training vs validation performance to detect overfitting:
    # - If training accuracy >> validation accuracy: Overfitting (memorizing training data)
    # - If both accuracies are similar and high: Good generalization
    # - If both accuracies are low: Underfitting (model too simple)

    train_val_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
    if train_val_gap > 0.1:
        print(f"- WARNING: Potential overfitting detected (train-val gap: {train_val_gap:.3f})")
    elif test_accuracy < history.history['val_accuracy'][-1] - 0.05:
        print(f"- WARNING: Test performance significantly worse than validation")
    else:
        print("- Model shows good generalization (no overfitting detected)")

if __name__ == "__main__":
    main()