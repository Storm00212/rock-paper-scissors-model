"""
Rock Paper Scissors Interactive Game

MACHINE LEARNING CONCEPTS COVERED:
================================
Human-AI Interaction: Designing intuitive interfaces for ML systems
Real-time Decision Making: ML models making instant predictions for gameplay
Confidence Thresholds: Using prediction probabilities to handle uncertainty
User Experience Design: Balancing technical accuracy with user enjoyment
Error Handling: Gracefully managing model failures and edge cases
Feedback Loops: Using game results to improve future interactions

This script demonstrates practical ML deployment challenges:
1. Model Reliability: Handling low-confidence predictions gracefully
2. User Feedback: Providing clear, immediate results users can understand
3. Timing Constraints: Managing real-time interaction within human attention spans
4. Error Recovery: Continuing gameplay despite occasional model failures
5. Performance Expectations: Balancing speed vs accuracy for gaming applications

The game serves as a perfect example of ML in human-facing applications where:
- Speed matters (real-time interaction)
- Reliability is crucial (fair gameplay)
- User experience trumps technical perfection
- Error handling must be graceful and non-disruptive

Features:
- Real-time gesture detection using trained CNN model
- Interactive gameplay with 3-second countdown timer
- Live score tracking across multiple rounds
- Visual feedback with detailed game results
- Computer opponent with random move selection (fair baseline)
- Confidence-based gesture validation

Game Rules (Rock Paper Scissors):
- ✊ Rock beats ✌️ Scissors (rock smashes scissors)
- ✋ Paper beats ✊ Rock (paper covers rock)
- ✌️ Scissors beats ✋ Paper (scissors cuts paper)
- Same gestures result in a tie

Controls:
- Press 's' to start a new round
- Press 'q' to quit the game anytime
- Position hand clearly in the blue rectangle during countdown

Prerequisites:
- Trained model file: rock_paper_scissors_model.h5
- Webcam/camera available
- Required libraries: opencv-python, tensorflow, numpy

Usage:
    python game.py

Game Flow:
1. Player sees live video feed with ROI indicator and instructions
2. Press 's' to begin each round (starts 3-second countdown)
3. Countdown timer guides gesture timing (3... 2... 1...)
4. Player forms gesture during countdown period
5. Computer randomly selects its move (unpredictable but fair)
6. ML model analyzes player's gesture and determines winner
7. Results displayed with scores and visual feedback
8. 3-second result display before returning to live feed
9. Repeat for continuous gameplay or quit anytime

Technical Notes:
- Model confidence affects gesture reliability and user feedback
- ROI is 80% of frame dimensions, centered for optimal detection
- Low-confidence predictions trigger "unclear" feedback
- Results displayed for 3 seconds to match human attention span
- Random computer moves ensure fair, unpredictable gameplay
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
import time

# Configuration
MODEL_PATH = 'rock_paper_scissors_model.h5'
CLASSES = ['rock', 'paper', 'scissors']
IMG_SIZE = 64
ROI_MARGIN = 0.1  # 10% margin on each side
COUNTDOWN_DURATION = 3  # seconds
RESULT_DISPLAY_TIME = 3000  # milliseconds
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for gesture acceptance

def determine_winner(player_gesture, computer_gesture):
    """
    Determine the winner based on rock-paper-scissors rules.

    Args:
        player_gesture (str): Player's gesture
        computer_gesture (str): Computer's gesture

    Returns:
        str: Result message ("Player wins", "Computer wins", or "Tie")
    """
    if player_gesture == computer_gesture:
        return "Tie"

    # Define winning combinations
    winning_moves = {
        'rock': 'scissors',
        'paper': 'rock',
        'scissors': 'paper'
    }

    if winning_moves[player_gesture] == computer_gesture:
        return "Player wins"
    else:
        return "Computer wins"

def preprocess_frame(frame):
    """
    Preprocess video frame for model prediction.

    Args:
        frame (numpy.ndarray): Input video frame

    Returns:
        numpy.ndarray: Preprocessed frame ready for model input
    """
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_prediction(frame):
    """
    Get gesture prediction from video frame.

    Args:
        frame (numpy.ndarray): Input video frame

    Returns:
        tuple: (predicted_gesture, confidence_score)
    """
    processed = preprocess_frame(frame)
    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return CLASSES[class_idx], confidence

def draw_game_ui(frame, player_score, computer_score, roi_coords):
    """
    Draw game user interface elements on the frame.

    Args:
        frame (numpy.ndarray): Video frame
        player_score (int): Player's current score
        computer_score (int): Computer's current score
        roi_coords (tuple): ROI coordinates (x1, y1, x2, y2)

    Returns:
        numpy.ndarray: Frame with UI elements
    """
    height, width = frame.shape[:2]

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_coords[0], roi_coords[1]),
                  (roi_coords[2], roi_coords[3]), (255, 0, 0), 2)

    # Display instructions and scores
    cv2.putText(frame, "Press 's' to play round, 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Score - You: {player_score} Computer: {computer_score}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Position hand in blue rectangle", (10, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame

def play_round(cap, model, player_score, computer_score, round_count):
    """
    Play a single round of the game.

    Args:
        cap: OpenCV VideoCapture object
        model: Trained Keras model
        player_score (int): Current player score
        computer_score (int): Current computer score
        round_count (int): Current round number

    Returns:
        tuple: (updated_player_score, updated_computer_score, round_result)
    """
    height, width = None, None

    # Get frame dimensions
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]

    # Countdown before capture
    print(f"Round {round_count + 1} - Get ready!")
    for countdown in range(COUNTDOWN_DURATION, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return player_score, computer_score, "Error"

        frame = cv2.flip(frame, 1)

        # Draw countdown number
        cv2.putText(frame, str(countdown), (width//2 - 50, height//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
        cv2.putText(frame, "Make your gesture!", (width//2 - 150, height//2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        cv2.imshow('Rock Paper Scissors Game', frame)
        cv2.waitKey(1000)

    # Capture player's gesture
    ret, frame = cap.read()
    if not ret:
        return player_score, computer_score, "Error"

    frame = cv2.flip(frame, 1)
    roi = frame[int(height*ROI_MARGIN):int(height*(1-ROI_MARGIN)),
                int(width*ROI_MARGIN):int(width*(1-ROI_MARGIN))]

    player_gesture, confidence = get_prediction(roi)

    # MACHINE LEARNING CONCEPT: Confidence Thresholds and Uncertainty Handling
    # ===================================================================
    # In real-world ML applications, models aren't always confident in their predictions.
    # Confidence thresholds help decide when to trust model predictions vs. asking for clarification.
    #
    # Why this matters in gaming:
    # - Low confidence predictions could lead to unfair game outcomes
    # - Users need clear feedback when gestures aren't recognized properly
    # - Better user experience by acknowledging uncertainty rather than guessing wrong

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"Low confidence prediction ({confidence:.2f} < {CONFIDENCE_THRESHOLD}) - marking as unclear")
        player_gesture = "unclear"

    # Computer makes random move
    computer_gesture = random.choice(CLASSES)

    # Determine winner
    if player_gesture == "unclear":
        result = "Gesture unclear - No points awarded"
        winner_text = "No winner"
    else:
        result = determine_winner(player_gesture, computer_gesture)
        winner_text = result

        # Update scores
        if result == "Player wins":
            player_score += 1
        elif result == "Computer wins":
            computer_score += 1

    # Display results
    result_frame = frame.copy()

    # Player's gesture
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.putText(result_frame, f"You: {player_gesture}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    if player_gesture != "unclear":
        cv2.putText(result_frame, f"Confidence: {confidence:.2f}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Computer's gesture
    cv2.putText(result_frame, f"Computer: {computer_gesture}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Result
    cv2.putText(result_frame, winner_text, (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    # Round number
    cv2.putText(result_frame, f"Round {round_count + 1}", (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Rock Paper Scissors Game', result_frame)
    cv2.waitKey(RESULT_DISPLAY_TIME)

    return player_score, computer_score, result

def main():
    """Main game loop."""
    print("Loading Rock Paper Scissors Game...")
    print("=" * 50)

    # Load model
    try:
        global model
        model = load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Model file '{MODEL_PATH}' not found.")
        print("  Run model.py first to train the model.")
        return

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Could not access camera.")
        return
    print("✓ Camera initialized")

    print("\n" + "=" * 50)
    print("ROCK PAPER SCISSORS GAME")
    print("=" * 50)
    print("Instructions:")
    print("- Position your hand in the blue rectangle")
    print("- Press 's' to start a round")
    print("- Make your gesture during the countdown")
    print("- Press 'q' to quit anytime")
    print("=" * 50)

    # Game state
    player_score = 0
    computer_score = 0
    round_count = 0

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Define ROI
            height, width = frame.shape[:2]
            roi_coords = (int(width*ROI_MARGIN), int(height*ROI_MARGIN),
                         int(width*(1-ROI_MARGIN)), int(height*(1-ROI_MARGIN)))

            # Draw game UI
            frame = draw_game_ui(frame, player_score, computer_score, roi_coords)

            # Display frame
            cv2.imshow('Rock Paper Scissors Game', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting game...")
                break
            elif key == ord('s'):
                # Play a round
                player_score, computer_score, round_result = play_round(
                    cap, model, player_score, computer_score, round_count)
                round_count += 1
                print(f"Round {round_count} result: {round_result}")
                print(f"Current score - You: {player_score}, Computer: {computer_score}")

    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Final results
        print("\n" + "=" * 50)
        print("GAME OVER")
        print("=" * 50)
        print(f"Final Score - You: {player_score}, Computer: {computer_score}")
        print(f"Rounds played: {round_count}")

        if round_count > 0:
            if player_score > computer_score:
                print("🎉 Congratulations! You won the game!")
            elif player_score < computer_score:
                print("🤖 Computer wins! Better luck next time!")
            else:
                print("🤝 It's a tie! Well played!")
        else:
            print("No rounds played.")

        print("=" * 50)

if __name__ == "__main__":
    main()