"""
Enhanced Rock Paper Scissors Game with Advanced Hand Tracking

MACHINE LEARNING CONCEPTS COVERED:
================================
Multi-Modal Integration: Combining CNN classification with hand landmark tracking
Real-time Feature Fusion: Merging multiple computer vision techniques
Advanced Visual Feedback: Providing rich, informative user interfaces
Error Handling and Robustness: Graceful degradation under adverse conditions
Temporal Smoothing: Maintaining visual stability in real-time applications

This enhanced version integrates advanced hand tracking with the existing CNN-based
gesture classification, providing visual landmark feedback and improved robustness.

Features:
- Advanced hand landmark detection and tracking
- Real-time visual feedback with colored landmark connections
- Smooth interpolation for fluid hand movement visualization
- Integration with existing CNN gesture classification
- Enhanced error handling for low-light and partial visibility conditions
- Confidence-based filtering and gesture validation
- Live annotations with gesture confidence scores overlaid on hand

The system combines two powerful ML approaches:
1. CNN for high-level gesture recognition (rock/paper/scissors classification)
2. Hand landmark tracking for detailed spatial hand analysis and visualization

This demonstrates how multiple ML models can work together to create more
robust and informative computer vision applications.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
import time
from hand_tracking import HandTracker

# Configuration
MODEL_PATH = 'rock_paper_scissors_model.h5'
CLASSES = ['rock', 'paper', 'scissors']
IMG_SIZE = 64
ROI_MARGIN = 0.1  # 10% margin on each side
COUNTDOWN_DURATION = 3  # seconds
RESULT_DISPLAY_TIME = 3000  # milliseconds
GESTURE_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for gesture acceptance
LANDMARK_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for landmark display

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

def draw_enhanced_ui(frame, player_score, computer_score, roi_coords, landmarks=None, gesture=None, confidence=None):
    """
    Draw enhanced game user interface with hand tracking visualization.

    MACHINE LEARNING CONCEPT: Multi-Modal Visual Feedback
    ===================================================
    This function demonstrates how to combine multiple sources of information
    into a coherent visual interface, making complex ML outputs interpretable.

    Args:
        frame (numpy.ndarray): Video frame
        player_score (int): Player's current score
        computer_score (int): Computer's current score
        roi_coords (tuple): ROI coordinates (x1, y1, x2, y2)
        landmarks (numpy.ndarray): Hand landmark coordinates (optional)
        gesture (str): Current gesture prediction (optional)
        confidence (float): Gesture confidence score (optional)

    Returns:
        numpy.ndarray: Frame with enhanced UI elements
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

    # Display gesture information if available
    if gesture and confidence is not None:
        # Color code based on confidence
        if confidence >= GESTURE_CONFIDENCE_THRESHOLD:
            color = (0, 255, 0)  # Green for confident predictions
            status = "✓"
        else:
            color = (0, 165, 255)  # Orange for low confidence
            status = "⚠️"

        cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f}) {status}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display hand tracking status
    if landmarks is not None:
        # Check if hand is fully visible
        hand_center = hand_tracker.get_hand_center(landmarks)
        fully_visible = hand_tracker.is_hand_fully_visible(landmarks, frame.shape)

        if fully_visible:
            cv2.putText(frame, "Hand: Fully Visible", (10, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Hand: Partially Visible", (10, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Display landmark count
        cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, "Position hand in blue rectangle", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame

def play_enhanced_round(cap, model, hand_tracker, player_score, computer_score, round_count):
    """
    Play a single round with enhanced hand tracking and visualization.

    MACHINE LEARNING CONCEPT: Real-Time Multi-Modal Processing
    ========================================================
    This function demonstrates the complexity of processing multiple ML models
    in real-time while maintaining temporal consistency and user experience.

    Args:
        cap: OpenCV VideoCapture object
        model: Trained Keras model for gesture classification
        hand_tracker: HandTracker instance for landmark detection
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

    # Countdown before capture with enhanced visualization
    print(f"Round {round_count + 1} - Get ready!")
    for countdown in range(COUNTDOWN_DURATION, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return player_score, computer_score, "Error"

        frame = cv2.flip(frame, 1)

        # Detect hand during countdown for preview
        landmarks, bbox = hand_tracker.detect_hand(frame)
        frame = hand_tracker.draw_hand_landmarks(frame, landmarks, bbox)

        # Draw countdown number prominently
        cv2.putText(frame, str(countdown), (width//2 - 100, height//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 15)
        cv2.putText(frame, "Make your gesture!", (width//2 - 200, height//2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        # Show current hand tracking status
        if landmarks is not None:
            cv2.putText(frame, "Hand detected - Hold steady!", (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Enhanced Rock Paper Scissors Game', frame)
        cv2.waitKey(1000)

    # Capture player's gesture with enhanced processing
    ret, frame = cap.read()
    if not ret:
        return player_score, computer_score, "Error"

    frame = cv2.flip(frame, 1)

    # Define ROI for gesture classification
    roi = frame[int(height*ROI_MARGIN):int(height*(1-ROI_MARGIN)),
                int(width*ROI_MARGIN):int(width*(1-ROI_MARGIN))]

    # Get gesture classification
    player_gesture, confidence = get_prediction(roi)

    # Get hand landmarks for enhanced visualization
    landmarks, bbox = hand_tracker.detect_hand(frame)

    # Check confidence threshold with enhanced feedback
    if confidence < GESTURE_CONFIDENCE_THRESHOLD:
        print(f"Low confidence prediction ({confidence:.2f} < {GESTURE_CONFIDENCE_THRESHOLD}) - marking as unclear")
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

    # Display enhanced results with hand tracking visualization
    result_frame = frame.copy()

    # Draw hand landmarks on result frame
    result_frame = hand_tracker.draw_hand_landmarks(result_frame, landmarks, bbox)

    # Player's gesture with enhanced feedback
    if player_gesture == "unclear":
        color = (0, 165, 255)  # Orange for unclear
        gesture_text = "UNCLEAR"
    else:
        color = (0, 255, 0)  # Green for clear
        gesture_text = player_gesture.upper()

    cv2.putText(result_frame, f"You: {gesture_text}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    if player_gesture != "unclear":
        cv2.putText(result_frame, f"Confidence: {confidence:.2f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Computer's gesture
    cv2.putText(result_frame, f"Computer: {computer_gesture.upper()}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    # Result with enhanced styling
    if winner_text == "Player wins":
        result_color = (0, 255, 0)  # Green
    elif winner_text == "Computer wins":
        result_color = (0, 0, 255)  # Red
    else:
        result_color = (255, 255, 0)  # Yellow

    cv2.putText(result_frame, winner_text.upper(), (10, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 4)

    # Round number and tracking info
    cv2.putText(result_frame, f"Round {round_count + 1}", (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    if landmarks is not None:
        cv2.putText(result_frame, f"Hand Landmarks: {len(landmarks)} tracked", (10, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow('Enhanced Rock Paper Scissors Game', result_frame)
    cv2.waitKey(RESULT_DISPLAY_TIME)

    return player_score, computer_score, result

def main():
    """Enhanced main game loop with advanced hand tracking."""
    print("Loading Enhanced Rock Paper Scissors Game with Hand Tracking...")
    print("=" * 70)

    # Load model
    try:
        global model
        model = load_model(MODEL_PATH)
        print("✓ Gesture classification model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Model file '{MODEL_PATH}' not found.")
        print("  Run model.py first to train the model.")
        return

    # Initialize hand tracker
    print("Initializing advanced hand tracking...")
    hand_tracker = HandTracker()
    print("✓ Hand tracking initialized")

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Could not access camera.")
        return
    print("✓ Camera initialized")

    print("\n" + "=" * 70)
    print("🎮 ENHANCED ROCK PAPER SCISSORS GAME")
    print("   Featuring Advanced Hand Tracking & Visual Feedback")
    print("=" * 70)
    print("🎯 New Features:")
    print("   • Real-time hand landmark detection")
    print("   • Colored landmark connections for each finger")
    print("   • Smooth tracking with interpolation")
    print("   • Enhanced gesture confidence visualization")
    print("   • Hand visibility detection")
    print("=" * 70)
    print("🎮 Instructions:")
    print("   • Position your hand in the blue rectangle")
    print("   • Press 's' to start a round")
    print("   • Watch the colored landmark connections")
    print("   • Make your gesture during the countdown")
    print("   • Press 'q' to quit anytime")
    print("=" * 70)

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

            # Detect hand and get landmarks for live preview
            landmarks, bbox = hand_tracker.detect_hand(frame)

            # Define ROI
            height, width = frame.shape[:2]
            roi_coords = (int(width*ROI_MARGIN), int(height*ROI_MARGIN),
                         int(width*(1-ROI_MARGIN)), int(height*(1-ROI_MARGIN)))

            # Get live gesture prediction for preview
            roi = frame[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
            if roi.size > 0:  # Check if ROI is valid
                live_gesture, live_confidence = get_prediction(roi)
            else:
                live_gesture, live_confidence = None, None

            # Draw enhanced UI with hand tracking
            frame = draw_enhanced_ui(frame, player_score, computer_score, roi_coords,
                                   landmarks, live_gesture, live_confidence)

            # Draw hand landmarks
            frame = hand_tracker.draw_hand_landmarks(frame, landmarks, bbox)

            # Display frame
            cv2.imshow('Enhanced Rock Paper Scissors Game', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting enhanced game...")
                break
            elif key == ord('s'):
                # Play an enhanced round with full hand tracking
                player_score, computer_score, round_result = play_enhanced_round(
                    cap, model, hand_tracker, player_score, computer_score, round_count)
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

        # Final results with enhanced summary
        print("\n" + "=" * 70)
        print("🎮 GAME OVER - Enhanced Session Summary")
        print("=" * 70)
        print(f"🎯 Final Score - You: {player_score}, Computer: {computer_score}")
        print(f"🎲 Rounds played: {round_count}")

        if round_count > 0:
            win_rate = (player_score / round_count) * 100
            print(f"🎯 Your win rate: {win_rate:.1f}%")
            if player_score > computer_score:
                print("🎉 Congratulations! You won the enhanced game!")
                print("🏆 Your gesture recognition and timing were excellent!")
            elif player_score < computer_score:
                print("🤖 Computer wins! Great effort though!")
                print("💡 Try adjusting your hand position or lighting for better detection.")
            else:
                print("🤝 It's a tie! Well played on both sides!")
                print("🎯 The AI is learning from your gestures!")
        else:
            print("❓ No rounds played.")

        print("\n" + "=" * 70)
        print("🚀 Thank you for trying the Enhanced Rock Paper Scissors!")
        print("   This demonstrates advanced computer vision and ML integration.")
        print("=" * 70)

if __name__ == "__main__":
    main()