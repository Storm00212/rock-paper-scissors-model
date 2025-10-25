# Rock Paper Scissors with Computer Vision 🤖✂️📄

An AI-powered rock-paper-scissors game that uses computer vision and machine learning to detect hand gestures in real-time through your webcam. Challenge the computer to a game of rock-paper-scissors using gesture recognition!

## 🎓 Machine Learning Learning Pathway

This project serves as a comprehensive introduction to machine learning concepts through hands-on computer vision. Follow the structured learning path below to understand the complete ML pipeline:

### 📚 Study Guide - Learn ML Through This Project

#### Phase 1: Data Preparation & Preprocessing
**File to Study**: `preprocess.py`
**Concepts Covered**:
- **Data Preprocessing**: Converting raw images into ML-ready format
- **Feature Engineering**: Transforming images into numerical representations
- **Data Splitting**: Train/Validation/Test sets and why stratification matters
- **One-Hot Encoding**: Converting categorical labels to numerical format
- **Data Normalization**: Scaling pixel values for better training convergence

**Key Questions to Explore**:
- Why do we resize images to 64x64 pixels?
- What happens if we don't normalize pixel values to [0,1]?
- Why split data into train/val/test sets instead of using all data for training?

#### Phase 2: Model Architecture & Training
**File to Study**: `model.py`
**Concepts Covered**:
- **Convolutional Neural Networks (CNNs)**: Why CNNs work better than regular neural networks for images
- **Feature Learning**: How CNNs automatically learn hierarchical features (edges → textures → shapes)
- **Backpropagation**: The algorithm that trains neural networks
- **Gradient Descent**: Mathematical optimization for finding best model parameters
- **Loss Functions**: Categorical cross-entropy for multi-class classification
- **Optimizers**: Adam optimizer and adaptive learning rates
- **Overfitting Prevention**: Dropout, early stopping, and validation monitoring

**Key Questions to Explore**:
- Why do CNNs have convolutional layers followed by pooling layers?
- What does "training loss" vs "validation loss" tell us about model performance?
- How does early stopping prevent overfitting?

#### Phase 3: Model Evaluation & Testing
**File to Study**: `test_system.py`
**Concepts Covered**:
- **Model Evaluation**: Testing on unseen data to measure generalization
- **Performance Metrics**: Accuracy, confidence scores, error analysis
- **Cross-Validation Concepts**: Ensuring robust performance evaluation
- **Confidence Interpretation**: Understanding prediction certainty
- **System Integration Testing**: Validating end-to-end ML pipelines

**Key Questions to Explore**:
- Why test on a separate dataset instead of training data?
- What do confidence scores tell us about model reliability?
- How do we know if our model is actually learning vs just memorizing?

#### Phase 4: Real-Time Deployment & Computer Vision
**File to Study**: `real_time_cv.py`
**Concepts Covered**:
- **Computer Vision Pipelines**: From video frames to ML predictions
- **Real-Time Inference**: Applying trained models to live data streams
- **Region of Interest (ROI)**: Focusing models on relevant image regions
- **Model Deployment**: Using trained models in production applications
- **Latency vs Accuracy Trade-offs**: Balancing speed and performance
- **Edge Case Handling**: Managing poor lighting, motion blur, camera issues

**Key Questions to Explore**:
- Why do real-time applications need to balance speed vs accuracy?
- How does ROI selection improve model performance?
- What challenges arise when deploying ML models to real-world applications?

#### Phase 5: Human-AI Interaction & UX Design
**File to Study**: `game.py`
**Concepts Covered**:
- **Human-AI Interaction**: Designing intuitive interfaces for ML systems
- **Confidence Thresholds**: Using prediction probabilities to handle uncertainty
- **User Experience Design**: Balancing technical accuracy with user enjoyment
- **Error Handling**: Gracefully managing model failures and edge cases
- **Feedback Loops**: Using user interactions to improve system performance

**Key Questions to Explore**:
- How should ML systems handle uncertainty in user-facing applications?
- Why is user experience more important than technical perfection in games?
- How do we design systems that users trust despite occasional errors?

### 🛣️ Recommended Learning Order

1. **Start Here**: Read the README overview and run the complete system
2. **Phase 1**: Study `preprocess.py` - understand data preparation
3. **Phase 2**: Study `model.py` - learn about CNNs and training
4. **Phase 3**: Study `test_system.py` - understand evaluation and testing
5. **Phase 4**: Study `real_time_cv.py` - explore computer vision deployment
6. **Phase 5**: Study `game.py` - learn about human-AI interaction

### 📖 Additional Resources

After completing this project, explore these topics:
- **Advanced CNN Architectures**: ResNet, DenseNet, EfficientNet
- **Data Augmentation**: Artificially increasing dataset size
- **Transfer Learning**: Using pre-trained models for better performance
- **Model Interpretability**: Understanding what neural networks learn
- **Production Deployment**: Serving ML models at scale

### 🎯 Learning Objectives

By the end of this project, you will understand:
- ✅ Complete machine learning pipeline from data to deployment
- ✅ How convolutional neural networks work for image classification
- ✅ Best practices for training, evaluating, and deploying ML models
- ✅ Real-world challenges in computer vision applications
- ✅ Balancing technical performance with user experience
- ✅ Debugging and improving machine learning systems

## 🎮 Features

- **Real-time Gesture Recognition**: Uses a trained CNN model to detect rock, paper, and scissors gestures
- **Interactive Gameplay**: Play against the computer with live scoring and round-by-round results
- **Computer Vision**: Webcam-based gesture detection with visual feedback
- **High Accuracy**: CNN model trained to ~97% accuracy on test data
- **Live Demo Mode**: Test gesture detection without playing the game
- **Comprehensive Testing**: Built-in system testing and validation

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Webcam/camera
- Required libraries (automatically installed via requirements.txt)

### Installation

1. **Clone or download the project**
   ```bash
   git clone https://github.com/Storm00212/rock-paper-scissors-model.git
   cd rock-paper-scissors
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow opencv-python numpy matplotlib scikit-learn
   ```

3. **Generate preprocessed data**
   ```bash
   python preprocess.py
   ```
   This will create `preprocessed_data.npz` locally (~205MB)

4. **Train the model** (if not already done)
   ```bash
   python model.py
   ```

5. **Test the system**
   ```bash
   python test_system.py
   ```

6. **Play the game!**
   ```bash
   python game.py
   ```

## 🎯 How to Play

1. **Launch the game**: Run `python game.py`
2. **Position yourself**: Make sure your webcam can see you clearly
3. **Get ready**: Position your hand in the blue rectangle on screen
4. **Start rounds**: Press 's' to begin each round
5. **Make gestures**: During the 3-second countdown, form your gesture (rock ✊, paper ✋, or scissors ✌️)
6. **See results**: Watch the computer make its move and see who wins!
7. **Keep playing**: Press 'q' to quit anytime

### Game Rules
- ✊ **Rock** beats ✌️ **Scissors**
- ✋ **Paper** beats ✊ **Rock**
- ✌️ **Scissors** beats ✋ **Paper**
- Same gestures = Tie

## 📁 Project Structure

```
rock-paper-scissors/
│
├── dataset/                    # Training images
│   ├── rock/                  # Rock gesture images
│   ├── paper/                 # Paper gesture images
│   └── scissors/              # Scissors gesture images
│
├── preprocess.py              # Data preprocessing and splitting
├── model.py                   # CNN model training and evaluation
├── real_time_cv.py            # Real-time gesture detection demo
├── game.py                    # Interactive rock-paper-scissors game
├── test_system.py             # System testing and validation
│
├── preprocessed_data.npz      # Processed training data (generated)
├── rock_paper_scissors_model.h5  # Trained model (generated)
├── training_history.png       # Training curves (generated)
│
└── README.md                  # This file
```

## 🔧 Technical Details

### Model Architecture

The system uses a Convolutional Neural Network (CNN) with the following architecture:

- **Input**: 64x64 RGB images
- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Conv2D Layer 3**: 128 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Flatten**: Convert 2D features to 1D
- **Dense Layer**: 128 neurons, ReLU activation
- **Dropout**: 50% dropout for regularization
- **Output Layer**: 3 neurons, softmax activation

### Training Details

- **Dataset**: ~2,200 images total (712 rock, 726 paper, 750 scissors)
- **Split**: 70% training, 15% validation, 15% testing
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-entropy
- **Early Stopping**: Monitors validation loss, patience=5 epochs
- **Data Augmentation**: Normalization (0-1 range)

### Performance

- **Test Accuracy**: ~97%
- **Training Time**: ~5-10 minutes (depending on hardware)
- **Real-time FPS**: Smooth performance on most modern systems

## 🎮 Usage Examples

### Play the Game
```bash
python game.py
```
Interactive game with scoring and visual feedback.

### Test Gesture Detection
```bash
python real_time_cv.py
```
Real-time gesture detection demo without gameplay.

### Run System Tests
```bash
python test_system.py
```
Comprehensive testing of model accuracy and real-time capabilities.

### Train New Model
```bash
python model.py
```
Train the CNN model from scratch (requires preprocessed data).

## 🛠️ Customization

### Adjust Gesture Detection
- Modify `CONFIDENCE_THRESHOLD` in scripts for sensitivity
- Change `ROI_MARGIN` to adjust detection area
- Update `IMG_SIZE` for different input resolutions

### Model Improvements
- Add more training data for better accuracy
- Experiment with different architectures
- Implement data augmentation techniques

### Game Features
- Add sound effects or background music
- Implement difficulty levels
- Add multiplayer support

## 🐛 Troubleshooting

### Common Issues

**"Model file not found"**
- Run `python model.py` to train the model first

**"Camera not accessible"**
- Check camera permissions
- Ensure no other applications are using the camera
- Try different camera index in `cv2.VideoCapture(0)`

**Low accuracy**
- Ensure proper lighting for gesture detection
- Position hand clearly within the blue rectangle
- Retrain model with more diverse training data

**Slow performance**
- Close other applications
- Reduce model complexity if needed
- Use GPU acceleration for TensorFlow

### System Requirements

- **OS**: Windows, macOS, Linux
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: Any webcam or built-in camera

## 📊 Model Training Results

The training process generates:
- **Model file**: `rock_paper_scissors_model.h5`
- **Training history plot**: `training_history.png`
- **Performance metrics**: Accuracy and loss curves

Typical training results:
- Final training accuracy: ~98%
- Final validation accuracy: ~97%
- Test accuracy: ~97%

## 🤝 Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset source: Rock-Paper-Scissors image dataset
- Built with TensorFlow, OpenCV, and other open-source libraries
- Inspired by classic rock-paper-scissors games

---

**Enjoy playing rock-paper-scissors with AI!** 🎉

For questions or issues, please check the troubleshooting section or create an issue in the repository.