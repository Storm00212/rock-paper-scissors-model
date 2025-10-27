# Rock Paper Scissors with Computer Vision

An AI-powered rock-paper-scissors game that uses computer vision and machine learning to detect hand gestures in real-time through your webcam. Challenge the computer to a game of rock-paper-scissors using gesture recognition!

##  Machine Learning Learning Pathway

This project serves as a comprehensive introduction to machine learning concepts through hands-on computer vision. Follow the structured learning path below to understand the complete ML pipeline:

###  Study Guide - Learn ML Through This Project

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

###  Recommended Learning Order

1. **Start Here**: Read the README overview and run the complete system
2. **Phase 1**: Study `preprocess.py` - understand data preparation
3. **Phase 2**: Study `model.py` - learn about CNNs and training
4. **Phase 3**: Study `test_system.py` - understand evaluation and testing
5. **Phase 4**: Study `real_time_cv.py` - explore computer vision deployment
6. **Phase 5**: Study `game.py` - learn about human-AI interaction

###  Additional Resources

After completing this project, explore these topics:
- **Advanced CNN Architectures**: ResNet, DenseNet, EfficientNet
- **Data Augmentation**: Artificially increasing dataset size
- **Transfer Learning**: Using pre-trained models for better performance
- **Model Interpretability**: Understanding what neural networks learn
- **Production Deployment**: Serving ML models at scale

###  Learning Objectives

By the end of this project, you will understand:
-  Complete machine learning pipeline from data to deployment
-  How convolutional neural networks work for image classification
-  Best practices for training, evaluating, and deploying ML models
-  Real-world challenges in computer vision applications
-  Balancing technical performance with user experience
-  Debugging and improving machine learning systems

##  Quick Start for Contributors

### Prerequisites for Development
- Python 3.7+
- Webcam/camera for testing
- Git and Git LFS for large file handling

### Development Setup
```bash
# Clone the repository
git clone https://github.com/Storm00212/rock-paper-scissors-model.git
cd rock-paper-scissors

# Install Git LFS and pull large files
git lfs install
git lfs pull

# Install dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn

# Optional: For enhanced hand tracking features
pip install mediapipe  # Recommended for MediaPipe hand tracking

# Run the complete pipeline
python preprocess.py
python model.py
python test_system.py
python game.py  # Play the game!
```

### Repository Structure Notes
- **Large files** (models, datasets) are tracked with Git LFS
- **Generated files** are not committed (use `.gitignore`)
- **Source code** includes comprehensive ML teaching comments
- **Documentation** serves as both guide and curriculum

### Contributing
1. Study the learning pathway in this README
2. Experiment with the code and understand each component
3. Propose improvements or extensions
4. Test thoroughly before submitting changes
5. Update documentation as needed

##  Features

- **Real-time Gesture Recognition**: Uses a trained CNN model to detect rock, paper, and scissors gestures
- **Interactive Gameplay**: Play against the computer with live scoring and round-by-round results
- **Computer Vision**: Webcam-based gesture detection with visual feedback
- **High Accuracy**: CNN model trained to ~97% accuracy on test data
- **Live Demo Mode**: Test gesture detection without playing the game
- **Comprehensive Testing**: Built-in system testing and validation

##  Quick Start

### Prerequisites

Before you begin, ensure you have the following installed on your system:

#### 1. Python 3.7 or Higher

**For Windows:**
1. Visit https://www.python.org/downloads/
2. Download the latest Python 3.7+ installer for Windows
3. Run the installer
4. **Important**: Check the box "Add Python to PATH" during installation
5. Click "Install Now"
6. Verify installation: Open Command Prompt and type `python --version`

**For macOS:**
1. Visit https://www.python.org/downloads/
2. Download the macOS installer
3. Run the installer package
4. Follow the installation wizard
5. Verify installation: Open Terminal and type `python3 --version`

**For Linux (Ubuntu/Debian):**
```bash
# Update package list
sudo apt update

# Install Python 3 and pip
sudo apt install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

**For Linux (CentOS/RHEL/Fedora):**
```bash
# Install Python 3
sudo yum install python3 python3-pip  # CentOS/RHEL
# or
sudo dnf install python3 python3-pip  # Fedora

# Verify installation
python3 --version
```

#### 2. Git Version Control

**For Windows:**
1. Visit https://git-scm.com/download/win
2. Download the installer
3. Run the installer with default settings
4. Verify: Open Command Prompt and type `git --version`

**For macOS:**
```bash
# Install via Homebrew (recommended)
brew install git

# Or download from https://git-scm.com/download/mac
```

**For Linux:**
```bash
# Ubuntu/Debian
sudo apt install git

# CentOS/RHEL
sudo yum install git

# Fedora
sudo dnf install git

# Verify installation
git --version
```

#### 3. Git LFS (Large File Storage)

Git LFS is required to download the trained model files.

**For Windows:**
1. Visit https://git-lfs.github.io/
2. Download the Windows installer
3. Run the installer
4. Verify: Open Command Prompt and type `git lfs version`

**For macOS:**
```bash
# Install via Homebrew
brew install git-lfs

# Or download from https://git-lfs.github.io/
```

**For Linux:**
```bash
# Ubuntu/Debian
sudo apt install git-lfs

# CentOS/RHEL/Fedora
# Download from https://git-lfs.github.io/
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
```

#### 4. Webcam/Camera

- Any built-in webcam or external USB camera
- Ensure camera permissions are granted (especially on macOS/Linux)
- Test your camera works with other applications before proceeding

#### 5. System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, Ubuntu 18.04+ or equivalent Linux
- **RAM**: Minimum 4GB, recommended 8GB
- **Storage**: 2GB free space for installation + generated files
- **Internet**: Required for downloading dependencies and cloning repository

### Installation

#### Step 1: Set Up a Virtual Environment (Recommended)

A virtual environment keeps your project dependencies isolated from your system Python installation.

**For Windows:**
```bash
# Create virtual environment
python -m venv rps_env

# Activate virtual environment
rps_env\Scripts\activate

# You should see (rps_env) in your command prompt
```

**For macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv rps_env

# Activate virtual environment
source rps_env/bin/activate

# You should see (rps_env) in your terminal
```

**To deactivate later:** Type `deactivate` and press Enter.

#### Step 2: Clone the Repository

```bash
# Clone the project
git clone https://github.com/Storm00212/rock-paper-scissors-model.git

# Enter the project directory
cd rock-paper-scissors

# If you created a virtual environment, make sure it's activated first
```

#### Step 3: Install Git LFS and Download Large Files

```bash
# Initialize Git LFS (if not already done globally)
git lfs install

# Pull large files (model and dataset files)
git lfs pull
```

#### Step 4: Install Python Dependencies

```bash
# Install core dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn

# Optional: For enhanced hand tracking features
pip install mediapipe  # Recommended for MediaPipe hand tracking

# Verify installations
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

#### Step 5: Generate Preprocessed Data

```bash
# Generate training data (this creates preprocessed_data.npz ~205MB)
python preprocess.py
```

This step processes the raw images into a format ready for training. It may take a few minutes.

#### Step 6: Train the Model (Optional - Pre-trained model included)

If you want to train your own model:

```bash
# Train the CNN model (takes 5-10 minutes)
python model.py
```

**Note**: A pre-trained model is already included via Git LFS, so this step is optional.

#### Step 7: Test the System

```bash
# Run comprehensive tests to verify everything works
python test_system.py
```

This will test model accuracy and real-time performance.

#### Step 8: Play the Game!

```bash
# Start the interactive rock-paper-scissors game
python game.py
```

**For enhanced version with hand tracking:**
```bash
# Run the advanced version (requires MediaPipe)
python enhanced_game.py
```

### Post-Installation Verification

After completing setup, verify everything works:

```bash
# Quick verification script
python -c "
import sys
print('Python version:', sys.version)
try:
    import tensorflow as tf
    print('✓ TensorFlow available')
except ImportError:
    print('✗ TensorFlow missing')

try:
    import cv2
    print('✓ OpenCV available')
except ImportError:
    print('✗ OpenCV missing')

try:
    import numpy as np
    print('✓ NumPy available')
except ImportError:
    print('✗ NumPy missing')
"
```

If all checks pass, you're ready to play!

##  How to Play

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
- I think you know the damn game though.

##  Project Structure

```
rock-paper-scissors/
│
├── dataset/                    # Training images (712 rock, 726 paper, 750 scissors)
│   ├── rock/                  # Rock gesture images
│   ├── paper/                 # Paper gesture images
│   └── scissors/              # Scissors gesture images
│
├── preprocess.py              # Data preprocessing and splitting
├── model.py                   # CNN model training and evaluation
├── real_time_cv.py            # Real-time gesture detection demo
├── game.py                    # Interactive rock-paper-scissors game
├── enhanced_game.py           # Advanced game with hand landmark tracking
├── hand_tracking.py           # Hand landmark detection and visualization
├── test_system.py             # System testing and validation
│
├── preprocessed_data.npz      # Processed training data (generated, ~205MB)
├── rock_paper_scissors_model.h5  # Trained model (generated, ~12MB)
├── training_history.png       # Training curves (generated)
│
├── .gitattributes             # Git LFS configuration for large files
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

###  Generated Files (Not in Repository)

The following files are generated when you run the scripts and are tracked with Git LFS:
- `preprocessed_data.npz` (~205MB) - Contains train/val/test splits
- `rock_paper_scissors_model.h5` (~12MB) - Trained CNN model weights
- `training_history.png` - Training accuracy/loss curves

These files are too large for standard Git but are handled by Git LFS for proper version control.

###  Enhanced Features Files

- `enhanced_game.py` - Advanced game with real-time hand landmark tracking
- `hand_tracking.py` - Hand detection, landmark extraction, and visualization module

##  Technical Details

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

##  Enhanced Version with Hand Tracking

### Advanced Features

The project now includes an **enhanced version** with advanced hand landmark tracking:

- **Real-time hand landmark detection** using computer vision
- **Colored landmark connections** for each finger (thumb=blue, index=green, etc.)
- **Smooth interpolation** for fluid movement tracking
- **Gesture confidence overlays** on the hand itself
- **Enhanced error handling** for low-light conditions
- **Multi-modal integration** combining CNN classification with landmark tracking

### Enhanced Installation

For the advanced hand tracking features:

```bash
# Install additional dependencies (optional, fallback implementation included)
pip install mediapipe  # For MediaPipe hand tracking (recommended)
# OR use the built-in OpenCV-based fallback implementation
```

### Try the Enhanced Game

```bash
# Run the enhanced version with hand tracking
python enhanced_game.py
```

**New Features in Enhanced Mode:**
-  **Visual landmark feedback**: See colored lines connecting hand joints
-  **Hand visibility detection**: System knows if your hand is fully visible
-  **Live gesture preview**: See your gesture prediction in real-time
-  **Confidence visualization**: Gesture confidence shown directly on hand
-  **Smooth tracking**: Fluid hand movement visualization

### Demo Videos

**Enhanced Hand Tracking Demo:**
- Colored landmark connections for each finger
- Real-time gesture confidence overlays
- Smooth hand movement interpolation
- Low-light condition handling

**Before/After Comparison:**
- Standard version: Basic gesture detection
- Enhanced version: Full hand landmark visualization + improved accuracy

##  Usage Examples

### Play the Game
```bash
python game.py
```
Interactive game with scoring and visual feedback.

### Enhanced Game with Hand Tracking
```bash
python enhanced_game.py
```
Advanced game with real-time hand landmark visualization and improved accuracy.

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

##  Customization

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

##  Troubleshooting

### Installation Issues

**"python is not recognized as an internal or external command" (Windows)**
- Reinstall Python and make sure to check "Add Python to PATH"
- Or use `py` instead of `python` in commands
- Try: `py -m venv rps_env` and `rps_env\Scripts\activate`

**"pip is not recognized"**
- pip is usually installed with Python
- Try: `python -m pip install <package>` instead of `pip install <package>`
- Or reinstall Python with pip included

**"git lfs command not found"**
- Make sure Git LFS is properly installed
- Try reinstalling Git LFS from https://git-lfs.github.io/
- On some systems, you may need to restart your terminal/command prompt

**Virtual environment activation fails**
- Make sure you're using the correct activation command for your OS
- On Windows: `rps_env\Scripts\activate` (not `rps_env/bin/activate`)
- On macOS/Linux: `source rps_env/bin/activate`
- If still failing, try: `python -m venv rps_env --clear` to recreate it

**Permission denied errors during installation**
- On macOS/Linux: Try `sudo pip install` (not recommended) or use `--user` flag
- Better: Use virtual environment which doesn't require admin rights
- On Windows: Run Command Prompt as Administrator

### Runtime Issues

**"Model file not found"**
- Run `python model.py` to train the model first
- Check if `git lfs pull` was successful - the model file should be ~12MB
- Verify you're in the correct directory (`rock-paper-scissors`)

**"Camera not accessible"**
- Check camera permissions in system settings
- Ensure no other applications are using the camera (Zoom, Skype, etc.)
- Try different camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- On macOS: Grant camera access to Terminal/Python in System Preferences > Security & Privacy
- On Linux: Install v4l-utils and check camera with `v4l2-ctl --list-devices`

**"Module not found" errors**
- Make sure virtual environment is activated (you should see `(rps_env)` in prompt)
- Try reinstalling the package: `pip install --upgrade <package_name>`
- Check if you installed in the wrong Python environment

**Low accuracy or poor gesture detection**
- Ensure proper lighting - bright, even lighting works best
- Position hand clearly within the blue rectangle ROI
- Keep background simple and uncluttered
- Retrain model with more diverse training data if needed
- Adjust `CONFIDENCE_THRESHOLD` in the scripts (try lower values like 0.3)

**Slow performance or lag**
- Close other applications using CPU/GPU
- Reduce model complexity if needed (edit model.py)
- Enable GPU acceleration: Install TensorFlow GPU version if you have NVIDIA GPU
- Lower camera resolution in scripts if needed

**OpenCV camera errors**
- Try different camera indices (0, 1, 2, etc.)
- Check camera drivers are up to date
- On Windows: Try running as Administrator
- On Linux: `sudo apt install v4l-utils` and test with `cheese` or similar

**TensorFlow installation issues**
- For CPU-only: `pip install tensorflow-cpu` (smaller download)
- For GPU support: Follow TensorFlow GPU setup guide for your system
- If installation fails, try: `pip install --upgrade pip` first

### Verification Steps

After installation, verify everything works:

```bash
# Check Python and packages
python --version
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
python -c "import cv2; print('CV2:', cv2.__version__)"
python -c "import numpy, matplotlib, sklearn; print('All imports OK')"

# Check Git LFS
git lfs version

# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera failed'); cap.release()"

# Run a quick test
python test_system.py
```

### Getting Help

If you still have issues:
1. Check the [GitHub Issues](https://github.com/Storm00212/rock-paper-scissors-model/issues) for similar problems
2. Create a new issue with:
   - Your operating system and Python version
   - Exact error message
   - Steps you followed
   - Output of verification commands above

### System Requirements

- **OS**: Windows, macOS, Linux
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: Any webcam or built-in camera

##  Model Training Results

The training process generates:
- **Model file**: `rock_paper_scissors_model.h5`
- **Training history plot**: `training_history.png`
- **Performance metrics**: Accuracy and loss curves

Typical training results:
- Final training accuracy: ~98%
- Final validation accuracy: ~97%
- Test accuracy: ~97%

##  Contributing
I am still working on it's accuracy in computer vision.
Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

##  License

This project is open source and available under the MIT License.

##  Acknowledgments

- Dataset source: Rock-Paper-Scissors image dataset
- Built with TensorFlow, OpenCV, and other open-source libraries
- Inspired by classic rock-paper-scissors games

---

**Enjoy playing rock-paper-scissors with AI!** 

For questions or issues, please check the troubleshooting section or create an issue in the repository. Or just email me and we can figure it out.