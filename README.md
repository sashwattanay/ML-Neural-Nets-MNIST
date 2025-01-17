# MNIST Digit Classification with Neural Networks

## Overview
This project focuses on classifying handwritten digits from the **MNIST dataset** using a **fully connected neural network (Multi-Layer Perceptron)**. The model leverages **Keras with TensorFlow backend** for implementation and demonstrates hyperparameter tuning for optimization.

The objective is to achieve high accuracy in digit classification while minimizing overfitting through techniques like **early stopping** and **validation splitting**.

## Skills Demonstrated
This project showcases the following skills:
1. **Data Preprocessing**:
   - Normalization of pixel values to the range `[0, 1]` for consistent input scaling.
   - One-hot encoding of digit labels for compatibility with the categorical cross-entropy loss.

2. **Neural Network Design and Training**:
   - Implementation of a sequential neural network with:
     - Input layer (28x28 pixels).
     - Hidden layers with `ReLU` activation.
     - Output layer with `softmax` activation for classification.
   - Early stopping and checkpointing for optimized training.

3. **Hyperparameter Tuning**:
   - Explored varying:
     - Learning rates (`5e-3`, `1e-3`, `1e-4`).
     - Neurons in hidden layers (`32` to `256` units).
   - Utilized **Keras Tuner's Hyperband** algorithm for automated tuning.

4. **Model Evaluation**:
   - Achieved robust performance with:
     - Validation accuracy: **97.85%**.
     - Test accuracy: **97.58%**.
   - Generated confusion matrices for detailed error analysis.

## Dataset
The **MNIST dataset** contains grayscale images of handwritten digits (0–9). 
- **Training Samples**: 60,000.
- **Test Samples**: 10,000.
- **Image Dimensions**: 28x28 pixels.

## Results
- **Best Hyperparameters**:
  - Learning rate: **0.001**.
  - Hidden layer neurons: 
    - First layer: **256 units**.
    - Second layer: **128 units**.
- **Final Test Performance**:
  - **Test Accuracy**: **97.58%**.
  - **Test Loss**: **0.0846**.

## Key Features
- **Early Stopping**: Prevents overfitting by halting training when validation loss stops improving.
- **Checkpointing**: Saves the best model during training for final evaluation.
- **Dynamic Learning Rate Scanning**: Ensures optimal learning rate selection using diagnostics.

## Requirements
The project requires:
- Python 3.7 or higher.
- Libraries:
  - `tensorflow`
  - `keras`
  - `numpy`
  - `matplotlib`
  - `keras-tuner`
  - `scikit-learn`
