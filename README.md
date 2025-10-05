# Garbage Classification using Deep Learning

This project involves the development and evaluation of a deep learning model to classify garbage into different categories. The goal is to aid in effective waste management and recycling by leveraging artificial intelligence.

## Overview

This project implements a Convolutional Neural Network (CNN) model to classify images of garbage into predefined categories. The model is trained, validated, and evaluated using labeled datasets.

## Features

- **Model Architecture**: Utilizes a sequential CNN built with TensorFlow and Keras.
- **Performance**: Achieved an accuracy of **90.12%** on the validation dataset.
- **Visualization**: Includes confusion matrix visualizations to analyze classification performance.
- **Deployment**: The trained model is saved and can be loaded for further inference.

## Workflow

1. **Data Preprocessing**:
   - Resized and normalized image data for model input.
   - Split data into training, validation, and test sets.

2. **Model Training**:
   - Implemented a sequential CNN model.
   - Optimized using appropriate loss functions and optimizers.

3. **Evaluation**:
   - Computed accuracy metrics.
   - Visualized results using a confusion matrix.

4. **Model Saving and Loading**:
   - Saved the trained model using Python's `pickle` module.
   - Reloaded the model for inference.

## Libraries and Tools

- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle

## Results

- **Accuracy**: The model achieved an accuracy of 98.85% on the validation dataset.
- **Confusion Matrix**:
  The confusion matrix highlights the performance across all categories, with minimal misclassifications.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/garbage-classification.git
