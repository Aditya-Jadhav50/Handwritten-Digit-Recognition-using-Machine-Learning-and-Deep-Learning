# Handwritten-Digit-Recognition-using-Machine-Learning-and-Deep-Learning

## MNIST Dataset
The MNIST dataset consists of handwritten digits from 0 to 9, with images of size 28x28 pixels. It contains 70,000 images: 60,000 for training and 10,000 for testing. The images are grayscale and centered for easier recognition.

## Requirements
- Python 3.5 or higher
- Scikit-learn (latest version)
- Numpy (with MKL support for Windows)
- Matplotlib
- Keras (with TensorFlow backend)

## Introduction
MNIST is a widely used dataset for training machine learning models to recognize handwritten digits. It includes 60,000 training images and 10,000 test images. The images are 28x28 pixels and are preprocessed to reduce noise and improve recognition accuracy.

This project utilizes Keras, a high-level neural network API, to create a Convolutional Neural Network (CNN) for digit recognition. Keras is user-friendly, supports rapid prototyping, and is compatible with deep learning frameworks like TensorFlow, Theano, and CNTK.

## Description
This project implements a 5-layer Sequential Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The model is built using the Keras API with the TensorFlow backend, which simplifies the process of creating and training deep learning models.

## Model Accuracy
- **Training Accuracy**: The CNN achieved 99.51% accuracy when trained on a GPU, taking approximately one minute to complete.
- **Test Accuracy**: On the test set, the model achieved 98.15% accuracy.

If you don't have access to a GPU, training might take a bit longer. You can reduce the number of epochs to speed up training.

## How to Run
1. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

2. Download the MNIST dataset and place it in the appropriate folder (if required by your setup).

3. Run the training script:
    ```bash
    python train_model.py
    ```

4. To make predictions with the trained model, use the following script:
    ```bash
    python predict_digit.py
    ```

## Notes
- If you're running this on a CPU, training may take longer, and you may need to adjust the number of epochs or batch size.
- Ensure that you have a GPU-enabled version of TensorFlow installed for optimal performance.
