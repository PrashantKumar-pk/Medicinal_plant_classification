# Medicinal Plant Classification

This project implements a Convolutional Neural Network (CNN) to classify medicinal plants using TensorFlow and Keras.

## Overview

The model is trained on a dataset of medicinal plant images and can classify plants into 30 different categories. It achieves high accuracy in identifying various medicinal plant species.

## Features

- Uses TensorFlow and Keras for model architecture
- Implements data augmentation techniques
- Achieves high accuracy on test data
- Includes visualization of training and validation metrics
- Provides functions for model prediction on new images

## Model Architecture

The CNN model consists of:
- Multiple convolutional and max pooling layers
- Flatten layer
- Dense layers with ReLU and Softmax activations

## Dataset

The dataset used is the "Medicinal_Plant_Dataset_(Augmented)", which includes images of 30 different medicinal plant species taken from https://www.kaggle.com/datasets/vishnuoum/medicinal-plant-dataset-augmented/

## Results

The model achieves 98% accuracy on the test dataset after 50 epochs of training.

## Usage

To use this model:
1. Ensure you have the required dependencies installed
2. Run the script to train the model
3. Use the provided prediction function to classify new plant images

## Future Improvements

- Experiment with different model architectures
- Expand the dataset with more plant species
- Implement transfer learning with pre-trained models
