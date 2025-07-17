# Speaker Recognition System: CNN and LSTM Combination

## Overview

This project implements a speaker recognition system using a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks.  The system is trained to identify speakers from audio samples, leveraging different audio feature extraction techniques. The following aspects are covered in this project:

*   **Feature Extraction**: Mel-Frequency Cepstral Coefficients (MFCCs), Mel Spectrograms, and Linear Predictive Coding (LPC).
*   **Model Architectures**: CNN and LSTM models are trained and evaluated.
*   **Data Augmentation:** Background noise augmentation to improve model robustness.
*   **Visualization**: Charts for accuracy comparison and confusion matrices for model performance analysis.
*   **Resource Utilization:** Comparison of model size, parameter count, and inference time.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Dependencies](#dependencies)
4.  [Dataset](#dataset)
5.  [Usage](#usage)
    *   [Installation](#installation)
    *   [Data Preparation](#data-preparation)
    *   [Training](#training)
    *   [Evaluation](#evaluation)
6.  [Model Details](#model-details)
7.  [Results](#results)
8.  [Visualizations](#visualizations)
9.  [Resource Utilization](#resource-utilization)
10. [Future Work](#future-work)
11. [License](#license)
12. [Contact](#contact)

## 1. Introduction

Speaker recognition is the task of identifying individuals based on their voice characteristics. This project explores different deep learning approaches, specifically CNNs and LSTMs, to build a robust and accurate speaker recognition system. It provides a detailed implementation, including data loading, feature extraction, model training, and evaluation, with visualizations to understand the model performance and resource utilization.

## 2. Features

*   **Feature Extraction:**
    *   **MFCC**: Extraction of Mel-Frequency Cepstral Coefficients using the `librosa` library.
    *   **Mel Spectrogram**: Generation and extraction of Mel spectrograms.
    *   **LPC**: Implementation of Linear Predictive Coding for feature extraction.
*   **Data Augmentation:**
    *   Background noise augmentation to enhance the model's ability to handle noisy environments.
*   **Model Training:**
    *   CNN model for image-like feature representations (MFCCs, Mel Spectrograms).
    *   LSTM model for sequential feature analysis (MFCCs, Mel Spectrograms, LPC).
*   **Evaluation Metrics:**
    *   Accuracy, F1-score, Confusion Matrices.
*   **Visualization:**
    *   Accuracy comparison charts to compare the performance of different models and feature extraction methods.
    *   Confusion matrices to visualize the classification performance of the best model.
    *   Resource utilization comparison (model size, parameter count, inference time)
*   **Early Stopping:** Implemented to prevent overfitting during model training.

## 3. Dependencies

The following Python libraries are required to run this project:

*   `librosa`: For audio analysis and feature extraction.
*   `numpy`: For numerical operations.
*   `scikit-learn`: For data splitting, label encoding, and scaling.
*   `tensorflow`: For building and training deep learning models.
*   `matplotlib`: For creating visualizations.
*   `seaborn`: For enhanced visualizations (confusion matrices).
*   `resampy`: For audio resampling.
*   `warnings`: For managing warnings during execution.
*   `os`: For interacting with the operating system.
*   `time`: For measuring training and inference time.

You can install these dependencies using pip:

```bash
pip install librosa numpy scikit-learn tensorflow matplotlib seaborn resampy
