# 🎵 Mini-Shazam: Deep Audio Fingerprinting

An end-to-end Machine Learning pipeline that identifies songs in highly noisy environments. Instead of relying on traditional classification, this project treats audio as images and uses **Metric Learning** to map acoustic signatures into a 128-dimensional mathematical space.

## Architecture Overview

1. **Audio Engineering:** Raw `.mp3` files are sliced into 5-second overlapping chunks. Background noise is mathematically overlaid using dynamic Signal-to-Noise Ratio (SNR) augmentation to force the model to ignore volume levels.
2. **Computer Vision for Audio:** The waveforms are converted into Decibel Mel Spectrograms, effectively transforming the audio problem into an image recognition problem.
3. **Modified ResNet-18 Front-End:** A pre-trained ResNet-18 CNN was gutted and modified to accept 1-channel mono spectrograms and output normalized spatial coordinates.
4. **Triplet Margin Loss:** The model bypasses standard Cross-Entropy entirely. Using `pytorch-metric-learning` with Online Hard Negative Mining, the model is penalized based on the Euclidean distance between "Anchor" (noisy audio) and "Positive" (clean audio) embeddings.
5. **Vector Search Inference:** The system compares new audio against a pre-computed database of pristine audio fingerprints to find the closest Euclidean match.

## Performance & Stress Testing

To prove the model wasn't simply suffering from Data Leakage or memorizing specific noise patterns, it was subjected to an **80/20 Train/Test Split** with heavily altered, out-of-distribution noise on the test set.

* **Standard Validation Accuracy:** `98.68%`
* **Heavy Distortion Stress Test:** `86.03%`

The 86% stress-test accuracy proves the model successfully learned to isolate the underlying melodic structures of the audio, remaining highly robust even when the acoustic data was severely degraded.

## Tech Stack
* **Deep Learning:** PyTorch, TorchAudio, TorchVision
* **Metric Learning:** PyTorch-Metric-Learning (Hard Triplet Miner)
* **Vector Search:** FAISS
* **Audio Processing:** PyDub

## 🚀 How to Run Inference

If you have a pre-trained `.pth` weights file, you can test the model on a raw audio recording using the inference script:

```python
import torch
from shazam_inference import shazam_this

# Load model and FAISS/PyTorch database
model.load_state_dict(torch.load("models/audio_resnet_epoch_20.pth"))

# Predict a raw audio file
predicted_song = shazam_this(
    "my_phone_recording.wav", 
    model, 
    database_matrix, 
    database_labels
)