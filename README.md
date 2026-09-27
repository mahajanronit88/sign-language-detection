# Sign Language Detection

Real-time sign language detection system using computer vision and machine learning. Detects hand gestures and classifies them into alphabets (A-Z) and common words like Hello, Yes, No, Please, Thank You.

## Tech Stack
- OpenCV
- MediaPipe
- CVZone
- TensorFlow / Keras
- NumPy

## How it works
1. Hand is detected and tracked in real-time using MediaPipe (via CVZone's HandDetector)
2. The hand region is cropped and resized into a fixed 300x300 image
3. A trained classification model (built using Google Teachable Machine) predicts the sign
4. The predicted label is displayed live on screen

## Files
- `sign.py` — Script used to collect hand gesture training images
- `test.py` — Main script for real-time sign detection and prediction
- `Model/` — Trained model (`keras_model.h5`) and labels (`labels.txt`)

## Signs Supported
A-Z, Hello, Yes, No, Please, Okay, I Love You, You
