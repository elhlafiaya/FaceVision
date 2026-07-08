# Hybrid Face Recognition System using FaceNet and Machine Learning

> Master's Final Project – Computer Science  
> Faculty of Sciences, Mohammed V University – Rabat, Morocco

---

## Overview

This repository presents a complete hybrid biometric authentication system based on deep learning and machine learning.

The project includes:

- Face detection and alignment using MTCNN.
- Face embedding extraction using FaceNet (512-dimensional embeddings).
- Subject-level train/test split.
- Controlled generation of genuine and impostor pairs.
- Cosine distance computation.
- Support Vector Machine (SVM) classifier with Platt probability calibration.
- Performance evaluation using biometric metrics.
- Interactive Flask web application for authentication and visualization.

The implementation follows a reproducible experimental protocol suitable for academic research.

---


## Research Pipeline

```
Face Images
      │
      ▼
Face Detection (MTCNN)
      │
      ▼
Face Alignment
      │
      ▼
FaceNet
512-D Embeddings
      │
      ▼
L2 Normalization
      │
      ▼
Pair Generation
      │
      ▼
Cosine Distance
      │
      ▼
StandardScaler
      │
      ▼
Support Vector Machine
      │
      ▼
Platt Scaling
      │
      ▼
Authentication Decision
```

---

## Web Platform

The repository also includes a Flask-based web application that allows:

- dataset exploration
- image visualization
- authentication simulation
- prediction confidence visualization
- biometric performance display
- comparison between baseline and SVM approaches

---

## Datasets

Experiments were conducted using:

- ORL Face Database
- Yale Face Database

The datasets are public research datasets.

Due to GitHub storage limitations, the datasets are **not included** in this repository.

---

## Experimental Protocol

The implementation follows good scientific practices:

- subject-level train/test split
- reproducible random seed
- independent pair generation
- no identity leakage
- calibrated probability estimation
- standardized preprocessing

---

## Evaluation Metrics

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- FAR
- FRR
- EER
- ROC Curve
- AUC
- DET Curve

---

## Technologies

- Python
- PyTorch
- FaceNet
- facenet-pytorch
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Flask
- Matplotlib
- Joblib

---

## Installation

```bash
pip install -r requirements.txt
```

Run the web application:

```bash
python app.py
```

---



## Copyright

© 2026Author

All Rights Reserved.
