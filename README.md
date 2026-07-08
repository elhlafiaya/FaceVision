# Hybrid Face Recognition System using FaceNet and Machine Learning

> **Master's Final Project – Computer Science**
> Faculty of Sciences, Mohammed V University – Rabat, Morocco

---

## Overview

This repository presents a hybrid face recognition system that combines deep facial representations with supervised machine learning for biometric authentication.

The objective of this project is to investigate whether a learning-based decision model can improve face verification performance compared with a conventional similarity-threshold approach.

The proposed framework integrates FaceNet for facial feature extraction and a calibrated Support Vector Machine (SVM) for authentication decision-making. A Flask-based web application is also provided to demonstrate the complete authentication workflow through an interactive interface.

---

## Main Features

* Face detection and alignment using MTCNN
* Deep facial embedding extraction with FaceNet
* Subject-level train/test partitioning
* Controlled generation of genuine and impostor verification pairs
* Cosine-distance based facial comparison
* Probability estimation using a calibrated Support Vector Machine
* Comparison with a conventional threshold-based verification system
* Interactive web application for visualization and authentication

---

## Research Methodology

The experimental workflow consists of the following stages:

```text
Face Images
      │
      ▼
Face Detection and Alignment (MTCNN)
      │
      ▼
FaceNet Embedding Extraction
      │
      ▼
Embedding Normalization
      │
      ▼
Verification Pair Generation
      │
      ▼
Cosine Distance Computation
      │
      ▼
Feature Standardization
      │
      ▼
Support Vector Machine
      │
      ▼
Probability Calibration
      │
      ▼
Authentication Decision
```

The entire pipeline was designed to ensure reproducibility through subject-level data partitioning, independent pair generation, deterministic preprocessing, and calibrated probabilistic prediction.

---

## Datasets

Experimental validation was conducted using two publicly available benchmark datasets:

* ORL Face Database
* Yale Face Database

To keep the repository lightweight, the original datasets are not distributed with this project.

---

## Evaluation

The proposed approach is evaluated using standard biometric verification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* False Acceptance Rate (FAR)
* False Rejection Rate (FRR)
* Equal Error Rate (EER)
* ROC Curve
* Area Under the Curve (AUC)
* Detection Error Tradeoff (DET) Curve

---

## Technologies

* Python
* PyTorch
* FaceNet (InceptionResnetV1)
* facenet-pytorch
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Flask
* Matplotlib
* Joblib

---

## Getting Started

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch the web application:

```bash
python app.py
```

---

## Notice

This repository is shared exclusively for academic reading and evaluation purposes.

Please do not copy, modify, redistribute, or reuse any part of this project without prior permission from the author.
