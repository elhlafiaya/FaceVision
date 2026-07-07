# Hybrid Face Recognition System using FaceNet and Machine Learning

## Overview

This repository presents a hybrid biometric authentication system for face recognition developed as part of a Master's Final Project.

The proposed system combines deep facial embeddings extracted by FaceNet with a supervised Machine Learning classifier in order to improve authentication reliability compared to a traditional fixed-threshold approach.

The project follows a rigorous experimental protocol based on subject-level data splitting, reproducible pair generation, and biometric evaluation metrics.

---

## Features

- Face detection and alignment using MTCNN
- Face embedding extraction using FaceNet (512-dimensional embeddings)
- Subject-level train/test split
- Controlled generation of genuine and impostor pairs
- Cosine similarity computation
- Support Vector Machine (SVM) classifier
- Probability calibration (Platt Scaling)
- Automatic decision based on Equal Error Rate (EER)
- Performance comparison with a classical threshold-based system
- Interactive Flask web application for visualization

---

## Project Pipeline

Dataset

↓

Subject-Level Split

↓

Face Detection (MTCNN)

↓

Face Alignment

↓

FaceNet Embedding Extraction

↓

L2 Normalization

↓

Pair Generation

↓

Cosine Distance Computation

↓

Feature Scaling

↓

SVM Training

↓

Probability Calibration

↓

Authentication Decision

↓

Performance Evaluation

---

## Datasets

The experiments were conducted using two publicly available datasets:

- ORL Face Database
- Yale Face Database

Both datasets were split at the subject level to guarantee that no identity appears simultaneously in the training and testing sets.

---

## Machine Learning Model

Classifier:

- Support Vector Machine (RBF Kernel)

Calibration:

- Platt Scaling

Feature:

- Cosine distance between FaceNet embeddings

Evaluation Metrics:

- Accuracy
- FAR
- FRR
- EER
- ROC Curve
- AUC
- DET Curve

---

## Technologies

Python

PyTorch

FaceNet

MTCNN

OpenCV

Scikit-learn

NumPy

Pandas

Matplotlib

Flask

Joblib

---

## Repository Structure

```
.
├── datasets/
│
├── notebooks/
│
├── models/
│
├── embeddings/
│
├── app/
│
├── results/
│
├── figures/
│
├── README.md
│
└── requirements.txt
```

---

## Experimental Protocol

The implementation follows several good practices:

- subject-level train/test split
- reproducible random seed
- independent train/test pair generation
- balanced genuine/impostor pairs
- standardized feature scaling
- calibrated probability estimation
- evaluation following biometric recognition standards

---

## Results

The proposed hybrid approach demonstrates improved authentication performance compared to a classical fixed-threshold decision by reducing both False Acceptance Rate (FAR) and False Rejection Rate (FRR) while maintaining a low Equal Error Rate (EER).

---

## Citation

If you use this repository, please cite:

Master's Final Project

Hybrid Face Recognition System using FaceNet and Machine Learning

Faculty of Sciences

Mohammed V University

Morocco

2026

---

## License

This project is released for academic and research purposes.
