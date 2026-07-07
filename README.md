# Hybrid Face Recognition using FaceNet and Support Vector Machines

> A reproducible biometric authentication framework combining deep facial embeddings and supervised machine learning.

---

## Abstract

Face recognition has become one of the most widely adopted biometric authentication technologies. Traditional verification systems generally rely on a fixed similarity threshold, which often struggles to generalize under varying acquisition conditions.

This project proposes a hybrid biometric authentication framework that combines **FaceNet embeddings** with a **Support Vector Machine (SVM)** classifier calibrated through **Platt Scaling** to improve verification performance. Rather than relying solely on a manually selected threshold, the proposed system learns the decision boundary directly from facial similarity scores.

The implementation follows a rigorous experimental protocol based on **subject-level data partitioning**, reproducible pair generation, and standardized biometric evaluation metrics.

---

## Research Contributions

The main contributions of this work include:

- Subject-level train/test partitioning to eliminate identity leakage.
- Reproducible generation of genuine and impostor verification pairs.
- Deep facial representation using FaceNet.
- Similarity-based verification using cosine distance.
- Probability estimation through Platt-calibrated SVM.
- Comparison with a conventional threshold-based biometric system.
- Evaluation following ISO/IEC biometric assessment practices.

---

## System Architecture

```text
Face Images
      │
      ▼
Face Detection & Alignment (MTCNN)
      │
      ▼
Image Preprocessing
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
Platt Probability Calibration
      │
      ▼
Authentication Decision
```

---

## Experimental Pipeline

### 1. Dataset Preparation

The datasets are organized by subject identity.

Each identity belongs **exclusively** to either the training or testing subset.

This protocol prevents data leakage and provides a more realistic biometric evaluation.

---

### 2. Face Detection

Faces are automatically detected and aligned using **MTCNN**.

Detected faces are resized to **160 × 160** pixels before embedding extraction.

---

### 3. Deep Feature Extraction

Face representations are extracted using the pretrained **FaceNet (InceptionResnetV1)** network.

Each image is converted into a normalized **512-dimensional embedding**.

---

### 4. Pair Construction

Verification pairs are generated independently for the training and testing datasets.

Two pair categories are considered:

- Genuine pairs (same identity)
- Impostor pairs (different identities)

Pair generation is fully reproducible through a fixed random seed.

---

### 5. Similarity Computation

For each verification pair, the cosine distance between embeddings is computed.

This distance represents the input feature used for classification.

---

### 6. Machine Learning

Classifier:

- Support Vector Machine (RBF Kernel)

Calibration:

- Platt Scaling

Feature Scaling:

- StandardScaler

---

### 7. Decision

The calibrated classifier predicts the probability that two facial embeddings belong to the same individual.

Authentication decisions are then derived from the learned model instead of relying on a manually selected threshold.

---

## Datasets

Experiments were conducted on two publicly available benchmark datasets.

| Dataset | Purpose |
|----------|----------|
| ORL Face Database | Controlled facial verification |
| Yale Face Database | Illumination robustness evaluation |

---

## Evaluation Metrics

Performance is evaluated using classical biometric verification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Equal Error Rate (EER)
- ROC Curve
- Area Under the Curve (AUC)
- DET Curve

---


## Software Stack

- Python
- PyTorch
- FaceNet
- facenet-pytorch
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Flask
- Joblib

---

## Reproducibility

The implementation has been designed to maximize experimental reproducibility.

Key reproducibility measures include:

- fixed random seed
- subject-level partitioning
- independent train/test pair generation
- deterministic preprocessing
- standardized feature scaling
- serialized trained models

---

## Future Work

Potential research directions include:

- ArcFace embeddings
- MagFace
- AdaFace
- Quality-aware biometric verification
- Open-set face recognition
- Deep metric learning
- Domain adaptation
- Multimodal biometric authentication

---

## References

1. Schroff F., Kalenichenko D., Philbin J.
   FaceNet: A Unified Embedding for Face Recognition and Clustering.
   CVPR 2015.

2. Taigman Y. et al.
   DeepFace.
   CVPR 2014.

3. Deng J. et al.
   ArcFace.
   CVPR 2019.

4. ISO/IEC 19795-1
   Biometric Performance Testing and Reporting.

---


## License

Released for academic and research purposes.
