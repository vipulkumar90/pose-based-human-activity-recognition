# Human Activity Recognition using 2D Pose Estimation

🇯🇵 日本語版はこちら → [README.ja.md](README.ja.md)

A machine learning pipeline for recognizing human activities from 2D pose keypoints. This project focuses on feature engineering, skeleton normalization, and subject-independent evaluation using Leave-One-Subject-Out (LOSO) cross-validation.

---

## Overview

Human Activity Recognition (HAR) is commonly used in healthcare, surveillance, and assistive technologies. Instead of relying on raw images, this project uses 2D human pose keypoints to represent body movement, making the system more computationally efficient while preserving privacy.

The pipeline extracts geometric and temporal features from pose sequences before training traditional machine learning models such as Random Forests.

The primary objective is to build a model that generalizes well to unseen individuals.

---

## Pipeline

```
Raw Pose Keypoints
        │
        ▼
Remove Invalid Frames
        │
        ▼
Skeleton Normalization
(Hip Centering + Body Scaling)
        │
        ▼
Feature Engineering
    • Joint Angles
    • Range of Motion
    • Velocity
    • Acceleration
        │
        ▼
Multi-scale Windowing
(Short + Long Temporal Context)
        │
        ▼
Feature Aggregation
(mean, std, min, max)
        │
        ▼
Random Forest Classifier
        │
        ▼
Leave-One-Subject-Out Evaluation
```

---

## Features

### Skeleton Normalization

To reduce variations caused by body size and camera position:

- Hip center is used as the coordinate origin.
- Coordinates are normalized by torso height (or shoulder width).

This allows poses from different subjects to become directly comparable.

---

### Feature Engineering

The following features are extracted from each frame:

#### Geometric Features

- Joint angles
- Shoulder angles
- Knee angles
- Neck-to-torso angle

#### Range of Motion Features

- Wrist height relative to shoulders
- Wrist height relative to hips
- Wrist-to-nose distance
- Wrist symmetry

#### Temporal Features

- Joint velocity
- Joint acceleration

---

### Multi-scale Temporal Windows

Instead of using a single window size, the pipeline computes statistics over two different temporal scales simultaneously.

Example:

- Short window: 15 frames
- Long window: 60 frames

For every feature, the following statistics are computed:

- Mean
- Standard Deviation
- Minimum
- Maximum

The resulting feature vectors capture both short-term movements and long-term posture.

---

### Model

The primary models evaluated in this project are:

- Random Forest (baseline)
- Extra Trees
- XGBoost
- LSTM (Deep Learning)

Random Forest serves as the baseline model throughout the feature engineering experiments. Extra Trees and XGBoost are evaluated using the optimized preprocessing pipeline and hyperparameter tuning to compare performance.

The project also includes modular model builders and default hyperparameter configurations for additional classifiers (e.g., Logistic Regression, SVM, KNN, AdaBoost, Decision Tree, Histogram Gradient Boosting, Gaussian Naive Bayes) as well as an LSTM implementation. These components are provided to simplify future experimentation but are not part of the current evaluation presented in this repository.


Random Forest is currently the primary classifier.

---

## Evaluation

The project uses **Leave-One-Subject-Out (LOSO)** cross-validation.

For each iteration:

1. One subject is held out for testing.
2. The remaining subjects are used for training.
3. Macro F1 score is computed.
4. Classification reports are generated for each fold.

This evaluation provides a realistic estimate of how well the model generalizes to unseen individuals.

---

## Project Structure

```
har_utils/
│
├── analysis.py          # Model evaluation and LOSO experiments
├── config.py            # Project configuration
├── data.py              # Data loading utilities
├── features.py          # Feature engineering
├── model.py             # Machine learning models
├── preprocessing.py     # Skeleton normalization and preprocessing pipeline
└── visualization.py     # Plots and visualizations
```

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* Optuna
* Matplotlib
* Seaborn

---

## Current Feature Set

* Skeleton normalization
* Joint angle computation
* Range-of-motion features
* Velocity and acceleration features
* Multi-scale temporal windowing
* Multiple classical machine learning models
* LSTM sequence modeling
* Leave-One-Subject-Out (LOSO) cross-validation
* Optuna hyperparameter optimization
* Feature importance analysis
* Confusion matrix visualization

---

## Future Improvements

- Hyperparameter optimization
- XGBoost and LightGBM models
- Feature selection
- Frequency-domain features
- Deep learning approaches (LSTM, TCN, Transformer)
- Real-time inference
- Additional biomechanical features

---

## Author

This project was developed as part of research into feature engineering techniques for Human Activity Recognition using pose estimation.
