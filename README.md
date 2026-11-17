# Real Estate Price Range Prediction

## Overview
Predict property price categories using structured real estate features.

## Setup
Install dependencies:
    python -m pip install -r requirements.txt

## Data Preprocessing
Run:
    python Data.py

Performs data exploration and an 80/20 train-test split.

## Training
Run:
    python Train.py

Includes:
- Numerical and categorical preprocessing
- Logistic Regression with GridSearchCV
- Held-out evaluation using accuracy and macro F1
- Classification report and confusion matrix
- Permutation feature importance

Saves the fitted pipeline to:
    artifacts/model.joblib

## Inference
Run:
    python predict.py

The prediction function supports dictionaries and DataFrams and tests individual and batch

## NLP Explanation
 Descriptions could be cleaned, converted to TF-IDF features, and combined with structured features using ColumnTransformer.

 The combined model would be evaluated against the structured-data baseline.

 ## Limitations

The baseline excludes descriptions.
Predictions are price categories, not exact prices.