# Employee Attrition Prediction with Sentiment Analysis

## Overview
This project predicts employee attrition using HR data enhanced with sentiment analysis from feedback text. Models are trained on 10,000 records, handling class imbalance with SMOTE. Sentiment is computed using VADER for new predictions.

## Features
- Data preprocessing and feature engineering.
- Model training (Logistic Regression, Random Forest, XGBoost).
- Streamlit dashboard for visualizations and predictions.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python src/model_training.py`
3. Run app: `streamlit run app.py`

## Data
- Place the CSV in `data/`.

## Notes
- Attrition is imbalanced (~20% positive).
- Sentiment scaling: VADER compound mapped to 1-4.25.
- Assumptions: Department/Role labels are illustrative (adjust if needed).

For questions, refer to the safety instructions (but this is safe!).