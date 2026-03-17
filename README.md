# Startup Success Prediction using Machine Learning

## Project Overview
This project predicts whether a startup is likely to succeed based on funding, investor participation, startup milestones, and company metadata.

The goal is to build an end-to-end machine learning workflow that can help investors, accelerators, and startup analysts identify high-potential startups.

## Problem Statement
Investors review a large number of startups and need a way to prioritize which companies deserve further due diligence. This project uses machine learning to estimate the probability of startup success.

## Dataset
The dataset contains startup-level information such as:
- funding rounds
- total funding amount
- investor participation
- startup category
- milestones
- geographic indicators
- venture capital and angel funding history

Target variable:
- `1` = successful startup
- `0` = failed startup

## Workflow
1. Data loading
2. Data cleaning
3. Leakage removal
4. Missing value imputation
5. Feature engineering
6. Train/test split
7. Baseline model comparison
8. Hyperparameter tuning
9. Feature importance analysis
10. ROC curve and confusion matrix
11. Final model saving
12. Startup success prediction
13. Streamlit app deployment

## Feature Engineering
The following engineered features were added:
- `funding_per_round`
- `participants_per_round`
- `milestones_per_round`
- `network_funding_strength`
- `rounds_x_top500`

These features improved the model by capturing investment and startup growth dynamics more effectively.

## Models Used
- Logistic Regression
- Random Forest
- XGBoost

## Final Model
The final saved model is a tuned Random Forest classifier.

## Evaluation Metrics
The project uses:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Key Insights
Important predictors of startup success include:
- relationships
- funding_total_usd
- avg_participants
- milestones
- funding_rounds
- is_top500

## Files
- `train.py` → baseline model comparison
- `tune_model.py` → hyperparameter tuning
- `feature_importance.py` → feature importance plot
- `model_evaluation.py` → ROC curve and confusion matrix
- `final_model.py` → final model training and saving
- `predict.py` → sample prediction
- `app.py` → Streamlit web app
- `preprocess.py` → common preprocessing pipeline

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
