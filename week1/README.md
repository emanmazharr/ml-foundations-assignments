# Adult Income Prediction – Supervised Learning

## 📌 Project Overview
This project uses supervised machine learning to predict whether a person's annual income is greater than $50K using the UCI Adult Income dataset. The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model training, hyperparameter tuning, calibration, evaluation, error analysis, and inference.

## 🎯 Objective
Predict whether an individual's income is:
- <=50K
- >50K

## 📊 Dataset
The project uses the Adult Income dataset (`adult_dataset.csv`), containing demographic, employment, education, and financial-related features.

The dataset is split into:
- 70% Training
- 10% Development/Validation
- 20% Untouched Test Set

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib/Cloudpickle
- Jupyter Notebook

## 🔧 Feature Engineering
Additional features were created to improve model performance, including:
- Age buckets
- Working-hours categories
- Capital-gain indicators
- Log-transformed capital gain
- Higher-education indicator
- Education × working-hours interaction
- Married-status indicator

## 🤖 Models
The following models were evaluated:
1. Logistic Regression
2. Random Forest
3. HistGradientBoostingClassifier

HistGradientBoosting performed best and was selected for further tuning.

## 🔍 Model Tuning
Hyperparameter tuning was performed using cross-validation. The best HistGradientBoosting configuration achieved a CV precision of **0.8026**.

The final model was also calibrated using sigmoid calibration, and a decision threshold of **0.62** was selected.

## 📈 Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 85.35% |
| Precision | 80.42% |
| Recall | 51.28% |
| F1 Score | 62.63% |
| ROC-AUC | 90.82% |
| PR-AUC | 77.78% |
| Brier Score | 0.1007 |

These results were obtained on the untouched 20% test set.

## 📊 Analysis & Visualizations
The project includes:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Calibration Curve
- Feature Importance
- Error and subgroup analysis

## 💡 Key Findings
The final model provides strong overall discrimination and precision. It performs particularly well for identifying the <=50K income class, while recall for the >50K class is comparatively lower. Feature analysis was used to understand the factors contributing to predictions.

## 🚀 How to Run

1. Clone this repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
