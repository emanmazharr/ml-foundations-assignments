# Week 1 — Adult Income Classification Final Project

## 1. Objective
Predict whether an individual earns more than $50K/year using the UCI Adult/Census Income dataset. The project follows an end-to-end supervised-learning workflow covering preprocessing, feature engineering, model comparison, hyperparameter tuning, calibration, threshold selection, final validation, interpretation, inference, and documentation.

## 2. Target Variable
- `1` = income >50K/year
- `0` = income <=50K/year

## 3. Validation Strategy
The dataset was divided using stratified splits:
- 70% training
- 10% development
- 20% untouched test

`random_state = 42` was used for reproducibility. The test set was kept untouched until final evaluation.

## 4. Feature Engineering
The project added:
- `age_bucket`
- `hours_bin`
- `has_capital_gain`
- `log_capital_gain`
- `higher_ed`
- `edu_hours_interaction`
- `is_married`

Feature transformations are part of the model workflow and are not manually repeated during inference.

## 5. Preprocessing
- Numeric features: median imputation + standardization.
- Categorical features: most-frequent imputation + one-hot encoding with `handle_unknown="ignore"`.
- Boolean features: most-frequent imputation.

Preprocessing is kept inside the sklearn pipeline to reduce leakage risk and ensure consistent training/inference behavior.

## 6. Models Evaluated
The Week 1 workflow evaluated:
- Logistic Regression
- Random Forest
- HistGradientBoosting

Day 3 cross-validation identified HistGradientBoosting as the strongest shortlisted model on the main predictive metrics.

## 7. Hyperparameter Tuning
Day 4 used `RandomizedSearchCV` with 5-fold `StratifiedKFold`, precision as the primary scoring metric, `random_state=42`, and `n_jobs=-1`.

### Logistic Regression
Best parameters:
- `classifier__penalty = 'l1'`
- `classifier__C = 0.029763514416313176`

Best CV precision: **0.7536**

The search space contained 40 possible parameter combinations, so scikit-learn evaluated all 40 rather than 50 requested iterations.

### HistGradientBoosting
Best parameters:
- `classifier__max_leaf_nodes = 31`
- `classifier__max_iter = 100`
- `classifier__max_depth = 3`
- `classifier__learning_rate = 0.02`
- `classifier__l2_regularization = 5.0`

Best CV precision: **0.8026**

## 8. Final Model
The final production workflow uses the tuned HistGradientBoosting pipeline with sigmoid probability calibration using `CalibratedClassifierCV`.

## 9. Classification Threshold
The operating threshold was selected on the development set by maximizing precision while maintaining recall >= 0.50.

**Selected threshold: 0.62**

## 10. Final Test Performance
The final calibrated HistGradientBoosting model was evaluated on the untouched test set.

| Metric | Final Result |
|---|---:|
| Accuracy | **0.8535 (85.35%)** |
| Precision | **0.8042 (80.42%)** |
| Recall | **0.5128 (51.28%)** |
| F1-score | **0.6263 (62.63%)** |
| ROC-AUC | **0.9082 (90.82%)** |
| PR-AUC | **0.7778 (77.78%)** |
| Brier score | **0.1007** |

Classification report on the test set:
- `<=50K`: precision **0.86**, recall **0.96**, F1 **0.91**, support **7431**
- `>50K`: precision **0.80**, recall **0.51**, F1 **0.63**, support **2338**
- Overall accuracy: **0.85** on **9769** test examples.

The selected threshold prioritizes precision while satisfying the minimum recall requirement, resulting in stronger precision than recall for the >50K class.

## 11. Model Comparison
The available Day 4 tuning results were:
| Model | Best CV Precision | Fit Time (sec) |
|---|---:|---:|
| Logistic Regression | 0.753630 | 646.79 |
| HistGradientBoosting | 0.802644 | 1030.17 |

HistGradientBoosting had the higher tuned CV precision and was selected for the final workflow.

## 12. Error Analysis
The final notebook generates a confusion matrix, false-positive/false-negative analysis, classification report, and subgroup summaries for sex, race, education, and marital status.

A false positive is a person predicted as >50K when the actual class is <=50K. A false negative is a person who actually earns >50K but is predicted as <=50K. With the chosen threshold, the model favors precision, so false negatives/missed >50K cases remain an important limitation.

## 13. Feature / Model Interpretation
The final notebook uses permutation importance on the raw input variables and produces a top-feature visualization. Important predictive signals identified during model development include:
- education-hours interaction
- log-transformed capital gain
- married-status information
- age-related information

These are predictive associations rather than causal explanations. The executed notebook's permutation-importance output should be used for the exact final top-10 ranking.

## 14. Production Inference
The saved artifact is:

`final_model.joblib`

Inference accepts a DataFrame containing the original predictor columns. The saved workflow automatically performs feature engineering, missing-value handling, scaling, encoding, probability generation, and thresholding.

The notebook tests the workflow on 10 unseen examples.

## 15. Limitations and Future Improvements
- The Adult dataset is historical and may not represent current populations.
- External validation on newer data is recommended.
- Subgroup/fairness analysis should be expanded before real-world deployment.
- Thresholds should be aligned with actual business costs.
- Monitoring for distribution drift and periodic retraining would improve production reliability.
- Additional models and feature engineering could be evaluated with more data and compute.

## 16. Reproduction
1. Install packages listed in `requirements.txt`.
2. Place `adult_dataset.csv` beside the notebook.
3. Open `Week1_Day5_Final_Validation.ipynb`.
4. Run all cells from top to bottom.
5. The notebook performs the required tuning/validation and creates the final artifact and evaluation outputs.

## 17. Inference Example
```python
import joblib

artifact = joblib.load("final_model.joblib")
model = artifact["model"]
threshold = artifact["threshold"]

probability = model.predict_proba(new_data)[:, 1]
prediction = (probability >= threshold).astype(int)
```

No manual preprocessing should be performed before calling the saved model.

## 18. Environment
The executed notebook records the Python, pandas, NumPy, SciPy and scikit-learn versions used for the run. Keep `requirements.txt` with the notebook and final artifact.
