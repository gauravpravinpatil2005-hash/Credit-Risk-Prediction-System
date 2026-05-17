# Credit Risk Scoring — GiveMeSomeCredit

A complete machine learning pipeline for predicting credit default risk using the **GiveMeSomeCredit** dataset from Kaggle.

---

## Project Structure

```
credit_risk_project/
├── credit_risk_scoring.ipynb     # Main Jupyter notebook
├── credit_risk_scoring.py        # Standalone Python script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── outputs/
    ├── models/                   # Saved model files (.pkl / .joblib)
    ├── plots/                    # All generated plots (PNG)
    └── reports/                  # Classification reports (TXT)
```

---

## Dataset

Place the following files inside a `data/` folder at the project root:

```
data/
├── cs-training.csv
└── cs-test.csv
```

Download from: https://www.kaggle.com/c/GiveMeSomeCredit/data

**Target column:** `SeriousDlqin2yrs` — binary (0 = good, 1 = defaulted)

---

## Setup & Execution

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3a. Run as Jupyter Notebook

```bash
jupyter notebook credit_risk_scoring.ipynb
```

Run cells **sequentially** from top to bottom.  
> ⚠️ Heavy training cells are marked — run them only when ready.

### 3b. Run as Python Script

```bash
python credit_risk_scoring.py
```

Outputs (plots, models, reports) will be saved to the `outputs/` folder.

---

## Models Implemented

| Category   | Model                |
|------------|----------------------|
| Baseline   | Logistic Regression  |
| Baseline   | Decision Tree        |
| Baseline   | K-Nearest Neighbors  |
| Baseline   | SVM (LinearSVC)      |
| Baseline   | Naive Bayes          |
| Ensemble   | Random Forest        |
| Ensemble   | XGBoost              |
| Ensemble   | AdaBoost             |
| Ensemble   | Gradient Boosting    |
| Ensemble   | Bagging Classifier   |

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Full Classification Report

---

## Notes

- Class imbalance is handled via `class_weight='balanced'` where applicable.
- Best model is automatically saved to `outputs/models/best_model.joblib`.
- All plots are saved to `outputs/plots/` as high-resolution PNGs.
- KNN and SVM use reduced sample sizes for memory efficiency.
