# Sales Quote Win/Loss Prediction Model

This repository contains a hybrid ensemble model to predict whether a sales quote will be Won or Lost. The model is intended for integration into CRM systems to support sales teams in forecasting deal outcomes.

## Files

- `sales_model_hybrid.py` – Main script for training and evaluating the model.
- `sales_quote_data.csv` – Input dataset with features and target labels.
- `requirements.txt` – Python dependencies.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/bibekbastola946/sales-quote-prediction.git
cd sales-quote-prediction
```

### 2. Set Up the Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare the Data

Ensure `sales_quote_data.csv` is in the project root. The CSV should contain:
- Input features
- A `InternalResult` column where:
  - `1` = Quote Won
  - `0` = Quote Lost

### 4. Run the Model

Execute the model pipeline:

```bash
python sales_model_hybrid.py
```

The script will:
- Load and preprocess the dataset
- Handle class imbalance with SMOTE
- Train base models (XGBoost, LightGBM, CatBoost, Neural Network)
- Perform Stratified K-Fold stacking
- Output evaluation metrics and performance plots

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- ROC AUC
- Confusion Matrix
- Precision-Recall Curve

## Future Work

- AutoML or Optuna-based hyperparameter tuning
- Feature importance analysis (e.g., SHAP)
- API wrapper for real-time integration with CRM
- Web interface for quote scoring
