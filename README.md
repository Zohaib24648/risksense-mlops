# RiskSense: loan-default prediction pipeline

My final-year project (BSc Computer Science, IBA Karachi, 2025): a ZenML training pipeline with
MLflow experiment tracking that predicts loan default for a distributor-financing portfolio of
39,503 records.

- **Data preparation** (`model/data_cleaning.py`, `steps/clean_data.py`): derives default
  labels and distributor-level repayment features, fills missing KIBOR tenures with KNN
  imputation, and handles class imbalance with class weighting.
- **Models** (`model/model_dev.py`, `steps/model_train.py`): five interchangeable classifiers,
  XGBoost, LightGBM, CatBoost, Random Forest and logistic regression, chosen per run in
  `config.yaml`.
- **Evaluation** (`steps/evaluation.py`): accuracy, F1, ROC AUC and confusion-matrix metrics,
  logged to MLflow for every run.
- **Pipelines**: `run_pipeline.py` (training) and `run_deployment.py` (deployment).
  `streamlit_app.py` is still ZenML's original customer-satisfaction demo, not adapted.

The dataset is confidential and is not in this repository, nor anywhere in its history.
`steps/ingest_data.py` expects it at `data/FYP.xlsx`.

Stack: Python, ZenML, MLflow, scikit-learn, XGBoost, LightGBM, CatBoost, pandas.
