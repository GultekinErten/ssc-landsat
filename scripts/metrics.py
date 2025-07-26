"""
Model Performance Evaluator for SSC Estimation

This script trains and evaluates multiple machine learning models (Linear Regression, Random Forest, LightGBM, XGBoost, CatBoost)
on Suspended Sediment Concentration (SSC) data across various thresholds (e.g., 500, 1000, 4000 mg/L).
It computes standard regression metrics including R², RMSE, MAE, CVRMSE, KGE, NSE, and others.

Usage:
-------
python metrics.py \
    --data path/to/input.csv \
    --output path/to/results.csv \
    --thresholds 500 1000 4000 \
    [--test_size 0.2]

Notes:
- Input CSV must contain SSC and surface reflectance features.
- Automatically computes log(SSC) and back-transformed metrics.
- Output CSV includes both training and testing results for all models and targets.

Author: Gültekin Erten  
License: MIT
"""

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


def evaluate(y_true, y_pred):
    """
    Calculate various regression evaluation metrics.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cvrmse = rmse / np.mean(y_true)
    mbe = np.mean(y_pred - y_true)
    kge = 1 - np.sqrt((r2 - 1)**2 + (rmse / np.std(y_true) - 1)**2 +
                      ((np.mean(y_pred) / np.mean(y_true)) - 1)**2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    rrmse = rmse / np.std(y_true)
    nse = 1 - mean_squared_error(y_true, y_pred) / np.var(y_true)
    return [r2, rmse, mae, cvrmse, mbe, kge, mape, nrmse, rrmse, nse]


def get_models():
    """
    Define a dictionary of regression models.
    """
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=20,
                                              min_samples_leaf=2, max_features='sqrt', random_state=42),
        "LightGBM": lgb.LGBMRegressor(learning_rate=0.05, n_estimators=200, max_depth=10,
                                      num_leaves=50, subsample=0.8, colsample_bytree=0.8, random_state=42),
        "XGBoost": xgb.XGBRegressor(learning_rate=0.03, n_estimators=300, max_depth=5,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42),
        "CatBoost": CatBoostRegressor(learning_rate=0.1, depth=7, iterations=600,
                                       l2_leaf_reg=8, subsample=0.8, colsample_bylevel=0.3,
                                       verbose=0, random_seed=42)
    }


def run_training(data_path, output_csv, test_size, thresholds):
    """
    Run training and evaluation across different thresholds and models.
    """
    results_all = []
    models = get_models()

    for limit in tqdm(thresholds, desc="Processing thresholds"):
        df = pd.read_csv(data_path)
        df = df[df['ssc'] < limit]
        df['B3_B2'] = df['SR_B3'] / df['SR_B2']
        df = df.dropna(subset=['SR_B2', 'ssc', 'SR_B3', 'SR_B4', 'filled', 'lon', 'lat', 'month'])
        df['ln_ssc'] = np.log(df['ssc'])

        features = ['B3_B2', 'filled', 'lon', 'SR_B3', 'SR_B4', 'lat', 'month']
        X = df[features]
        y_direct = df['ssc']
        y_ln = df['ln_ssc']

        X_train, X_test, y_train_d, y_test_d = train_test_split(X, y_direct, test_size=test_size, random_state=42)
        _, _, y_train_ln, y_test_ln = train_test_split(X, y_ln, test_size=test_size, random_state=42)

        for name, model in models.items():
            # Direct SSC
            model.fit(X_train, y_train_d)
            pred_train_d = model.predict(X_train)
            pred_test_d = model.predict(X_test)
            m_train_d = evaluate(y_train_d, pred_train_d)
            m_test_d = evaluate(y_test_d, pred_test_d)
            results_all.append([limit, name, "Direct SSC", "Train"] + m_train_d)
            results_all.append([limit, name, "Direct SSC", "Test"] + m_test_d)

            # ln(SSC)
            model.fit(X_train, y_train_ln)
            pred_train_ln = model.predict(X_train)
            pred_test_ln = model.predict(X_test)
            m_train_ln = evaluate(y_train_ln, pred_train_ln)
            m_test_ln = evaluate(y_test_ln, pred_test_ln)
            results_all.append([limit, name, "ln(SSC)", "Train"] + m_train_ln)
            results_all.append([limit, name, "ln(SSC)", "Test"] + m_test_ln)

            # ln(SSC) → SSC (back-transform)
            pred_train_back = np.exp(pred_train_ln)
            pred_test_back = np.exp(pred_test_ln)
            m_train_back = evaluate(y_train_d, pred_train_back)
            m_test_back = evaluate(y_test_d, pred_test_back)
            results_all.append([limit, name, "ln(SSC) → SSC", "Train"] + m_train_ln)  # R² matches ln(SSC)
            results_all.append([limit, name, "ln(SSC) → SSC", "Test"] + m_test_back)

    # Save results
    columns = ["Threshold", "Model", "Strategy", "Dataset",
               "R2", "RMSE", "MAE", "CVRMSE", "MBE", "KGE", "MAPE", "NRMSE", "RRMSE", "NSE"]
    results_df = pd.DataFrame(results_all, columns=columns)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple ML models on SSC data with various thresholds.")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save results CSV.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio. Default is 0.2")
    parser.add_argument("--thresholds", type=int, nargs='+', default=[8000, 4000, 1000, 500],
                        help="List of SSC thresholds (mg/L). Default: 8000 4000 1000 500")

    args = parser.parse_args()

    run_training(
        data_path=args.data,
        output_csv=args.output,
        test_size=args.test_size,
        thresholds=args.thresholds
    )
