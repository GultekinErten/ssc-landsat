"""
CatBoost Model Trainer for SSC and ln(SSC)

This script trains two separate CatBoost regression models to predict Suspended Sediment Concentration (SSC) using
Landsat-derived features. It supports both raw SSC and log-transformed SSC (ln(SSC)) as targets, and exports trained
models as .pkl files for further use or deployment.

Usage:
-------
python train_catboost.py \
    --input_csv path/to/input.csv \
    --model_dir path/to/save/models \
    --log_dir path/to/save/catboost/logs

Notes:
- Input CSV must contain surface reflectance and terrain features.
- Output directory will contain two .pkl files (direct SSC and ln(SSC)).
- Default parameters can be customized in the script.

Author: Gültekin Erten  
License: MIT
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from tqdm import tqdm
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['ssc'] < 1000].copy()
    df['B3_B2'] = df['SR_B3'] / df['SR_B2']
    df['ln_ssc'] = np.log(df['ssc'])
    features = ['B3_B2', 'filled', 'lon', 'SR_B3', 'SR_B4', 'lat', 'month']
    X = df[features].apply(pd.to_numeric, errors='coerce')
    y_direct = df['ssc']
    y_ln = df['ln_ssc']
    combined = pd.concat([X, y_direct, y_ln], axis=1).dropna()
    return combined[features], combined['ssc'], combined['ln_ssc']

def train_and_save_model(X, y, params, output_path, model_name):
    model = CatBoostRegressor(**params)
    with tqdm(total=1, desc=f"Training {model_name}", unit="model") as pbar:
        model.fit(X, y)
        pbar.update(1)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

def main(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("📦 Loading data...")
    X, y_direct, y_ln = load_data(args.input_csv)
    X_train, _, y_train_d, _ = train_test_split(X, y_direct, test_size=0.2, random_state=42)
    _, _, y_train_ln, _ = train_test_split(X, y_ln, test_size=0.2, random_state=42)

    print("🧠 Training models...")

    # Direct SSC model
    params_direct = {
        'learning_rate': 0.1,
        'depth': 7,
        'iterations': 600,
        'l2_leaf_reg': 8,
        'subsample': 0.8,
        'colsample_bylevel': 0.3,
        'random_seed': 42,
        'logging_level': 'Silent',
        'task_type': 'CPU',
        'train_dir': args.log_dir
    }

    direct_model_path = os.path.join(args.model_dir, "catboost_direct_ssc.pkl")
    train_and_save_model(X_train, y_train_d, params_direct, direct_model_path, "Direct SSC")

    # ln(SSC) model
    params_ln = {
        'learning_rate': 0.1,
        'depth': 8,
        'iterations': 1000,
        'l2_leaf_reg': 7,
        'subsample': 0.9,
        'colsample_bylevel': 0.4,
        'random_seed': 42,
        'logging_level': 'Silent',
        'task_type': 'CPU',
        'train_dir': args.log_dir
    }

    ln_model_path = os.path.join(args.model_dir, "catboost_ln_ssc.pkl")
    train_and_save_model(X_train, y_train_ln, params_ln, ln_model_path, "ln(SSC)")

    print("✅ Models trained and saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CatBoost models for SSC and ln(SSC).")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save .pkl models.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory for CatBoost training logs.")
    args = parser.parse_args()
    main(args)
