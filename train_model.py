import os
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

def train_and_evaluate(csv_path="solar_dataset.csv"):
    if not os.path.exists(csv_path):
        sys.path.insert(0, os.path.dirname(__file__))
        from dataset_generator import generate_solar_dataset
        generate_solar_dataset(csv_path, 6000)

    print(f"[LOAD] Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    feature_cols = [
        "hour",
        "month",
        "temp",
        "irradiance",
        "cloudCoverage",
        "humidity",
        "windSpeed",
        "capacityKW"
    ]
    target_col = "solar"

    X = df[feature_cols]
    y = df[target_col]

    # Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[TRAIN] Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

    # 1. Linear Regression Pipeline
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    lr_mae = float(mean_absolute_error(y_test, y_pred_lr))
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    lr_r2 = float(r2_score(y_test, y_pred_lr))

    # 2. Random Forest Regressor Pipeline
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    rf_r2 = float(r2_score(y_test, y_pred_rf))

    print("\n=================== MODEL EVALUATION ===================")
    print(f"Linear Regression      -> MAE: {lr_mae:.4f} kW | RMSE: {lr_rmse:.4f} kW | R2: {lr_r2:.4f}")
    print(f"Random Forest (Winner) -> MAE: {rf_mae:.4f} kW | RMSE: {rf_rmse:.4f} kW | R2: {rf_r2:.4f}")
    print("========================================================\n")

    # Feature Importances from Random Forest
    rf_model = rf_pipeline.named_steps["regressor"]
    importances = rf_model.feature_importances_
    feature_importance_dict = {col: round(float(imp), 4) for col, imp in zip(feature_cols, importances)}

    # Save winning model (Random Forest)
    winning_model = rf_pipeline
    model_save_path = "model_pipeline.joblib"
    joblib.dump(winning_model, model_save_path)
    print(f"[SAVE] Winning model saved to '{model_save_path}'")

    # Save metrics JSON for API & Frontend Hackathon Judges Showcase
    metrics_data = {
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": feature_cols,
            "target": target_col
        },
        "models": {
            "linear_regression": {
                "name": "Linear Regression",
                "mae": round(lr_mae, 4),
                "rmse": round(lr_rmse, 4),
                "r2_score": round(lr_r2, 4)
            },
            "random_forest": {
                "name": "Random Forest Regressor (100 trees)",
                "mae": round(rf_mae, 4),
                "rmse": round(rf_rmse, 4),
                "r2_score": round(rf_r2, 4),
                "selected": True
            }
        },
        "feature_importances": feature_importance_dict,
        "best_model": "Random Forest Regressor"
    }

    metrics_save_path = "metrics.json"
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"[SAVE] Model evaluation metrics saved to '{metrics_save_path}'")

if __name__ == "__main__":
    train_and_evaluate()
