#!/usr/bin/env python3
"""
Retrain models and save artifacts:
  • XGB regressor (log target) → models/xgb_regressor.joblib
  • XGB classifier (SMOTE-balanced) → models/xgb_classifier.joblib
  • RandomForest regressor & classifier → models/rf_*.joblib
  • Keras MLP regressor → models/mlp_regressor.h5
Metrics → models/metrics_new.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump
from tqdm import tqdm
import sklearn

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from xgboost import XGBRegressor, XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from packaging import version

# reproducibility
RND = 42
np.random.seed(RND)
tf.random.set_seed(RND)

# feature lists
NUM_COLS = [
    "lat", "lon", "temp_c", "wind_speed", "precip_flag",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
CAT_COLS = ["location_name", "day_of_week", "weather_summary"]
TARGET_REG = "total_traffic_volume"
TARGET_CLF = "congestion_level"

# Helper functions

def cyclicalEncode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2 * np.pi * series / period
    return pd.DataFrame({f"{prefix}_sin": np.sin(radians), f"{prefix}_cos": np.cos(radians)})


def loadDataset(csv_path: Path) -> pd.DataFrame:
    print(f"[{datetime.now():%H:%M:%S}] Loading {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], low_memory=False)
    df = df.assign(
        **cyclicalEncode(df["hour"], 24, "hour"),
        **cyclicalEncode(df["month"], 12, "month"),
    )
    return df.dropna(subset=[TARGET_REG, TARGET_CLF])


def buildPreprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
    ])
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    else:
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    return ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ], remainder="drop", n_jobs=-1)


def splitTrainTest(df: pd.DataFrame):
    return train_test_split(df, test_size=0.2, random_state=RND, stratify=df[TARGET_CLF])

# Main training

def trainAll(data_csv: Path, model_dir: Path):
    df = loadDataset(data_csv)
    model_dir.mkdir(exist_ok=True)
    metrics = {}

    # encode classifier target
    le = LabelEncoder()
    df["y_clf_enc"] = le.fit_transform(df[TARGET_CLF])

    train_df, test_df = splitTrainTest(df)
    X_train = train_df[NUM_COLS + CAT_COLS]
    X_test = test_df[NUM_COLS + CAT_COLS]
    y_reg_train, y_reg_test = train_df[TARGET_REG], test_df[TARGET_REG]
    y_clf_train, y_clf_test = train_df["y_clf_enc"], test_df["y_clf_enc"]

    pre = buildPreprocessor()

    # 1. XGB Regressor
    print(f"[{datetime.now():%H:%M:%S}] Training XGB Regressor")
    xgb_base = XGBRegressor(objective="reg:squarederror", random_state=RND, tree_method="hist")
    regressor = TransformedTargetRegressor(regressor=xgb_base, func=np.log1p, inverse_func=np.expm1)
    pipe_reg = Pipeline([("pre", pre), ("model", regressor)])
    param_reg = {
        "model__regressor__max_depth": [4, 6, 8],
        "model__regressor__n_estimators": [300, 500, 800],
        "model__regressor__learning_rate": [0.03, 0.07, 0.15],
        "model__regressor__subsample": [0.7, 0.9],
    }
    gs_reg = RandomizedSearchCV(pipe_reg, param_distributions=param_reg, n_iter=15,
                                cv=3, scoring="neg_mean_absolute_error",
                                random_state=RND, verbose=1, n_jobs=-1)
    gs_reg.fit(X_train, y_reg_train)
    best_reg = gs_reg.best_estimator_
    pred_reg = best_reg.predict(X_test)
    metrics["xgb_reg_mae"] = mean_absolute_error(y_reg_test, pred_reg)
    metrics["xgb_reg_rmse"] = np.sqrt(mean_squared_error(y_reg_test, pred_reg))
    dump(best_reg, model_dir / "xgb_regressor.joblib")

    # 2. XGB Classifier
    print(f"[{datetime.now():%H:%M:%S}] Training XGB Classifier")
    clf_pipe = ImbPipeline([
        ("pre", pre),
        ("smote", SMOTE(random_state=RND)),
        ("model", XGBClassifier(objective="multi:softprob", num_class=len(le.classes_),
                                 eval_metric="mlogloss", random_state=RND, tree_method="hist")),
    ])
    param_clf = {
        "model__max_depth": [4, 6, 8],
        "model__n_estimators": [300, 500, 800],
        "model__learning_rate": [0.03, 0.07, 0.15],
        "model__subsample": [0.7, 0.9],
    }
    gs_clf = RandomizedSearchCV(clf_pipe, param_distributions=param_clf, n_iter=12,
                                cv=3, scoring="accuracy",
                                random_state=RND, verbose=1, n_jobs=1)
    gs_clf.fit(X_train, y_clf_train)
    best_clf = gs_clf.best_estimator_
    pred_clf = best_clf.predict(X_test)
    metrics["xgb_clf_acc"] = accuracy_score(y_clf_test, pred_clf)
    metrics["xgb_clf_f1"] = f1_score(y_clf_test, pred_clf, average="weighted")
    dump(best_clf, model_dir / "xgb_classifier.joblib")

    # 3. RandomForest
    print(f"[{datetime.now():%H:%M:%S}] Training RandomForest")
    rf_reg = Pipeline([("pre", pre),
                       ("model", RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                                        random_state=RND, n_jobs=-1))])
    rf_reg.fit(X_train, y_reg_train)
    pred_rf_reg = rf_reg.predict(X_test)
    metrics["rf_reg_mae"] = mean_absolute_error(y_reg_test, pred_rf_reg)
    metrics["rf_reg_rmse"] = np.sqrt(mean_squared_error(y_reg_test, pred_rf_reg))
    dump(rf_reg, model_dir / "rf_regressor.joblib")

    rf_clf = Pipeline([("pre", pre),
                       ("model", RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
                                                         class_weight="balanced_subsample",
                                                         random_state=RND, n_jobs=-1))])
    rf_clf.fit(X_train, y_clf_train)
    pred_rf_clf = rf_clf.predict(X_test)
    metrics["rf_clf_acc"] = accuracy_score(y_clf_test, pred_rf_clf)
    metrics["rf_clf_f1"] = f1_score(y_clf_test, pred_rf_clf, average="weighted")
    dump(rf_clf, model_dir / "rf_classifier.joblib")

    # 4. Keras MLP Regressor (numeric only)
    print(f"[{datetime.now():%H:%M:%S}] Training MLP Regressor")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[NUM_COLS])
    X_test_num = scaler.transform(test_df[NUM_COLS])

    mlp = models.Sequential([
        layers.Input(shape=(X_train_num.shape[1],)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])
    mlp.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    mlp.fit(X_train_num, y_reg_train, epochs=60, batch_size=512,
            validation_split=0.1, verbose=1,
            callbacks=[callbacks.EarlyStopping(patience=6, restore_best_weights=True),
                       callbacks.ReduceLROnPlateau(patience=3, factor=0.5)])
    pred_mlp = mlp.predict(X_test_num, verbose=0).flatten()
    metrics["mlp_reg_mae"] = mean_absolute_error(y_reg_test, pred_mlp)
    metrics["mlp_reg_rmse"] = np.sqrt(mean_squared_error(y_reg_test, pred_mlp))
    mlp.save(model_dir / "mlp_regressor.h5")
    dump(scaler, model_dir / "mlp_num_scaler.joblib")

    # Save metrics
    metrics_path = model_dir / "metrics_new.json"
    metrics_path.write_text(json.dumps(metrics, indent=4))
    print(f"Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=4))

# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="combined_final_dataset.csv")
    parser.add_argument("--modeldir", default="models")
    args = parser.parse_args()
    trainAll(Path(args.data), Path(args.modeldir))

if __name__ == "__main__":
    main()
