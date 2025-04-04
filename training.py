import pandas as pd
import numpy as np
import math
import json
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
from sklearn.compose import TransformedTargetRegressor
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from joblib import dump, load

# -------------------- Part 1: Training and Tuning --------------------

print("========== Loading dataset ==========")
df = pd.read_csv("final_congestion_dataset.csv")
print("Dataset loaded. Shape:", df.shape)

print("\n========== Defining Features and Targets ==========")
X = df[["lat", "lon", "temp_c", "wind_speed", "hour", "day_of_week", "month", "weather_summary", "location_name"]].copy()
y_reg = df["total_traffic_volume"]
y_clf = df["congestion_level"]

print("\nUnique congestion_level classes before encoding:")
print(y_clf.unique())

print("\n========== Encoding Classification Target ==========")
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)
print("Encoded classes:", le.classes_)

print("\n========== Splitting Data into Train, Eval, and Test Sets ==========")
# 70% train, 15% eval, 15% test stratified by classification target
train_idx, temp_idx = train_test_split(df.index, test_size=0.30, random_state=42, stratify=y_clf_encoded)
eval_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y_clf_encoded[temp_idx])
X_train = X.loc[train_idx]
X_eval  = X.loc[eval_idx]
X_test  = X.loc[test_idx]
y_reg_train = y_reg.loc[train_idx]
y_reg_eval  = y_reg.loc[eval_idx]
y_reg_test  = y_reg.loc[test_idx]
y_clf_train = y_clf_encoded[train_idx]
y_clf_eval  = y_clf_encoded[eval_idx]
y_clf_test  = y_clf_encoded[test_idx]
print("Train set size:", len(train_idx))
print("Evaluation set size:", len(eval_idx))
print("Test set size:", len(test_idx))

print("\n========== Checking Skewness of Regression Target ==========")
skewness = y_reg_train.skew()
print("Skewness of traffic volume (train set):", skewness)

print("\n========== Setting Up TransformedTargetRegressor for Regression ==========")
base_regressor = XGBRegressor(random_state=42, objective="reg:squarederror")
regressor = TransformedTargetRegressor(regressor=base_regressor, func=np.log1p, inverse_func=np.expm1)

print("\n========== Creating Preprocessor ==========")
numerical_features = ["lat", "lon", "temp_c", "wind_speed", "hour", "month"]
categorical_features = ["day_of_week", "weather_summary", "location_name"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

print("\n========== Building Regression Pipeline ==========")
reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", regressor)
])

print("\n========== Building Classification Pipeline with SMOTE ==========")
clf_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", XGBClassifier(random_state=42, eval_metric="mlogloss"))
])

print("\n========== Setting Up Hyperparameter Grids for 3-Fold CV ==========")
param_grid_reg = {
    "regressor__regressor__max_depth": [3, 5, 7],
    "regressor__regressor__n_estimators": [50, 100, 150],
    "regressor__regressor__learning_rate": [0.01, 0.1, 0.2]
}
param_grid_clf = {
    "classifier__max_depth": [3, 5, 7],
    "classifier__n_estimators": [50, 100, 150],
    "classifier__learning_rate": [0.01, 0.1, 0.2]
}

print("\n========== Running GridSearchCV for Regression ==========")
grid_reg = GridSearchCV(reg_pipeline, param_grid_reg, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=2)
start_time = time.time()
grid_reg.fit(X_train, y_reg_train)
print("GridSearchCV for regression completed in {:.2f} seconds.".format(time.time() - start_time))
print("Best regression parameters:", grid_reg.best_params_)

print("\n========== Running GridSearchCV for Classification ==========")
# To avoid heavy parallelism issues with SMOTE inside CV, you might set n_jobs=1 if needed.
grid_clf = GridSearchCV(clf_pipeline, param_grid_clf, cv=3, scoring="accuracy", n_jobs=1, verbose=2)
start_time = time.time()
grid_clf.fit(X_train, y_clf_train)
print("GridSearchCV for classification completed in {:.2f} seconds.".format(time.time() - start_time))
print("Best classification parameters:", grid_clf.best_params_)

# Collect final metrics
metrics = {}

print("\n========== Evaluating Best Regression Model on Evaluation Set ==========")
y_reg_pred_eval = grid_reg.predict(X_eval)
metrics["reg_mae_eval"] = mean_absolute_error(y_reg_eval, y_reg_pred_eval)
metrics["reg_rmse_eval"] = math.sqrt(mean_squared_error(y_reg_eval, y_reg_pred_eval))
print("Regression Evaluation - MAE: {:.2f}, RMSE: {:.2f}".format(metrics["reg_mae_eval"], metrics["reg_rmse_eval"]))

print("\n========== Evaluating Best Classification Model on Evaluation Set ==========")
y_clf_pred_eval = grid_clf.predict(X_eval)
metrics["clf_acc_eval"] = accuracy_score(y_clf_eval, y_clf_pred_eval)
metrics["clf_f1_eval"] = f1_score(y_clf_eval, y_clf_pred_eval, average="weighted")
print("Classification Evaluation - Accuracy: {:.2f}, F1 Score: {:.2f}".format(metrics["clf_acc_eval"], metrics["clf_f1_eval"]))
print("Classification Report (Evaluation Set):")
print(classification_report(y_clf_eval, y_clf_pred_eval, target_names=le.classes_))

print("\n========== Evaluating Best Regression Model on Test Set ==========")
y_reg_pred_test = grid_reg.predict(X_test)
metrics["reg_mae_test"] = mean_absolute_error(y_reg_test, y_reg_pred_test)
metrics["reg_rmse_test"] = math.sqrt(mean_squared_error(y_reg_test, y_reg_pred_test))
print("Regression Test - MAE: {:.2f}, RMSE: {:.2f}".format(metrics["reg_mae_test"], metrics["reg_rmse_test"]))

print("\n========== Evaluating Best Classification Model on Test Set ==========")
y_clf_pred_test = grid_clf.predict(X_test)
metrics["clf_acc_test"] = accuracy_score(y_clf_test, y_clf_pred_test)
metrics["clf_f1_test"] = f1_score(y_clf_test, y_clf_pred_test, average="weighted")
print("Classification Test - Accuracy: {:.2f}, F1 Score: {:.2f}".format(metrics["clf_acc_test"], metrics["clf_f1_test"]))
print("Classification Report (Test Set):")
print(classification_report(y_clf_test, y_clf_pred_test, target_names=le.classes_))

# Save final metrics to a JSON file for record keeping
print("\n========== Saving Final Metrics ==========")
with open("final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Final metrics saved to 'final_metrics.json'.")

print("\n========== Saving Best Models to Disk ==========")
dump(grid_reg.best_estimator_, "tuned_xgb_regressor.joblib")
dump(grid_clf.best_estimator_, "tuned_xgb_classifier.joblib")
print("Models saved to 'tuned_xgb_regressor.joblib' and 'tuned_xgb_classifier.joblib'.")

# -------------------- Part 2: Model Loading and Prediction Demo --------------------

print("\n========== Loading Saved Models for Prediction Demo ==========")
best_regressor = load("tuned_xgb_regressor.joblib")
best_classifier = load("tuned_xgb_classifier.joblib")
print("Models loaded successfully.")

# Create a sample input (a new, unseen instance)
sample_data = {
    "lat": [43.67],
    "lon": [-79.45],
    "temp_c": [10.0],
    "wind_speed": [2.5],
    "hour": [15],
    "day_of_week": ["Friday"],
    "month": [7],
    "weather_summary": ["Sunny, 10.0°C"],
    "location_name": ["Test Location"]
}
sample_df = pd.DataFrame(sample_data)
print("\nSample input for prediction:")
print(sample_df)

# Predict using the regression model (traffic volume)
sample_reg_pred = best_regressor.predict(sample_df)
print("\nRegression model prediction (traffic volume): {:.2f}".format(sample_reg_pred[0]))

# Predict using the classification model (congestion level)
sample_clf_pred = best_classifier.predict(sample_df)
# Convert prediction back to original label
sample_clf_label = le.inverse_transform(sample_clf_pred)
print("Classification model prediction (congestion level):", sample_clf_label[0])
