import pandas as pd
from joblib import load

# Load saved models
best_regressor = load("tuned_xgb_regressor.joblib")
best_classifier = load("tuned_xgb_classifier.joblib")

# Create a sample input DataFrame with 15 rows
sample_data = {
    "lat": [43.67, 43.65, 43.68, 43.70, 43.66, 43.64, 43.69, 43.67, 43.66, 43.65, 43.70, 43.68, 43.67, 43.66, 43.64],
    "lon": [-79.45, -79.40, -79.43, -79.47, -79.44, -79.42, -79.46, -79.45, -79.41, -79.43, -79.48, -79.46, -79.44, -79.42, -79.40],
    "temp_c": [10.0, 15.0, 8.0, 12.0, 9.5, 11.0, 14.0, 10.0, 13.0, 7.0, 10.5, 9.0, 12.5, 11.5, 8.5],
    "wind_speed": [2.5, 3.0, 2.0, 4.0, 3.5, 2.5, 3.0, 2.0, 4.0, 3.0, 2.5, 3.5, 2.5, 3.0, 2.0],
    "hour": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"],
    "month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3],
    "weather_summary": [
        "Sunny, 10.0°C", "Rainy, 15.0°C", "Snowy, 8.0°C", "Sunny, 12.0°C", "Rainy, 9.5°C",
        "Sunny, 11.0°C", "Rainy, 14.0°C", "Snowy, 10.0°C", "Sunny, 13.0°C", "Rainy, 7.0°C",
        "Sunny, 10.5°C", "Rainy, 9.0°C", "Snowy, 12.5°C", "Sunny, 11.5°C", "Rainy, 8.5°C"
    ],
    "location_name": [
        "Location A", "Location B", "Location C", "Location D", "Location E",
        "Location F", "Location G", "Location H", "Location I", "Location J",
        "Location K", "Location L", "Location M", "Location N", "Location O"
    ]
}
sample_df = pd.DataFrame(sample_data)
print("=== Sample Input Data ===")
print(sample_df)

# Predict traffic volume using the regression model
sample_reg_pred = best_regressor.predict(sample_df)
print("\n=== Regression Predictions (Traffic Volume) ===")
for i, pred in enumerate(sample_reg_pred):
    print(f"Sample {i+1}: Predicted Traffic Volume = {pred:.2f}")

# Predict congestion level using the classification model
sample_clf_pred = best_classifier.predict(sample_df)

# Convert numeric predictions back to labels.
# Adjust the mapping below based on your original LabelEncoder; here we assume:
# 0: "High", 1: "Low", 2: "Medium"
congestion_mapping = {0: "High", 1: "Low", 2: "Medium"}
sample_clf_labels = [congestion_mapping.get(pred, "Unknown") for pred in sample_clf_pred]

print("\n=== Classification Predictions (Congestion Level) ===")
for i, label in enumerate(sample_clf_labels):
    print(f"Sample {i+1}: Predicted Congestion Level = {label}")
