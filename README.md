# Weather-Based Traffic Prediction

A machine learning project that predicts traffic volume and congestion levels in Toronto using weather data and historical traffic counts.

## Project Overview

This project combines Toronto traffic count data with weather data to:
- **Predict traffic volume** (regression) at specific locations
- **Classify congestion levels** (Low, Medium, High) based on traffic and weather conditions
- **Visualize results** on an interactive web map with heat maps and collision hot spots

## Data Sources

- **TMC (Turning Movement Count) Data** – Toronto traffic counts (2010–2019)
- **Environment Canada Weather Data** – Hourly weather observations
- **ERA5 Weather Data** – Temperature, precipitation, wind components
- **Toronto Traffic Collisions Data** – Collision records with coordinates and fatality info
- **KSI (Killed or Seriously Injured) Data** – Motor vehicle collision data

## Models

### XGBoost Regressor
- Predicts traffic volume using log-transformed target
- Features: lat, lon, temp_c, wind_speed, hour, day_of_week, month, weather_summary, location_name
- Test MAE: 52.57, RMSE: 62.26

### XGBoost Classifier
- Classifies congestion as Low, Medium, or High
- Uses SMOTE for class imbalance
- Test Accuracy: 49.57%, F1 Score: 0.329

Additional models trained (in `scripts/trainNew.py`):
- RandomForest Regressor & Classifier
- Keras MLP Regressor

## Project Structure

```
├── app.py                        # Flask web application
├── training.py                   # Model training with GridSearchCV
├── preprocess.ipynb              # Data preprocessing notebook
├── test.py                       # Model testing script
├── requirements.txt              # Python dependencies
├── final_congestion_dataset.csv  # Processed dataset for training
├── final_metrics.json            # Model evaluation metrics
├── tuned_xgb_regressor.joblib    # Trained regression model (root)
├── tuned_xgb_classifier.joblib   # Trained classification model (root)
├── tmc_raw_data_2010_2019.csv    # TMC traffic data
├── hourly_final.csv              # Env Canada weather data
├── Traffic_Collisions_Toronto_data.csv  # Collision records
├── Motor Vehicle Collisions with KSI Data - 4326.csv  # KSI data
├── models/
│   ├── xgb_regressor.joblib      # XGBoost regressor (for app.py)
│   ├── xgb_classifier.joblib     # XGBoost classifier (for app.py)
│   └── metrics_new.json          # Additional model metrics
├── scripts/
│   ├── trainNew.py               # Extended training script
│   ├── updatedPreprocess.py      # Alternative preprocessing
│   └── training.py               # Training script copy
├── templates/
│   ├── index.html                # Web form for predictions
│   └── result.html               # Results display page
```

## Features Used

| Feature | Description |
|---------|-------------|
| lat, lon | Location coordinates |
| temp_c | Temperature in Celsius |
| wind_speed | Wind speed (m/s) |
| hour | Hour of day (0–23) |
| day_of_week | Day name (Monday–Sunday) |
| month | Month (1–12) |
| weather_summary | Weather description (Sunny/Rainy/Snowy) |
| location_name | Street intersection name |
| precip_flag | Precipitation indicator (0/1) |
| hour_sin, hour_cos | Cyclical encoding of hour |
| month_sin, month_cos | Cyclical encoding of month |

## Web Application

The Flask app (`app.py`) provides:
- **Single-point prediction** – Enter location and weather to get traffic volume and congestion level
- **Traffic heat map** – Hour-selectable visualization of city-wide traffic
- **Collision hot spots** – Configurable radius and severity filter for nearby collisions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the web app
```bash
python app.py
```
Then open http://localhost:5000 in your browser.

### Train models
```bash
python training.py
```

### Test predictions
```bash
python test.py
```

## Requirements

Key dependencies:
- pandas, numpy
- scikit-learn, xgboost, imbalanced-learn
- flask, folium, geopy
- joblib

See `requirements.txt` for full list.
