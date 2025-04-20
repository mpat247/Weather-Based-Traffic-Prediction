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

def loadData(path):
    """Read CSV into a DataFrame."""
    return pd.read_csv(path)

def encodeTarget(series):
    """Encode categorical target labels."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(series)
    return encoded, encoder

def splitData(df, features, y_reg, y_clf, test_size=0.3, eval_size=0.5, random_state=42):
    """Split into train, eval, and test sets."""
    train_idx, temp_idx = train_test_split(
        df.index, test_size=test_size, stratify=y_clf, random_state=random_state
    )
    eval_idx, test_idx = train_test_split(
        temp_idx, test_size=eval_size, stratify=y_clf[temp_idx], random_state=random_state
    )
    X = df[features]
    return {
        'X_train': X.loc[train_idx], 'X_eval': X.loc[eval_idx], 'X_test': X.loc[test_idx],
        'y_reg_train': y_reg.loc[train_idx], 'y_reg_eval': y_reg.loc[eval_idx], 'y_reg_test': y_reg.loc[test_idx],
        'y_clf_train': y_clf[train_idx], 'y_clf_eval': y_clf[eval_idx], 'y_clf_test': y_clf[test_idx]
    }

def buildPreprocessor(numerical_features, categorical_features):
    """Create a transformer for numeric and categorical data."""
    num_pipe = Pipeline([('scale', StandardScaler())])
    cat_pipe = OneHotEncoder(handle_unknown='ignore')
    return ColumnTransformer([
        ('num', num_pipe, numerical_features),
        ('cat', cat_pipe, categorical_features)
    ])

def buildRegressor():
    """Build a regressor with log-target transform."""
    base = XGBRegressor(random_state=42, objective='reg:squarederror')
    return TransformedTargetRegressor(
        regressor=base,
        func=np.log1p,
        inverse_func=np.expm1
    )

def buildPipelines(preprocessor):
    """Assemble training pipelines for regression and classification."""
    reg_pipeline = Pipeline([
        ('pre', preprocessor),
        ('regressor', buildRegressor())
    ])
    clf_pipeline = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(random_state=42, eval_metric='mlogloss'))
    ])
    return reg_pipeline, clf_pipeline

def tuneModel(pipeline, param_grid, X_train, y_train, cv=3, scoring=None, n_jobs=-1):
    """Perform grid search to find best hyperparameters."""
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring=scoring, n_jobs=n_jobs, verbose=2
    )
    start = time.time()
    grid.fit(X_train, y_train)
    print(f"Tuning completed in {time.time() - start:.2f}s")
    return grid

def evaluateModel(model, X, y_true, metrics, prefix):
    """Compute and store evaluation metrics."""
    preds = model.predict(X)
    if prefix.startswith('reg'):
        metrics[f'{prefix}_mae'] = mean_absolute_error(y_true, preds)
        metrics[f'{prefix}_rmse'] = math.sqrt(mean_squared_error(y_true, preds))
    else:
        metrics[f'{prefix}_acc'] = accuracy_score(y_true, preds)
        metrics[f'{prefix}_f1'] = f1_score(y_true, preds, average='weighted')
    return preds

def saveMetrics(metrics, path):
    """Write metrics dict to a JSON file."""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    df = loadData('final_congestion_dataset.csv')

    features = [
        'lat', 'lon', 'temp_c', 'wind_speed',
        'hour', 'day_of_week', 'month',
        'weather_summary', 'location_name'
    ]
    y_reg = df['total_traffic_volume']
    y_clf_raw = df['congestion_level']
    y_clf, label_encoder = encodeTarget(y_clf_raw)

    splits = splitData(df, features, y_reg, y_clf)

    num_feats = ['lat', 'lon', 'temp_c', 'wind_speed', 'hour', 'month']
    cat_feats = ['day_of_week', 'weather_summary', 'location_name']
    preprocessor = buildPreprocessor(num_feats, cat_feats)

    reg_pipe, clf_pipe = buildPipelines(preprocessor)

    param_grid_reg = {
        'regressor__regressor__max_depth': [3, 5, 7],
        'regressor__regressor__n_estimators': [50, 100, 150],
        'regressor__regressor__learning_rate': [0.01, 0.1, 0.2]
    }
    param_grid_clf = {
        'classifier__max_depth': [3, 5, 7],
        'classifier__n_estimators': [50, 100, 150],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }

    grid_reg = tuneModel(
        reg_pipe, param_grid_reg,
        splits['X_train'], splits['y_reg_train'],
        scoring='neg_mean_absolute_error'
    )
    grid_clf = tuneModel(
        clf_pipe, param_grid_clf,
        splits['X_train'], splits['y_clf_train'],
        scoring='accuracy', n_jobs=1
    )

    metrics = {}
    evaluateModel(grid_reg, splits['X_eval'], splits['y_reg_eval'], metrics, 'reg_eval')
    evaluateModel(grid_clf, splits['X_eval'], splits['y_clf_eval'], metrics, 'clf_eval')
    evaluateModel(grid_reg, splits['X_test'], splits['y_reg_test'], metrics, 'reg_test')
    evaluateModel(grid_clf, splits['X_test'], splits['y_clf_test'], metrics, 'clf_test')

    saveMetrics(metrics, 'final_metrics.json')

    dump(grid_reg.best_estimator_, 'tuned_xgb_regressor.joblib')
    dump(grid_clf.best_estimator_, 'tuned_xgb_classifier.joblib')

    # Demo prediction
    demo = pd.DataFrame({
        'lat': [43.67], 'lon': [-79.45], 'temp_c': [10.0],
        'wind_speed': [2.5], 'hour': [15], 'day_of_week': ['Friday'],
        'month': [7], 'weather_summary': ['Sunny,10.0°C'],
        'location_name': ['Test']
    })
    print('Reg pred:', grid_reg.predict(demo)[0])
    print('Clf pred:', label_encoder.inverse_transform(grid_clf.predict(demo))[0])

if __name__ == '__main__':
    main()
