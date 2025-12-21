"""
Test Model Performance on Individual Sensors
Loads trained models and evaluates accuracy separately on Sensor 1 and Sensor 2 data
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

print("="*80)
print("MODEL PERFORMANCE EVALUATION: SENSOR 1 vs SENSOR 2")
print("="*80)

# Load sensor data
print("\n[STEP 1] Loading sensor data...")
df1 = pd.read_excel('Sensor1+24_mar_11_20.xlsx')
df2 = pd.read_excel('sesnor2_24_mar_11_20.xlsx')

print(f"  - Sensor 1: {df1.shape[0]} samples")
print(f"  - Sensor 2: {df2.shape[0]} samples")

# Quick CO2 comparison
if 'co2' in df1.columns and 'co2' in df2.columns:
    avg_co2_sensor1 = df1['co2'].mean()
    avg_co2_sensor2 = df2['co2'].mean()
    diff = avg_co2_sensor2 - avg_co2_sensor1
    print("\nCO2 Comparison:")
    print(f"  Outside sensor average CO2: {avg_co2_sensor1:.2f}")
    print(f"  Inside sensor average CO2: {avg_co2_sensor2:.2f}")
    print(f"  Difference (inside - outside): {diff:.2f}")
else:
    print("\nCO2 column not found in one of the datasets.")

# Preprocess both sensors
def preprocess_sensor_data(df, sensor_name):
    print(f"\n[STEP 2] Preprocessing {sensor_name} data...")
    
    if 'received_at' in df.columns:
        df['received_at'] = pd.to_datetime(df['received_at'], errors='coerce')
        df = df.sort_values('received_at').set_index('received_at')
    
    cols_to_drop = ['correlation_ids', 'frm_payload', 'rx_metadata', 'beep']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df = df.select_dtypes(include=[np.number])
    
    # Drop columns with too many missing values
    df = df.drop(columns=df.columns[df.isnull().sum() / len(df) > 0.5])
    df = df.ffill().bfill().interpolate(method='linear').dropna()
    
    print(f"  - {sensor_name} preprocessed shape: {df.shape}")
    return df

df1_clean = preprocess_sensor_data(df1, "Sensor 1")
df2_clean = preprocess_sensor_data(df2, "Sensor 2")

# Target mapping
target_mapping = {
    'pm2_5': 'PM2.5',
    'pm10': 'PM10',
    'co2': 'CO2',
    'tvoc': 'TVOC',
    'temperature': 'Temperature',
    'humidity': 'Humidity',
    'pressure': 'Pressure'
}

# Find target columns
target_columns = {}
for key, name in target_mapping.items():
    matching_cols = [c for c in df1_clean.columns if key in c.lower()]
    if matching_cols:
        target_columns[name] = matching_cols[0]

print(f"\n[STEP 3] Found {len(target_columns)} target pollutants")

# Models to test
models_to_test = {
    'Ridge Regression': 'models_ridge',
    'Linear Regression': 'models_lr',
    'XGBoost': 'models',
    'Gradient Boosting': 'models_gb'
}

# Results storage
all_results = {}

print("\n" + "="*80)
print("TESTING MODELS ON INDIVIDUAL SENSORS")
print("="*80)

for model_name, model_dir in models_to_test.items():
    if not os.path.exists(model_dir):
        print(f"\n⚠ Skipping {model_name} - directory not found: {model_dir}")
        continue
    
    print(f"\n{'='*80}")
    print(f"TESTING: {model_name}")
    print(f"{'='*80}")
    
    model_results = {
        'Sensor 1': {},
        'Sensor 2': {}
    }
    
    for target_name, target_col in target_columns.items():
        model_file = f'{model_dir}/{target_name.lower().replace(".", "")}_model.pkl'
        scaler_file = f'{model_dir}/{target_name.lower().replace(".", "")}_scaler.pkl'
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            continue
        
        # Load model and scaler
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        # Test on Sensor 1
        df_temp1 = df1_clean.copy()
        df_temp1[f'{target_col}_lag1'] = df_temp1[target_col].shift(1)
        df_temp1[f'{target_col}_lag2'] = df_temp1[target_col].shift(2)
        
        for col in [c for c in df1_clean.columns if c != target_col]:
            if any(k in col.lower() for k in ['pm10', 'pm2', 'co2', 'humidity', 'temp', 'tvoc', 'pressure']):
                df_temp1[f'{col}_rolling_mean_5'] = df_temp1[col].rolling(5, min_periods=1).mean()
        
        df_temp1 = df_temp1.dropna()
        X1 = df_temp1[[c for c in df_temp1.columns if c != target_col]]
        y1 = df_temp1[target_col]
        
        # Align features with scaler's expected features
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
        if expected_features is not None:
            # Add missing columns with zeros
            for feat in expected_features:
                if feat not in X1.columns:
                    X1[feat] = 0
            # Keep only expected features in correct order
            X1 = X1[expected_features]
        
        X1_scaled = scaler.transform(X1)
        y1_pred = model.predict(X1_scaled)
        
        r2_s1 = r2_score(y1, y1_pred)
        rmse_s1 = np.sqrt(mean_squared_error(y1, y1_pred))
        mae_s1 = mean_absolute_error(y1, y1_pred)
        
        # Test on Sensor 2
        df_temp2 = df2_clean.copy()
        df_temp2[f'{target_col}_lag1'] = df_temp2[target_col].shift(1)
        df_temp2[f'{target_col}_lag2'] = df_temp2[target_col].shift(2)
        
        for col in [c for c in df2_clean.columns if c != target_col]:
            if any(k in col.lower() for k in ['pm10', 'pm2', 'co2', 'humidity', 'temp', 'tvoc', 'pressure']):
                df_temp2[f'{col}_rolling_mean_5'] = df_temp2[col].rolling(5, min_periods=1).mean()
        
        df_temp2 = df_temp2.dropna()
        X2 = df_temp2[[c for c in df_temp2.columns if c != target_col]]
        y2 = df_temp2[target_col]
        
        # Align features with scaler's expected features
        if expected_features is not None:
            # Add missing columns with zeros
            for feat in expected_features:
                if feat not in X2.columns:
                    X2[feat] = 0
            # Keep only expected features in correct order
            X2 = X2[expected_features]
        
        X2_scaled = scaler.transform(X2)
        y2_pred = model.predict(X2_scaled)
        
        r2_s2 = r2_score(y2, y2_pred)
        rmse_s2 = np.sqrt(mean_squared_error(y2, y2_pred))
        mae_s2 = mean_absolute_error(y2, y2_pred)
        
        # Store results
        model_results['Sensor 1'][target_name] = {
            'R2': r2_s1,
            'RMSE': rmse_s1,
            'MAE': mae_s1,
            'Accuracy': max(0, r2_s1 * 100) if r2_s1 >= 0 else None
        }
        
        model_results['Sensor 2'][target_name] = {
            'R2': r2_s2,
            'RMSE': rmse_s2,
            'MAE': mae_s2,
            'Accuracy': max(0, r2_s2 * 100) if r2_s2 >= 0 else None
        }
        
        print(f"\n{target_name}:")
        print(f"  Sensor 1: R²={r2_s1:.3f}, RMSE={rmse_s1:.2f}, Accuracy={model_results['Sensor 1'][target_name]['Accuracy']:.1f}%" if r2_s1 >= 0 else f"  Sensor 1: R²={r2_s1:.3f}, RMSE={rmse_s1:.2f}, Accuracy=N/A")
        print(f"  Sensor 2: R²={r2_s2:.3f}, RMSE={rmse_s2:.2f}, Accuracy={model_results['Sensor 2'][target_name]['Accuracy']:.1f}%" if r2_s2 >= 0 else f"  Sensor 2: R²={r2_s2:.3f}, RMSE={rmse_s2:.2f}, Accuracy=N/A")
        
        # Show difference
        if r2_s1 >= 0 and r2_s2 >= 0:
            diff = abs(r2_s1 - r2_s2) * 100
            better = "Sensor 1" if r2_s1 > r2_s2 else "Sensor 2"
            print(f"  → {better} performs better by {diff:.1f}%")
    
    all_results[model_name] = model_results

# Summary comparison
print("\n" + "="*80)
print("SUMMARY: SENSOR COMPARISON")
print("="*80)

for model_name, results in all_results.items():
    print(f"\n{model_name}:")
    print("-" * 60)
    
    s1_accuracies = [r['Accuracy'] for r in results['Sensor 1'].values() if r['Accuracy'] is not None]
    s2_accuracies = [r['Accuracy'] for r in results['Sensor 2'].values() if r['Accuracy'] is not None]
    
    if s1_accuracies and s2_accuracies:
        avg_s1 = np.mean(s1_accuracies)
        avg_s2 = np.mean(s2_accuracies)
        
        print(f"  Sensor 1 Average Accuracy: {avg_s1:.1f}%")
        print(f"  Sensor 2 Average Accuracy: {avg_s2:.1f}%")
        print(f"  Difference: {abs(avg_s1 - avg_s2):.1f}%")
        
        if avg_s1 > avg_s2:
            print(f"  → Sensor 1 performs better overall")
        elif avg_s2 > avg_s1:
            print(f"  → Sensor 2 performs better overall")
        else:
            print(f"  → Both sensors perform equally")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
