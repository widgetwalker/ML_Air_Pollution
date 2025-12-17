
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("MULTI-TARGET AIR QUALITY PREDICTION - GRADIENT BOOSTING")
print("="*80)

print("\n[STEP 1] Loading data...")
df1 = pd.read_excel('Sensor1+24_mar_11_20.xlsx')
df2 = pd.read_excel('sesnor2_24_mar_11_20.xlsx')
df = pd.concat([df1, df2], axis=0, ignore_index=True)

print("\n[STEP 2] Preprocessing...")
if 'received_at' in df.columns:
    df['received_at'] = pd.to_datetime(df['received_at'], errors='coerce')
    df = df.sort_values('received_at').set_index('received_at')

cols_to_drop = ['correlation_ids', 'frm_payload', 'rx_metadata', 'beep']
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
df = df.select_dtypes(include=[np.number])

target_mapping = {'pm2_5': 'PM2.5', 'pm10': 'PM10', 'co2': 'CO2', 'tvoc': 'TVOC',
                  'temperature': 'Temperature', 'humidity': 'Humidity', 'pressure': 'Pressure'}
target_columns = {name: [c for c in df.columns if key in c.lower()][0] 
                  for key, name in target_mapping.items() 
                  if any(key in c.lower() for c in df.columns)}

df = df.drop(columns=df.columns[df.isnull().sum() / len(df) > 0.5])
df = df.ffill().bfill().interpolate(method='linear').dropna()

os.makedirs('models_gb', exist_ok=True)
all_results = {}

print("\n[STEP 3] Training Gradient Boosting models...")
for target_name, target_col in target_columns.items():
    print(f"\n{'='*80}\nTRAINING: {target_name}\n{'='*80}")
    
    df_temp = df.copy()
    df_temp[f'{target_col}_lag1'] = df_temp[target_col].shift(1)
    df_temp[f'{target_col}_lag2'] = df_temp[target_col].shift(2)
    
    for col in [c for c in df.columns if c != target_col]:
        if any(k in col.lower() for k in ['pm10', 'pm2', 'co2', 'humidity', 'temp', 'tvoc', 'pressure']):
            df_temp[f'{col}_rolling_mean_5'] = df_temp[col].rolling(5, min_periods=1).mean()
    
    df_temp = df_temp.dropna()
    X = df_temp[[c for c in df_temp.columns if c != target_col]]
    y = df_temp[target_col]
    
    split_idx = int(len(df_temp) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        cv_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        cv_model.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
        cv_pred = cv_model.predict(X_train_scaled[val_idx])
        cv_scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], cv_pred)))
    
    y_test_pred = model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, CV RMSE: {np.mean(cv_scores):.4f}")
    
    all_results[target_name] = {
        'test_rmse': test_rmse, 'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': test_r2, 'cv_rmse': np.mean(cv_scores),
        'y_test': y_test, 'y_test_pred': y_test_pred
    }
    
    joblib.dump(model, f'models_gb/{target_name.lower().replace(".", "")}_model.pkl')
    joblib.dump(scaler, f'models_gb/{target_name.lower().replace(".", "")}_scaler.pkl')

summary_df = pd.DataFrame({
    'Target': list(all_results.keys()),
    'Test RMSE': [all_results[t]['test_rmse'] for t in all_results],
    'Test MAE': [all_results[t]['test_mae'] for t in all_results],
    'Test R2': [all_results[t]['test_r2'] for t in all_results],
    'CV RMSE': [all_results[t]['cv_rmse'] for t in all_results]
})
summary_df.to_csv('models_gb/performance_summary_gb.csv', index=False)

fig, axes = plt.subplots(len(all_results), 2, figsize=(12, 4*len(all_results)))
if len(all_results) == 1:
    axes = axes.reshape(1, -1)

for idx, (name, res) in enumerate(all_results.items()):
    axes[idx, 0].scatter(res['y_test'], res['y_test_pred'], alpha=0.5, s=10)
    axes[idx, 0].plot([res['y_test'].min(), res['y_test'].max()], 
                      [res['y_test'].min(), res['y_test'].max()], 'r--', lw=2)
    axes[idx, 0].set_title(f'{name} - Gradient Boosting (R²: {res["test_r2"]:.3f})')
    axes[idx, 0].set_xlabel('Actual')
    axes[idx, 0].set_ylabel('Predicted')
    axes[idx, 0].grid(True, alpha=0.3)
    
    n = min(200, len(res['y_test']))
    axes[idx, 1].plot(res['y_test'].iloc[-n:].values, label='Actual', alpha=0.7)
    axes[idx, 1].plot(res['y_test_pred'][-n:], label='Predicted', alpha=0.7)
    axes[idx, 1].set_title(f'{name} - Time Series')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('graphs/model_evaluations', exist_ok=True)
plt.savefig('graphs/model_evaluations/evaluation_gradient_boosting.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: graphs/model_evaluations/evaluation_gradient_boosting.png")
print(f"✓ Trained {len(all_results)} Gradient Boosting models")
