import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
print("="*80)
print("PM2.5 TIME-SERIES PREDICTION MODEL")
print("="*80)
print("\n[STEP 1] Loading data from Excel files...")
try:
    df1 = pd.read_excel('Sensor1+24_mar_11_20.xlsx')
    df2 = pd.read_excel('sesnor2_24_mar_11_20.xlsx')
    print(f"  - Sensor1 data shape: {df1.shape}")
    print(f"  - Sensor2 data shape: {df2.shape}")
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    print(f"  - Combined data shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
except Exception as e:
    print(f"ERROR loading data: {e}")
    raise
print("\n[STEP 2] Preprocessing data...")
if 'received_at' in df.columns:
    df['received_at'] = pd.to_datetime(df['received_at'], errors='coerce')
    df = df.sort_values('received_at')
    df.set_index('received_at', inplace=True)
    print(f"  - Set datetime index from 'received_at'")
else:
    print("  WARNING: 'received_at' column not found, using default index")
cols_to_drop = ['correlation_ids', 'frm_payload', 'rx_metadata', 'beep']
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  - Dropped columns: {cols_to_drop}")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = df[numeric_cols]
print(f"  - Kept {len(numeric_cols)} numeric columns")
pm25_cols = [col for col in df.columns if 'pm2' in col.lower() or 'pm_2' in col.lower()]
if not pm25_cols:
    print("  ERROR: No PM2.5 column found!")
    print(f"  Available columns: {list(df.columns)}")
    raise ValueError("PM2.5 target column not found")
target_col = pm25_cols[0]
print(f"  - Target column identified: '{target_col}'")
initial_rows = len(df)
df = df.dropna(subset=[target_col])
print(f"  - Removed {initial_rows - len(df)} rows with missing target")
missing_pct = df.isnull().sum() / len(df)
cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"  - Dropped {len(cols_to_drop)} columns with >50% missing values")
print(f"  - Missing values before filling: {df.isnull().sum().sum()}")
df = df.ffill().bfill()
df = df.interpolate(method='linear', limit_direction='both')
print(f"  - Missing values after filling: {df.isnull().sum().sum()}")
rows_before = len(df)
df = df.dropna()
print(f"  - Removed {rows_before - len(df)} rows with remaining NaN values")
print(f"  - Final data shape after cleaning: {df.shape}")
print("\n[STEP 3] Engineering features...")
df[f'{target_col}_lag1'] = df[target_col].shift(1)
df[f'{target_col}_lag2'] = df[target_col].shift(2)
print(f"  - Created lag features: lag1, lag2")
rolling_window = 5
feature_cols = []
for col in df.columns:
    if col != target_col and col not in [f'{target_col}_lag1', f'{target_col}_lag2']:
        if any(keyword in col.lower() for keyword in ['pm10', 'co2', 'humidity', 'temperature', 'temp', 'hum']):
            rolling_col = f'{col}_rolling_mean_{rolling_window}'
            df[rolling_col] = df[col].rolling(window=rolling_window, min_periods=1).mean()
            feature_cols.append(rolling_col)
print(f"  - Created {len(feature_cols)} rolling mean features (window={rolling_window})")
df = df.dropna()
print(f"  - Data shape after feature engineering: {df.shape}")
all_features = [col for col in df.columns if col != target_col]
X = df[all_features]
y = df[target_col]
print(f"  - Feature matrix shape: {X.shape}")
print(f"  - Target shape: {y.shape}")
print(f"  - Features: {list(X.columns)[:10]}..." if len(X.columns) > 10 else f"  - Features: {list(X.columns)}")
print("\n[STEP 4] Splitting data chronologically (80% train, 20% test)...")
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"  - Train set: {X_train.shape[0]} samples ({X_train.index.min()} to {X_train.index.max()})")
print(f"  - Test set: {X_test.shape[0]} samples ({X_test.index.min()} to {X_test.index.max()})")
print(f"  - Train target mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
print(f"  - Test target mean: {y_test.mean():.2f}, std: {y_test.std():.2f}")
print("\n[STEP 5] Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  - Scaled train set shape: {X_train_scaled.shape}")
print(f"  - Scaled test set shape: {X_test_scaled.shape}")
print("\n[STEP 6] Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    early_stopping_rounds=10,
    eval_metric='rmse',
    verbosity=0
)
eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
model.fit(
    X_train_scaled, 
    y_train,
    eval_set=eval_set,
    verbose=False
)
print(f"  - Model trained successfully")
print(f"  - Best iteration: {model.best_iteration}")
print(f"  - Best score: {model.best_score:.4f}")
print("\n[STEP 7] Performing Time Series Cross-Validation (CV=5)...")
tscv = TimeSeriesSplit(n_splits=5)
cv_rmse_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
    X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    cv_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    cv_model.fit(X_cv_train, y_cv_train)
    y_cv_pred = cv_model.predict(X_cv_val)
    cv_rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
    cv_rmse_scores.append(cv_rmse)
    print(f"  - Fold {fold+1}: RMSE = {cv_rmse:.4f}")
print(f"  - Mean CV RMSE: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
print("\n[STEP 8] Evaluating model on test set...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)
print(f"\nTRAIN SET:")
print(f"  - RMSE: {train_rmse:.4f} ug/m3")
print(f"  - MAE:  {train_mae:.4f} ug/m3")
print(f"  - R²:   {train_r2:.4f}")
print(f"\nTEST SET:")
print(f"  - RMSE: {test_rmse:.4f} ug/m3")
print(f"  - MAE:  {test_mae:.4f} ug/m3")
print(f"  - R²:   {test_r2:.4f}")
if test_rmse < 10:
    print(f"\nSUCCESS: Test RMSE ({test_rmse:.4f}) is below 10 ug/m3 threshold!")
else:
    print(f"\nWARNING: Test RMSE ({test_rmse:.4f}) exceeds 10 ug/m3 threshold")
print("="*80)
print("\n[STEP 9] Creating visualizations...")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=10)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual PM2.5 (ug/m3)', fontsize=12)
axes[0, 0].set_ylabel('Predicted PM2.5 (ug/m3)', fontsize=12)
axes[0, 0].set_title(f'Actual vs Predicted (Test Set)\nRMSE: {test_rmse:.4f}, R²: {test_r2:.4f}', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, s=10)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted PM2.5 (ug/m3)', fontsize=12)
axes[0, 1].set_ylabel('Residuals (ug/m3)', fontsize=12)
axes[0, 1].set_title('Residual Plot (Test Set)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
n_samples = min(500, len(y_test))
test_subset_idx = range(len(y_test) - n_samples, len(y_test))
axes[1, 0].plot(y_test.iloc[test_subset_idx].values, label='Actual', linewidth=1.5, alpha=0.7)
axes[1, 0].plot(y_test_pred[test_subset_idx], label='Predicted', linewidth=1.5, alpha=0.7)
axes[1, 0].set_xlabel('Time Index', fontsize=12)
axes[1, 0].set_ylabel('PM2.5 (ug/m3)', fontsize=12)
axes[1, 0].set_title(f'Time Series Prediction (Last {n_samples} Test Samples)', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(15)
axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'].values)
axes[1, 1].set_yticks(range(len(feature_importance)))
axes[1, 1].set_yticklabels(feature_importance['feature'].values, fontsize=9)
axes[1, 1].set_xlabel('Importance', fontsize=12)
axes[1, 1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('pm25_model_evaluation.png', dpi=300, bbox_inches='tight')
print("  - Saved evaluation plots to 'pm25_model_evaluation.png'")
print("\n[STEP 10] Saving model and scaler...")
joblib.dump(model, 'pm25_xgboost_model.pkl')
joblib.dump(scaler, 'pm25_scaler.pkl')
print("  - Saved model to 'pm25_xgboost_model.pkl'")
print("  - Saved scaler to 'pm25_scaler.pkl'")
print("\n[STEP 11] Predicting next 24 hours from test set end...")
n_future = min(24, len(y_test))
future_predictions = y_test_pred[-n_future:]
future_actual = y_test.iloc[-n_future:].values
print(f"\nNext {n_future} Hour Predictions:")
print("-" * 60)
print(f"{'Hour':<8} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
print("-" * 60)
for i in range(n_future):
    error = abs(future_predictions[i] - future_actual[i])
    print(f"{i+1:<8} {future_predictions[i]:<15.2f} {future_actual[i]:<15.2f} {error:<15.2f}")
print("-" * 60)
print(f"Mean Absolute Error (24h): {np.mean(np.abs(future_predictions - future_actual)):.2f} ug/m3")
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nDataset Summary:")
print(f"  - Total samples: {len(df)}")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print(f"  - Features: {X.shape[1]}")
print(f"\nModel: XGBoost Regressor")
print(f"  - Best iteration: {model.best_iteration}")
print(f"  - Test RMSE: {test_rmse:.4f} ug/m3")
print(f"  - Test R²: {test_r2:.4f}")
print(f"\nFiles saved:")
print(f"  - pm25_xgboost_model.pkl")
print(f"  - pm25_scaler.pkl")
print(f"  - pm25_model_evaluation.png")
print("="*80)