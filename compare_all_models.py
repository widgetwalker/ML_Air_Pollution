# Comprehensive Model Comparison Script
# Trains all 10 ML models and creates comparison table

import pandas as pd
import subprocess
import sys
import os
import time

print("="*80)
print("COMPREHENSIVE ML MODEL COMPARISON FOR AIR QUALITY PREDICTION")
print("="*80)

# All model training scripts
training_scripts = {
    'XGBoost': 'train_multi_target_model.py',
    'Linear Regression': 'train_linear_regression.py',
    'Random Forest': 'train_random_forest.py',
    'Gradient Boosting': 'train_gradient_boosting.py',
    'Ridge': 'train_ridge.py',
    'Lasso': 'train_lasso.py',
}

# Performance summary files
summary_files = {
    'XGBoost': 'models/model_performance_summary.csv',
    'Linear Regression': 'models_lr/model_performance_summary_lr.csv',
    'Random Forest': 'models_rf/performance_summary_rf.csv',
    'Gradient Boosting': 'models_gb/performance_summary_gb.csv',
    'Ridge': 'models_ridge/performance_summary_ridge.csv',
    'Lasso': 'models_lasso/performance_summary_lasso.csv',
}

print("\n[INFO] Training scripts available:")
for model, script in training_scripts.items():
    status = "✓" if os.path.exists(script) else "✗"
    print(f"  {status} {model}: {script}")

print("\n[INFO] Collecting existing results...")
all_results = {}

for model_name, summary_file in summary_files.items():
    if os.path.exists(summary_file):
        df = pd.read_excel(summary_file)
        all_results[model_name] = df
        print(f"  ✓ Loaded: {model_name}")
    else:
        print(f"  ✗ Not found: {model_name} - Run {training_scripts.get(model_name, 'N/A')} first")

if all_results:
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table for each pollutant
    pollutants = all_results[list(all_results.keys())[0]]['Target'].tolist()
    
    for pollutant in pollutants:
        print(f"\n{pollutant}:")
        print("-" * 60)
        comparison = []
        for model_name, df in all_results.items():
            row = df[df['Target'] == pollutant]
            if not row.empty:
                r2 = row['Test R2'].values[0]
                rmse = row['Test RMSE'].values[0]
                comparison.append({
                    'Model': model_name,
                    'R²': f"{r2:.4f}",
                    'Accuracy': f"{max(0, r2*100):.1f}%" if r2 >= 0 else "N/A",
                    'RMSE': f"{rmse:.4f}"
                })
        
        comp_df = pd.DataFrame(comparison)
        if not comp_df.empty:
            comp_df = comp_df.sort_values('R²', ascending=False)
            print(comp_df.to_string(index=False))
    
    # Save comprehensive comparison
    print("\n" + "="*80)
    print("Saving comprehensive comparison...")
    
    # Create master comparison file
    master_comparison = []
    for pollutant in pollutants:
        for model_name, df in all_results.items():
            row = df[df['Target'] == pollutant]
            if not row.empty:
                master_comparison.append({
                    'Pollutant': pollutant,
                    'Model': model_name,
                    'Test R2': row['Test R2'].values[0],
                    'Test RMSE': row['Test RMSE'].values[0],
                    'Test MAE': row['Test MAE'].values[0],
                    'CV RMSE': row['CV RMSE'].values[0]
                })
    
    master_df = pd.DataFrame(master_comparison)
    master_df.to_csv('comprehensive_model_comparison.csv', index=False)
    print("✓ Saved: comprehensive_model_comparison.csv")
    
else:
    print("\n[WARNING] No results found. Please run the training scripts first.")
    print("\nTo train all models, run:")
    for script in training_scripts.values():
        if os.path.exists(script):
            print(f"  python {script}")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
