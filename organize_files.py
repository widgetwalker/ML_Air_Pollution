import os
import shutil
from pathlib import Path

print("="*80)
print("ORGANIZING PROJECT FILES AND GRAPHS")
print("="*80)

# Create graphs directory structure
print("\n[STEP 1] Creating directory structure...")
dirs = [
    'graphs',
    'graphs/sensor_data',
    'graphs/correlations',
    'graphs/model_evaluations',
    'graphs/model_comparisons'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"  ✓ Created: {dir_path}")

# Move PNG files to appropriate directories
print("\n[STEP 2] Moving PNG files...")

# Sensor data graphs
sensor_files = [
    ('Sensor1_co2_graph.png', 'graphs/sensor_data/'),
    ('Sensor1_tvoc_graph.png', 'graphs/sensor_data/'),
    ('Sensor2_co2_graph.png', 'graphs/sensor_data/'),
    ('Sensor2_tvoc_graph.png', 'graphs/sensor_data/'),
]

for src, dest_dir in sensor_files:
    if os.path.exists(src):
        shutil.move(src, os.path.join(dest_dir, src))
        print(f"  ✓ Moved: {src} → {dest_dir}")

# Correlation graphs
correlation_files = [
    'correlation_co2_sensor1_vs_sensor2.png',
    'correlation_humidity_sensor1_vs_sensor2.png',
    'correlation_pm10_sensor1_vs_sensor2.png',
    'correlation_pm2_5_sensor1_vs_sensor2.png',
    'correlation_temperature_sensor1_vs_sensor2.png',
    'correlation_tvoc_sensor1_vs_sensor2.png',
]

for file in correlation_files:
    if os.path.exists(file):
        shutil.move(file, f'graphs/correlations/{file}')
        print(f"  ✓ Moved: {file} → graphs/correlations/")

# Model evaluation graphs
eval_files = [
    ('multi_target_evaluation.png', 'xgboost_evaluation.png'),
    ('multi_target_evaluation_lr.png', 'linear_regression_evaluation.png'),
]

for src, dest in eval_files:
    if os.path.exists(src):
        shutil.move(src, f'graphs/model_evaluations/{dest}')
        print(f"  ✓ Moved: {src} → graphs/model_evaluations/{dest}")

# Model comparison graphs
comp_files = [
    ('model_comparison.png', 'xgboost_comparison.png'),
    ('model_comparison_lr.png', 'linear_regression_comparison.png'),
]

for src, dest in comp_files:
    if os.path.exists(src):
        shutil.move(src, f'graphs/model_comparisons/{dest}')
        print(f"  ✓ Moved: {src} → graphs/model_comparisons/{dest}")

# Remove unwanted files
print("\n[STEP 3] Removing unwanted files...")
unwanted_files = [
    'test_gemini.py',
    'test_gemini_quick.py',
    'test_hf_direct.py',
    'test_system.py',
    'create_all_model_scripts.py',
    'remove_comments.py',
    'switch.py',
    'graph.py',
    'Multi_Target_Walkthrough.md',
]

for file in unwanted_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"  ✓ Removed: {file}")

print("\n" + "="*80)
print("ORGANIZATION COMPLETE!")
print("="*80)

# Summary
print("\nDirectory structure:")
for root, dirs, files in os.walk('graphs'):
    level = root.replace('graphs', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')
