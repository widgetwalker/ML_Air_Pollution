# Auto-organize PNG files into graphs directory
# Run this after training any model to automatically organize generated PNGs

import os
import shutil
from pathlib import Path

print("="*80)
print("AUTO-ORGANIZING PNG FILES")
print("="*80)

# Ensure graphs directories exist
os.makedirs('graphs/model_evaluations', exist_ok=True)
os.makedirs('graphs/model_comparisons', exist_ok=True)

moved_count = 0

# Find all PNG files in root directory
for file in Path('.').glob('*.png'):
    filename = file.name
    
    # Determine destination based on filename pattern
    if 'evaluation' in filename:
        dest = f'graphs/model_evaluations/{filename}'
        shutil.move(str(file), dest)
        print(f"✓ Moved: {filename} → graphs/model_evaluations/")
        moved_count += 1
    elif 'comparison' in filename:
        dest = f'graphs/model_comparisons/{filename}'
        shutil.move(str(file), dest)
        print(f"✓ Moved: {filename} → graphs/model_comparisons/")
        moved_count += 1
    elif 'correlation' in filename:
        os.makedirs('graphs/correlations', exist_ok=True)
        dest = f'graphs/correlations/{filename}'
        shutil.move(str(file), dest)
        print(f"✓ Moved: {filename} → graphs/correlations/")
        moved_count += 1
    elif 'Sensor' in filename:
        os.makedirs('graphs/sensor_data', exist_ok=True)
        dest = f'graphs/sensor_data/{filename}'
        shutil.move(str(file), dest)
        print(f"✓ Moved: {filename} → graphs/sensor_data/")
        moved_count += 1

if moved_count == 0:
    print("No PNG files found in root directory to organize.")
else:
    print(f"\n✓ Organized {moved_count} PNG file(s)")

print("="*80)
