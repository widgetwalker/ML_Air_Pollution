import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .config import MODELS_DIR

class PredictionService:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.target_names = ["PM25", "PM10", "CO2", "TVOC", "Temperature", "Humidity", "Pressure"]
        self.load_models()
    def load_models(self):
        print("Loading trained models...")
        for target in self.target_names:
            model_file = self.models_dir / f"{target.lower()}_model.pkl"
            scaler_file = self.models_dir / f"{target.lower()}_scaler.pkl"
            
            if model_file.exists() and scaler_file.exists():
                try:
                    self.models[target] = joblib.load(model_file)
                    self.scalers[target] = joblib.load(scaler_file)
                    print(f"  ✓ Loaded {target} model")
                except Exception as e:
                    print(f"  ✗ Error loading {target} model: {e}")
            else:
                print(f"  ✗ Model files not found for {target}")
        
        print(f"\nSuccessfully loaded {len(self.models)} models")
    
    def predict(self, sensor_data: Dict[str, float]) -> Dict[str, any]:
        predictions = {
            'timestamp': datetime.now().isoformat(),
            'current': sensor_data.copy(),
            'predicted': {},
            'status': 'success'
        }
        try:
            feature_values = list(sensor_data.values())
            
            for target in self.target_names:
                if target in self.models:
                    try:
                        X = np.array(feature_values).reshape(1, -1)
                        
                        X_scaled = self.scalers[target].transform(X)
                        
                        pred = self.models[target].predict(X_scaled)[0]
                        predictions['predicted'][target] = float(pred)
                    
                    except Exception as e:
                        predictions['predicted'][target] = None
                        print(f"Error predicting {target}: {e}")
        
        except Exception as e:
            predictions['status'] = 'error'
            predictions['error'] = str(e)
        
        return predictions
    
    def predict_simple(self, **kwargs) -> Dict[str, float]:
        sensor_data = {
            'pm25': kwargs.get('pm25', 25.0),
            'pm10': kwargs.get('pm10', 50.0),
            'co2': kwargs.get('co2', 800.0),
            'tvoc': kwargs.get('tvoc', 150.0),
            'temperature': kwargs.get('temperature', 22.0),
            'humidity': kwargs.get('humidity', 50.0),
            'pressure': kwargs.get('pressure', 1013.0),
        }
        result = self.predict(sensor_data)
        return result.get('predicted', {})
    
    def get_model_info(self) -> Dict[str, any]:
        return {
            'loaded_models': list(self.models.keys()),
            'total_models': len(self.models),
            'models_dir': str(self.models_dir),
        }