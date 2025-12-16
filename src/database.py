import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

from .config import DATA_DIR

class Database:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or (DATA_DIR / "air_quality.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pm25 REAL,
                pm10 REAL,
                co2 REAL,
                tvoc REAL,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                aqi_category TEXT,
                primary_pollutant TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS advice_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conditions_hash TEXT UNIQUE NOT NULL,
                advice_text TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_accessed TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TEXT NOT NULL,
                user_message TEXT,
                assistant_response TEXT,
                context_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def save_prediction(self, prediction_data: Dict):
        cursor = self.conn.cursor()
        current = prediction_data.get('current', {})
        overall_aqi = prediction_data.get('overall_aqi', {})
        
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, pm25, pm10, co2, tvoc, temperature, humidity, pressure,
                aqi_category, primary_pollutant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_data.get('timestamp'),
            current.get('pm25'),
            current.get('pm10'),
            current.get('co2'),
            current.get('tvoc'),
            current.get('temperature'),
            current.get('humidity'),
            current.get('pressure'),
            overall_aqi.get('category'),
            overall_aqi.get('primary_pollutant')
        ))
        self.conn.commit()
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def save_conversation(self, user_id: str, user_message: str, 
                         assistant_response: str, context_data: Dict = None):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_history (
                user_id, timestamp, user_message, assistant_response, context_data
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            datetime.now().isoformat(),
            user_message,
            assistant_response,
            json.dumps(context_data) if context_data else None
        ))
        self.conn.commit()
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM conversation_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_cache(self, days: int = 30):
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("""
            DELETE FROM advice_cache
            WHERE created_at < ?
        """, (cutoff_date,))
        self.conn.commit()
    
    def close(self):
        if self.conn:
            self.conn.close()