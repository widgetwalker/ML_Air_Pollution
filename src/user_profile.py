import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from .config import DATA_DIR

class UserProfile:
    def __init__(self, profiles_file: Path = None):
        self.profiles_file = profiles_file or (DATA_DIR / "user_profiles.json")
        self.profiles = self._load_profiles()
    def _load_profiles(self) -> Dict:
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading profiles: {e}")
                return {}
        return {}
    def _save_profiles(self):
        try:
            self.profiles_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            print(f"Error saving profiles: {e}")
    def create_profile(
        self,
        profile_id: str,
        age_group: str = "adult",
        gender: str = "not_specified",
        respiratory_issues: bool = False,
        cardiovascular_issues: bool = False,
        allergies: bool = False,
        pregnancy: bool = False,
        activity_level: str = "moderate",
        outdoor_exposure: str = "moderate",
        sensitivity_level: str = "normal",
        **kwargs
    ) -> Dict:
        profile = {
            'profile_id': profile_id,
            'age_group': age_group,
            'gender': gender,
            'respiratory_issues': respiratory_issues,
            'cardiovascular_issues': cardiovascular_issues,
            'allergies': allergies,
            'pregnancy': pregnancy,
            'activity_level': activity_level,
            'outdoor_exposure': outdoor_exposure,
            'sensitivity_level': sensitivity_level,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            **kwargs
        }
        self.profiles[profile_id] = profile
        self._save_profiles()
        return profile
    
    def get_profile(self, profile_id: str) -> Optional[Dict]:
        return self.profiles.get(profile_id)
    def update_profile(self, profile_id: str, **updates) -> Optional[Dict]:
        if profile_id in self.profiles:
            self.profiles[profile_id].update(updates)
            self.profiles[profile_id]['updated_at'] = datetime.now().isoformat()
            self._save_profiles()
            return self.profiles[profile_id]
        return None
    def delete_profile(self, profile_id: str) -> bool:
        if profile_id in self.profiles:
            del self.profiles[profile_id]
            self._save_profiles()
            return True
        return False
    def list_profiles(self) -> List[Dict]:
        return list(self.profiles.values())
    def get_default_profiles(self) -> Dict[str, Dict]:
        return {
            'normal_adult': {
                'profile_id': 'normal_adult',
                'age_group': 'adult',
                'respiratory_issues': False,
                'cardiovascular_issues': False,
                'allergies': False,
                'pregnancy': False,
                'activity_level': 'moderate',
                'outdoor_exposure': 'moderate',
                'sensitivity_level': 'normal',
            },
            'child_with_asthma': {
                'profile_id': 'child_with_asthma',
                'age_group': 'child',
                'respiratory_issues': True,
                'cardiovascular_issues': False,
                'allergies': True,
                'pregnancy': False,
                'activity_level': 'active',
                'outdoor_exposure': 'high',
                'sensitivity_level': 'highly_sensitive',
            },
            'senior_with_heart_condition': {
                'profile_id': 'senior_with_heart_condition',
                'age_group': 'senior',
                'respiratory_issues': False,
                'cardiovascular_issues': True,
                'allergies': False,
                'pregnancy': False,
                'activity_level': 'sedentary',
                'outdoor_exposure': 'minimal',
                'sensitivity_level': 'sensitive',
            },
            'pregnant_woman': {
                'profile_id': 'pregnant_woman',
                'age_group': 'adult',
                'respiratory_issues': False,
                'cardiovascular_issues': False,
                'allergies': False,
                'pregnancy': True,
                'activity_level': 'moderate',
                'outdoor_exposure': 'moderate',
                'sensitivity_level': 'sensitive',
            },
            'athlete': {
                'profile_id': 'athlete',
                'age_group': 'adult',
                'respiratory_issues': False,
                'cardiovascular_issues': False,
                'allergies': False,
                'pregnancy': False,
                'activity_level': 'active',
                'outdoor_exposure': 'high',
                'sensitivity_level': 'normal',
            },
            'office_worker': {
                'profile_id': 'office_worker',
                'age_group': 'adult',
                'respiratory_issues': False,
                'cardiovascular_issues': False,
                'allergies': False,
                'pregnancy': False,
                'activity_level': 'sedentary',
                'outdoor_exposure': 'minimal',
                'sensitivity_level': 'normal',
            },
        }
    def load_default_profiles(self):
        defaults = self.get_default_profiles()
        for profile_id, profile_data in defaults.items():
            if profile_id not in self.profiles:
                self.profiles[profile_id] = profile_data
        self._save_profiles()
        print(f"Loaded {len(defaults)} default profiles")