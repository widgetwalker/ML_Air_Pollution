from typing import Dict, List, Tuple
from datetime import datetime
from .aqi_standards import (
    get_aqi_category, 
    get_health_impact, 
    get_recommended_actions,
    get_pollutant_health_effects,
    AQICategory
)

class ContextEnrichment:
    def __init__(self):
        pass
    def enrich_predictions(
        self, 
        current_values: Dict[str, float],
        predicted_values: Dict[str, float],
        user_profile: Dict[str, any] = None
    ) -> Dict[str, any]:
        enriched = {
            'timestamp': datetime.now().isoformat(),
            'current': {},
            'predicted': {},
            'overall_aqi': {},
            'trends': {},
            'user_context': user_profile or {},
            'temporal_context': self._get_temporal_context(),
        }
        for pollutant, value in current_values.items():
            category, color = get_aqi_category(pollutant, value)
            enriched['current'][pollutant] = {
                'value': value,
                'category': category.value,
                'color': color,
                'health_impact': get_health_impact(category),
            }
        
        for pollutant, value in predicted_values.items():
            if value is not None:
                category, color = get_aqi_category(pollutant, value)
                enriched['predicted'][pollutant] = {
                    'value': value,
                    'category': category.value,
                    'color': color,
                    'health_impact': get_health_impact(category),
                }
        
        enriched['overall_aqi'] = self._calculate_overall_aqi(enriched['current'])
        
        enriched['trends'] = self._analyze_trends(current_values, predicted_values)
        
        return enriched
    
    def _calculate_overall_aqi(self, current_data: Dict) -> Dict[str, any]:
        category_priority = {
            'Good': 0,
            'Moderate': 1,
            'Unhealthy for Sensitive Groups': 2,
            'Unhealthy': 3,
            'Very Unhealthy': 4,
            'Hazardous': 5,
        }
        worst_category = 'Good'
        worst_pollutant = None
        worst_priority = -1
        
        for pollutant, data in current_data.items():
            category = data.get('category', 'Good')
            priority = category_priority.get(category, 0)
            
            if priority > worst_priority:
                worst_priority = priority
                worst_category = category
                worst_pollutant = pollutant
        
        return {
            'category': worst_category,
            'primary_pollutant': worst_pollutant,
            'color': current_data.get(worst_pollutant, {}).get('color', 'green'),
        }
    
    def _analyze_trends(
        self, 
        current: Dict[str, float], 
        predicted: Dict[str, float]
    ) -> Dict[str, str]:
        trends = {}
        for pollutant in current.keys():
            pollutant_key = pollutant.upper()
            if pollutant_key in predicted and predicted[pollutant_key] is not None:
                current_val = current[pollutant]
                predicted_val = predicted[pollutant_key]
                
                if current_val > 0:
                    change_pct = ((predicted_val - current_val) / current_val) * 100
                    
                    if change_pct > 10:
                        trends[pollutant] = f"worsening (↑{change_pct:.1f}%)"
                    elif change_pct < -10:
                        trends[pollutant] = f"improving (↓{abs(change_pct):.1f}%)"
                    else:
                        trends[pollutant] = "stable"
                else:
                    trends[pollutant] = "stable"
        
        return trends
    
    def _get_temporal_context(self) -> Dict[str, str]:
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        return {
            'time_of_day': time_of_day,
            'day_of_week': now.strftime('%A'),
            'date': now.strftime('%Y-%m-%d'),
            'hour': hour,
        }
    
    def get_user_sensitivity_level(self, user_profile: Dict) -> bool:
        if not user_profile:
            return False
        sensitive_conditions = [
            user_profile.get('respiratory_issues', False),
            user_profile.get('cardiovascular_issues', False),
            user_profile.get('allergies', False),
            user_profile.get('pregnancy', False),
            user_profile.get('age_group') in ['child', 'senior'],
            user_profile.get('sensitivity_level') in ['sensitive', 'highly_sensitive'],
        ]
        
        return any(sensitive_conditions)
    
    def format_for_llm(self, enriched_data: Dict) -> str:
        lines = []
        overall = enriched_data['overall_aqi']
        lines.append(f"Overall Air Quality: {overall['category']}")
        lines.append(f"Primary Pollutant: {overall['primary_pollutant']}")
        lines.append("")
        
        lines.append("Current Pollutant Levels:")
        for pollutant, data in enriched_data['current'].items():
            lines.append(f"  - {pollutant.upper()}: {data['value']:.1f} ({data['category']})")
        lines.append("")
        
        if enriched_data['trends']:
            lines.append("Predicted Trends:")
            for pollutant, trend in enriched_data['trends'].items():
                lines.append(f"  - {pollutant.upper()}: {trend}")
            lines.append("")
        
        temporal = enriched_data['temporal_context']
        lines.append(f"Time: {temporal['time_of_day']} ({temporal['day_of_week']})")
        lines.append("")
        
        if enriched_data['user_context']:
            user = enriched_data['user_context']
            lines.append("User Profile:")
            if 'age_group' in user:
                lines.append(f"  - Age Group: {user['age_group']}")
            if 'activity_level' in user:
                lines.append(f"  - Activity Level: {user['activity_level']}")
            if any(user.get(k, False) for k in ['respiratory_issues', 'cardiovascular_issues', 'allergies']):
                lines.append("  - Health Conditions: Yes (sensitive group)")
        
        return "\n".join(lines)
