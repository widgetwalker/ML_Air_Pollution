from typing import Dict, Tuple
from enum import Enum

class AQICategory(Enum):
    GOOD = "Good"
    MODERATE = "Moderate"
    UNHEALTHY_SENSITIVE = "Unhealthy for Sensitive Groups"
    UNHEALTHY = "Unhealthy"
    VERY_UNHEALTHY = "Very Unhealthy"
    HAZARDOUS = "Hazardous"
class AQIColor(Enum):
    GOOD = "green"
    MODERATE = "yellow"
    UNHEALTHY_SENSITIVE = "orange"
    UNHEALTHY = "red"
    VERY_UNHEALTHY = "purple"
    HAZARDOUS = "maroon"
PM25_BREAKPOINTS = [
    (0, 12.0, AQICategory.GOOD),
    (12.1, 35.4, AQICategory.MODERATE),
    (35.5, 55.4, AQICategory.UNHEALTHY_SENSITIVE),
    (55.5, 150.4, AQICategory.UNHEALTHY),
    (150.5, 250.4, AQICategory.VERY_UNHEALTHY),
    (250.5, float('inf'), AQICategory.HAZARDOUS),
]

PM10_BREAKPOINTS = [
    (0, 54, AQICategory.GOOD),
    (55, 154, AQICategory.MODERATE),
    (155, 254, AQICategory.UNHEALTHY_SENSITIVE),
    (255, 354, AQICategory.UNHEALTHY),
    (355, 424, AQICategory.VERY_UNHEALTHY),
    (425, float('inf'), AQICategory.HAZARDOUS),
]

CO2_BREAKPOINTS = [
    (0, 1000, AQICategory.GOOD),
    (1001, 2000, AQICategory.MODERATE),
    (2001, 5000, AQICategory.UNHEALTHY_SENSITIVE),
    (5001, float('inf'), AQICategory.UNHEALTHY),
]

TVOC_BREAKPOINTS = [
    (0, 220, AQICategory.GOOD),
    (221, 660, AQICategory.MODERATE),
    (661, 2200, AQICategory.UNHEALTHY_SENSITIVE),
    (2201, float('inf'), AQICategory.UNHEALTHY),
]

def get_aqi_category(pollutant: str, value: float) -> Tuple[AQICategory, str]:
    pollutant = pollutant.lower().replace(".", "").replace("_", "")
    if pollutant == "pm25":
        breakpoints = PM25_BREAKPOINTS
    elif pollutant == "pm10":
        breakpoints = PM10_BREAKPOINTS
    elif pollutant == "co2":
        breakpoints = CO2_BREAKPOINTS
    elif pollutant == "tvoc":
        breakpoints = TVOC_BREAKPOINTS
    else:
        return AQICategory.GOOD, AQIColor.GOOD.value
    
    for min_val, max_val, category in breakpoints:
        if min_val <= value <= max_val:
            color = AQIColor[category.name].value
            return category, color
    
    return AQICategory.HAZARDOUS, AQIColor.HAZARDOUS.value

def get_health_impact(category: AQICategory) -> str:
    impacts = {
        AQICategory.GOOD: "Air quality is satisfactory, and air pollution poses little or no risk.",
        AQICategory.MODERATE: "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.",
        AQICategory.UNHEALTHY_SENSITIVE: "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        AQICategory.UNHEALTHY: "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.",
        AQICategory.VERY_UNHEALTHY: "Health alert: The risk of health effects is increased for everyone.",
        AQICategory.HAZARDOUS: "Health warning of emergency conditions: everyone is more likely to be affected.",
    }
    return impacts.get(category, "Unknown health impact")
def get_recommended_actions(category: AQICategory, user_sensitive: bool = False) -> list:
    if category == AQICategory.GOOD:
        return [
            "It's a great day to be active outside!",
            "Enjoy outdoor activities without restrictions.",
        ]
    elif category == AQICategory.MODERATE:
        if user_sensitive:
            return [
                "Consider reducing prolonged or heavy outdoor exertion.",
                "Watch for symptoms such as coughing or shortness of breath.",
                "Take more breaks during outdoor activities.",
            ]
        else:
            return [
                "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
                "Generally safe for outdoor activities.",
            ]
    elif category == AQICategory.UNHEALTHY_SENSITIVE:
        if user_sensitive:
            return [
                "Reduce prolonged or heavy outdoor exertion.",
                "Take more breaks during outdoor activities.",
                "Consider moving activities indoors.",
                "Watch for symptoms and reduce activity if they occur.",
            ]
        else:
            return [
                "Sensitive groups should reduce prolonged outdoor exertion.",
                "General public can continue normal activities.",
                "Consider shorter outdoor activities.",
            ]
    elif category == AQICategory.UNHEALTHY:
        if user_sensitive:
            return [
                "Avoid prolonged or heavy outdoor exertion.",
                "Move activities indoors or reschedule.",
                "Keep windows closed to avoid dirty outdoor air.",
                "Run an air purifier if available.",
            ]
        else:
            return [
                "Reduce prolonged or heavy outdoor exertion.",
                "Take more breaks during outdoor activities.",
                "Sensitive groups should avoid outdoor activities.",
                "Consider wearing a mask (N95 or KN95) if going outside.",
            ]
    elif category == AQICategory.VERY_UNHEALTHY:
        return [
            "Avoid all outdoor physical activities.",
            "Move activities indoors.",
            "Keep windows and doors closed.",
            "Run air purifiers on high.",
            "Wear a high-quality mask (N95/KN95) if you must go outside.",
            "Sensitive groups should remain indoors.",
        ]
    else:
        return [
            "STAY INDOORS and keep windows closed.",
            "Avoid all outdoor activities.",
            "Run air purifiers continuously.",
            "Use N95/KN95 masks if you must go outside.",
            "Seek medical attention if experiencing symptoms.",
            "Consider relocating to a cleaner air environment if possible.",
        ]
def get_pollutant_health_effects(pollutant: str) -> str:
    effects = {
        "pm25": "Fine particulate matter can penetrate deep into lungs and bloodstream, causing respiratory and cardiovascular problems.",
        "pm10": "Coarse particulate matter can irritate airways and worsen asthma and heart disease.",
        "co2": "High CO2 levels can cause headaches, dizziness, restlessness, and difficulty breathing.",
        "tvoc": "Volatile organic compounds can cause eye, nose, and throat irritation, headaches, and nausea.",
        "temperature": "Extreme temperatures can affect comfort and exacerbate health conditions.",
        "humidity": "High humidity can promote mold growth and worsen respiratory conditions; low humidity can cause dry skin and respiratory irritation.",
        "pressure": "Atmospheric pressure changes can trigger headaches and joint pain in sensitive individuals.",
    }
    pollutant = pollutant.lower().replace(".", "").replace("_", "")
    return effects.get(pollutant, "May affect health and comfort.")