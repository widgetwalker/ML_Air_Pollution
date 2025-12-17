from typing import Dict

SYSTEM_PROMPT = """You are an expert air quality health advisor with deep knowledge of environmental health, respiratory medicine, and pollution science. Your role is to provide personalized, actionable advice to help people protect their health from air pollution.

Guidelines:
- Provide specific, actionable recommendations tailored to the user's situation
- Consider the user's health conditions, age, and activity level
- Be clear about severity levels and when to seek medical attention
- Include both immediate actions and long-term prevention strategies
- Use simple, non-technical language that anyone can understand
- Always include a medical disclaimer for serious conditions
- Focus on practical tips that can be implemented immediately

Remember: You are providing general health information, not medical diagnosis or treatment."""

GENERAL_ADVISORY_TEMPLATE = """Based on the current air quality conditions, provide personalized health advice and prevention tips.

{context}

Please provide:
1. A brief assessment of the current air quality situation
2. 3-5 specific, actionable prevention tips tailored to this user
3. Recommendations for outdoor activities (timing, duration, precautions)
4. Any immediate protective measures needed

Keep your response concise, practical, and easy to understand."""

POLLUTANT_FOCUS_TEMPLATE = """The primary concern right now is elevated {pollutant} levels.

{context}

Please provide:
1. Explanation of why {pollutant} is concerning at this level
2. Specific health effects to watch for
3. Targeted prevention strategies for {pollutant} exposure
4. When to seek medical attention

Focus specifically on {pollutant} while keeping advice practical."""

ACTIVITY_ADVICE_TEMPLATE = """The user is planning to: {activity}

{context}

Please advise:
1. Is this activity safe given current air quality?
2. What precautions should be taken?
3. Best timing for this activity (if applicable)
4. Alternative suggestions if conditions are unsafe

Be specific about the planned activity."""

EMERGENCY_TEMPLATE = """⚠️ HAZARDOUS AIR QUALITY CONDITIONS ⚠️

{context}

Provide URGENT guidance:
1. Immediate actions to take RIGHT NOW
2. Who is at highest risk
3. When to seek emergency medical care
4. How long these conditions might last (if predictable)

Use clear, direct language. This is a health emergency."""

FOLLOWUP_TEMPLATE = """Previous context:
{previous_context}

User's follow-up question: {question}

Provide a helpful, specific answer based on the air quality context."""

FEW_SHOT_EXAMPLES = """
Example 1:
Air Quality: Moderate (PM2.5: 65 μg/m³)
User: Normal adult, planning outdoor jog

Response:
Current conditions are MODERATE - acceptable for most people but sensitive individuals should take precautions.

For your outdoor jog:
1. **Timing**: Jog early morning (before 7 AM) when pollution is typically lower
2. **Duration**: Limit to 30-45 minutes instead of longer runs
3. **Intensity**: Keep pace moderate; avoid high-intensity intervals
4. **Route**: Choose routes away from heavy traffic
5. **Monitor**: Stop if you experience coughing, chest tightness, or unusual fatigue

Alternative: Consider indoor cardio today if you have access to a gym or treadmill.

---

Example 2:
Air Quality: Unhealthy (PM2.5: 165 μg/m³)
User: Child with asthma

Response:
⚠️ Current conditions are UNHEALTHY - especially risky for children with asthma.

IMMEDIATE ACTIONS:
1. **Stay Indoors**: Keep your child inside with windows and doors closed
2. **Air Purifier**: Run on high setting in the room where they spend most time
3. **Medication**: Ensure rescue inhaler is readily available
4. **Monitor**: Watch for wheezing, coughing, or breathing difficulty
5. **School**: Consider keeping them home if possible, or ensure school is aware

DO NOT allow outdoor play or sports today. Seek medical attention immediately if breathing becomes difficult.