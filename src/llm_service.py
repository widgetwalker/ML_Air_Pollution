import hashlib
import json
import requests
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from .config import settings
from .prompt_templates import (
    get_system_prompt,
    get_general_advisory_prompt,
    get_pollutant_focus_prompt,
    get_activity_advice_prompt,
    get_emergency_prompt,
    get_followup_prompt,
    select_appropriate_template,
)

class LLMService:
    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.cache = {}
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            if self.provider == "openai":
                import openai
                api_key = settings.openai_api_key
                if api_key and api_key != "your_openai_api_key_here":
                    self.client = openai.OpenAI(api_key=api_key)
                    print(f"✓ Initialized OpenAI client")
                else:
                    print("⚠ OpenAI API key not configured")
            elif self.provider == "gemini":
                import google.generativeai as genai
                api_key = settings.google_api_key
                if api_key and api_key != "your_gemini_api_key_here":
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel(self.model)
                    print(f"✓ Initialized Gemini client with model: {self.model}")
                else:
                    print("⚠ Gemini API key not configured")
            
            elif self.provider == "anthropic":
                import anthropic
                api_key = settings.anthropic_api_key
                if api_key and api_key != "your_anthropic_api_key_here":
                    self.client = anthropic.Anthropic(api_key=api_key)
                    print(f"✓ Initialized Anthropic client")
                else:
                    print("⚠ Anthropic API key not configured")
            
            elif self.provider == "huggingface":
                api_key = settings.huggingface_api_key
                if api_key and api_key != "your_huggingface_api_key_here":
                    # Using FLAN-T5 which is more reliable on free tier and good for instruction-following
                    # New Hugging Face endpoint format: model-name.huggingface.co
                    self.client = {
                        'api_key': api_key,
                        'api_url': "https://flan-t5-large.huggingface.co"
                    }
                    print(f"✓ Initialized Hugging Face client with model: google/flan-t5-large")
                else:
                    print("⚠ Hugging Face API key not configured")
        
        except ImportError as e:
            print(f"⚠ Could not import {self.provider} library: {e}")
            print("  Run: pip install -r requirements.txt")
        except Exception as e:
            print(f"⚠ Error initializing LLM client: {e}")
    
    def generate_advice(
        self,
        enriched_context: Dict,
        activity: Optional[str] = None,
        custom_question: Optional[str] = None
    ) -> Tuple[str, bool]:
        cache_key = self._generate_cache_key(enriched_context, activity, custom_question)
        if settings.enable_cache and cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(days=settings.cache_expiry_days):
                return cached_data['advice'], True
        
        if self.client is None:
            advice = self._generate_fallback_advice(enriched_context)
            return advice, False
        
        try:
            from .context_enrichment import ContextEnrichment
            context_enricher = ContextEnrichment()
            context_str = context_enricher.format_for_llm(enriched_context)
            
            if custom_question:
                prompt = get_followup_prompt(context_str, custom_question)
            elif activity:
                prompt = get_activity_advice_prompt(activity, context_str)
            else:
                template_type = select_appropriate_template(enriched_context)
                if template_type == 'emergency':
                    prompt = get_emergency_prompt(context_str)
                elif template_type == 'pollutant_focus':
                    primary = enriched_context.get('overall_aqi', {}).get('primary_pollutant', 'PM2.5')
                    prompt = get_pollutant_focus_prompt(primary, context_str)
                else:
                    prompt = get_general_advisory_prompt(context_str)
            
            advice = self._call_llm_api(prompt)
            
            if settings.enable_cache:
                self.cache[cache_key] = {
                    'advice': advice,
                    'timestamp': datetime.now(),
                }
            
            return advice, False
        
        except Exception as e:
            print(f"Error generating advice: {e}")
            advice = self._generate_fallback_advice(enriched_context)
            return advice, False
    
    def _call_llm_api(self, prompt: str) -> str:
        system_prompt = get_system_prompt()
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        
        elif self.provider == "gemini":
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens,
                },
                safety_settings={
                    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
                }
            )
            # Check if response has valid content
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text
            # If no valid content, raise an error with finish reason
            finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            raise Exception(f"Gemini API returned no content. Finish reason: {finish_reason}")
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        
        elif self.provider == "huggingface":
            full_prompt = f"{system_prompt}\n\n{prompt}"
            headers = {"Authorization": f"Bearer {self.client['api_key']}"}
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False
                }
            }
            
            # Retry logic for Hugging Face model loading (410 errors)
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(self.client['api_url'], headers=headers, json=payload, timeout=30)
                    
                    # Handle 410 Gone (model loading)
                    if response.status_code == 410:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            print(f"⏳ Model is loading... Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise Exception("Model is still loading. Please try again in a few moments.")
                    
                    # Handle 503 Service Unavailable
                    if response.status_code == 503:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"⏳ Service temporarily unavailable... Retrying in {wait_time}s")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise Exception("Hugging Face service is currently unavailable.")
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Handle error responses
                    if isinstance(result, dict) and 'error' in result:
                        error_msg = result.get('error', 'Unknown error')
                        if 'loading' in error_msg.lower():
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)
                                print(f"⏳ Model is loading... Retrying in {wait_time}s")
                                time.sleep(wait_time)
                                continue
                        raise Exception(f"Hugging Face API error: {error_msg}")
                    
                    # Extract generated text
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        return result.get('generated_text', result.get('text', ''))
                    else:
                        raise Exception(f"Unexpected Hugging Face API response format: {result}")
                
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        print(f"⏳ Request timeout... Retrying (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("Request timed out. The model may be overloaded.")
                
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1 and '410' in str(e):
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"⏳ Model is loading... Retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    raise

        
        return "LLM provider not configured."
    
    def _generate_cache_key(
        self,
        enriched_context: Dict,
        activity: Optional[str],
        question: Optional[str]
    ) -> str:
        rounded_values = {}
        for pollutant, data in enriched_context.get('current', {}).items():
            value = data.get('value', 0)
            rounded = round(value * 0.95 / 5) * 5
            rounded_values[pollutant] = rounded
        
        cache_data = {
            'values': rounded_values,
            'user': enriched_context.get('user_context', {}),
            'activity': activity,
            'question': question,
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _generate_fallback_advice(self, enriched_context: Dict) -> str:
        from .aqi_standards import get_recommended_actions, AQICategory
        overall = enriched_context.get('overall_aqi', {})
        category_str = overall.get('category', 'Good')
        primary_pollutant = overall.get('primary_pollutant', 'PM2.5')
        
        category_map = {
            'Good': AQICategory.GOOD,
            'Moderate': AQICategory.MODERATE,
            'Unhealthy for Sensitive Groups': AQICategory.UNHEALTHY_SENSITIVE,
            'Unhealthy': AQICategory.UNHEALTHY,
            'Very Unhealthy': AQICategory.VERY_UNHEALTHY,
            'Hazardous': AQICategory.HAZARDOUS,
        }
        category = category_map.get(category_str, AQICategory.GOOD)
        
        user_profile = enriched_context.get('user_context', {})
        is_sensitive = any([
            user_profile.get('respiratory_issues', False),
            user_profile.get('cardiovascular_issues', False),
            user_profile.get('age_group') in ['child', 'senior'],
        ])
        
        actions = get_recommended_actions(category, is_sensitive)
        
        advice = f"**Air Quality Status: {category_str}**\n\n"
        advice += f"Primary Pollutant: {primary_pollutant}\n\n"
        advice += "**Recommended Actions:**\n"
        for i, action in enumerate(actions, 1):
            advice += f"{i}. {action}\n"
        
        advice += "\n*Note: LLM service is currently unavailable. This is rule-based advice.*"
        
        return advice
    
    def get_cache_stats(self) -> Dict:
        return {
            'cache_size': len(self.cache),
            'cache_enabled': settings.enable_cache,
            'cache_expiry_days': settings.cache_expiry_days,
        }