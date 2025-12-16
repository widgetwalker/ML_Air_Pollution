import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    models_dir: Path = Path(os.getenv("MODELS_DIR", "./models"))
    database_path: Path = Path(os.getenv("DATABASE_PATH", "./data/air_quality.db"))
    
    enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    cache_expiry_days: int = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))
    
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "500"))
    
    default_user_profile: str = os.getenv("DEFAULT_USER_PROFILE", "normal_adult")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def validate_api_key(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")
        elif self.llm_provider == "gemini":
            return bool(self.google_api_key and self.google_api_key != "your_gemini_api_key_here")
        elif self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key_here")
        elif self.llm_provider == "huggingface":
            return bool(self.huggingface_api_key and self.huggingface_api_key != "your_huggingface_api_key_here")
        return False
    def get_api_key(self) -> Optional[str]:
        if self.llm_provider == "openai":
            return self.openai_api_key
        elif self.llm_provider == "gemini":
            return self.google_api_key
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key
        elif self.llm_provider == "huggingface":
            return self.huggingface_api_key
        return None

settings = Settings()

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
