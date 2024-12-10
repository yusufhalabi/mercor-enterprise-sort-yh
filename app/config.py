from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-4"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 150
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Validate settings
if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables or .env file") 