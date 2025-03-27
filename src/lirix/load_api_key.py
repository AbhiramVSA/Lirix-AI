from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=True)

class Settings(BaseSettings):
    GROQ_API_KEY: str
    TAVILY_API_KEY: str
    SUPABASE_URL: str = ""
    SUPABASE_PASSWORD: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# Check if working

#print(f"Your Groq API key is: {settings.GROQ_API_KEY}")
#print(f"Your Tavily API key is: {settings.TAVILY_API_KEY}")


