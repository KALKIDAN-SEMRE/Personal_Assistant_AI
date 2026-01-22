"""
Configuration management for Personal AI Assistant.
Uses environment variables with sensible defaults.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Configuration
    app_name: str = "Personal AI Assistant"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8000
    
    # AI/LLM Configuration
    llm_provider: str = "ollama"  # mock, openai, ollama
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Ollama Configuration (local LLM)
    ollama_model: str = "llama3"
    ollama_url: str = "http://localhost:11434/api/generate"

    
    # Database Configuration
    database_url: str = "sqlite:///./assistant.db"
    
    # System Personality
    system_personality: str = "You are a helpful, friendly, and intelligent personal assistant."
    
    # Memory Configuration
    max_conversation_history: int = 10  # Maximum messages to keep per session
    
    # Semantic Memory Configuration
    embedding_provider: str = "mock"  # mock, openai, ollama
    embedding_dimension: int = 384  # Embedding vector dimension
    openai_embedding_model: str = "text-embedding-3-small"
    semantic_memory_min_similarity: float = 0.3  # Minimum similarity for retrieval
    semantic_memory_max_retrieved: int = 5  # Max memories to retrieve per query


# Global settings instance
settings = Settings()
