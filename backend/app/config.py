from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    data_dir: str = "./data"
    db_path: str = "./data/second_brain.db"

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 8
    hybrid_alpha: float = 0.65
    max_context_chars: int = 14000

    llm_provider: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

settings = Settings()
